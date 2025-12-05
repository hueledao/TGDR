import torch
import torch.nn as nn
import scipy.io
from einops import rearrange
import torch.nn.functional as F
import torch.fft as fft
# from diffusers_local.src.models.embeddings import GaussianFourierProjection, TextTimeEmbedding, TimestepEmbedding, Timesteps
# from diffusers_local.src.models.unet_2d_condition import UNet2DConditionModel
# from utils.TTSR_utils import MeanShift
# from torchvision import models
# from utils import MainNet, LTE, SearchTransfer
from Network.spade_file import SPADE
# from guided_diffusion.unet import UNetModel, UNetModel_TimeAwareEncoder
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from guided_diffusion.script_util_x0_modified import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_model,
    create_time_aware_enc,
)
from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
def Torchtensor2Array(input):
    channel, height, width = input.size()
    output = torch.reshape(input, (channel, height, width))

    # Convert to numpy array hei x wid x 3
    HDR = output.cpu().detach().numpy()
    HDR = HDR.swapaxes(0, 2).swapaxes(0, 1)

    return HDR
    

class Latent_Transformation(nn.Module):
    def __init__(self, in_size = 8, out_size = 8):
        super(Latent_Transformation, self).__init__()
        self.num_subspace = 4
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_block = UNetConvBlock(in_size, out_size)
        self.subnet = Subspace(in_size, self.num_subspace)

    def forward(self, z_guided, z_t):
        out =  torch.cat([z_guided, z_t], 1)
        b_, c_, h_, w_ = z_t.shape #B, 8, 32, 32
        sub = self.subnet(out) 
        V_t = sub.reshape(b_, self.num_subspace, h_*w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        # V = V_t.transpose(0, 2, 1)
        V = rearrange(V_t, 'b d c -> b c d')

        mat = torch.matmul(V_t, V)
        # mat_inv = torch.matinv(mat)
        mat_inv = torch.inverse(mat)

        project_mat = torch.matmul(mat_inv, V_t)
        z_t_ = z_t.reshape(b_, c_, h_*w_)
        #Projecting z_t
        z_t_ = rearrange(z_t_, 'b c d -> b d c')
        project_feature_z_t = torch.matmul(project_mat, z_t_)
        z_hat_t = torch.matmul(V, project_feature_z_t)
        z_hat_t = rearrange(z_hat_t, 'b d c -> b c d').reshape(b_, c_, h_, w_)

        # return z_hat_guided, z_hat_t
        return z_hat_t
    
class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope=0.2):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),#half() if fp16
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))#half() if fp16
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        return out
 
        
class Subspace(nn.Module):

    def __init__(self, in_size=8, out_size=8):
        super(Subspace, self).__init__()
        self.blocks = []
        self.blocks.append(UNetConvBlock(in_size, out_size, 0.2).cuda())
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc

    
# depthwise-separable convolution (DSC)
class DSC(nn.Module):

    def __init__(self, nin: int) -> None:
        super(DSC, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out
    
# Efficient Feature Fusion(EFF)
class EFF(nn.Module):
    def __init__(self, nin: int, nout: int, num_splits= 4) -> None:
        super(EFF, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.num_splits = num_splits
        self.subspaces = nn.ModuleList(
            [DSC(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sub_feat = torch.chunk(x, self.num_splits, dim=1)
        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))
        out = torch.cat(out, dim=1)

        return out   

class Latent_Fusion(nn.Module):

    def __init__(self):
        super(Latent_Fusion, self).__init__()
        
    def forward(self, z1, z2):
        z_fused = z1
        return z_fused
    

class Content_Adaptor_SubSpace(nn.Module):

    def __init__(self):
        super(Content_Adaptor_SubSpace, self).__init__()
        self.conv_in_bg = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.conv_in_roi = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.conv_in_t = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.trans = Latent_Transformation(in_size=32, out_size=32)
        self.latent_fusion = EFF(16, 16)
        self.conv_out = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
    def forward(self, z_ROI, z_BG, z_t):
        ########### Transformation transforms inputs into a new subspace
        z_ROI_t_tilde = self.trans(self.conv_in_roi(z_ROI), self.conv_in_t(z_t))
        z_BG_t_tilde = self.trans(self.conv_in_bg(z_BG), self.conv_in_t(z_t))

        ########### Enhancement by learning similarity map
        # z_ROI_t_enhanced = self.latent_enhancement(z_t, z_ROI_tilde, z_ROI_t_tilde)
        # z_BG_t_enhanced = self.latent_enhancement(z_t, z_BG_tilde, z_BG_t_tilde)

        ########### Fusion
        z_t_eff = self.latent_fusion(z_ROI_t_tilde+z_BG_t_tilde)
        z_t_hat = self.conv_out(z_t_eff)

        return z_t_hat      
        # return self.conv_in_bg(z_BG), self.conv_in_roi(z_ROI), self.conv_in_t(z_t),  z_BG_t_tilde, z_ROI_t_tilde, z_t_eff


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out
    
class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class FTB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FTB, self).__init__()
        self.FTB_scale_conv0 = nn.Conv2d(in_channel, in_channel, 1)
        self.FTB_scale_conv1 = nn.Conv2d(in_channel, out_channel, 1)
        self.FTB_shift_conv0 = nn.Conv2d(in_channel, in_channel, 1)
        self.FTB_shift_conv1 = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x, affine=True, skip=False):
        if affine:
            # x[0]: fea; x[1]: cond
            scale = self.FTB_scale_conv1(F.leaky_relu(self.FTB_scale_conv0(x[1]), 0.1, inplace=True))
            shift = self.FTB_shift_conv1(F.leaky_relu(self.FTB_shift_conv0(x[1]), 0.1, inplace=True))
            if skip:
                return x[0] * (scale + 1) + shift + x[0]  # + x[0] in case x[1] is clean (=0)
            else:
                return x[0] * (scale + 1) + shift
        else:
            return x[0]
        
class CAB(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CAB, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out = torch.cat([out0, out2], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = torch.sigmoid(out)
        w0, w2 = torch.chunk(out, 2, dim=1)
        x = x0 * w0 + x2 * w2

        return x
    


class Dual_UNet_TTAFE(nn.Module):#using all feature scale from 256 to 8, UNetModel_TimeAwareEncoder, replace resblock by transformer blocks as BFRffusion

    def __init__(self, args):
        super(Dual_UNet_TTAFE, self).__init__()

        self.unet, _ = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        self.time_aware_enc = create_time_aware_enc(
            **args_to_dict(args, model_and_diffusion_defaults().keys()), transformers = True, text_guided = True,
        )
        self.spade_256 = SPADE(256, 128).apply(convert_module_to_f16)
        self.conv_256 = nn.Conv2d(256, 256, kernel_size=3, padding=1).apply(convert_module_to_f16)
        
        self.spade_128 = SPADE(256, 128).apply(convert_module_to_f16)
        self.conv_128 = nn.Conv2d(256, 256, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_64 = SPADE(512, 256).apply(convert_module_to_f16)
        self.conv_64 = nn.Conv2d(512, 512, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_32 = SPADE(512, 256).apply(convert_module_to_f16)
        self.conv_32 = nn.Conv2d(512, 512, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_16 = SPADE(1024, 512).apply(convert_module_to_f16)
        self.conv_16 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_8 = SPADE(1024, 512).apply(convert_module_to_f16)
        self.conv_8 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1).apply(convert_module_to_f16)


        self.spade_256_dec = SPADE(256, 128).apply(convert_module_to_f16)
        self.conv_256_dec = nn.Conv2d(256, 256, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_128_dec = SPADE(256, 128).apply(convert_module_to_f16)
        self.conv_128_dec = nn.Conv2d(256, 256, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_64_dec = SPADE(512, 256).apply(convert_module_to_f16)
        self.conv_64_dec = nn.Conv2d(512, 512, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_32_dec = SPADE(512, 256).apply(convert_module_to_f16)
        self.conv_32_dec = nn.Conv2d(512, 512, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_16_dec = SPADE(1024, 512).apply(convert_module_to_f16)
        self.conv_16_dec = nn.Conv2d(1024, 1024, kernel_size=3, padding=1).apply(convert_module_to_f16)
        self.spade_8_dec = SPADE(1024, 512).apply(convert_module_to_f16)
        self.conv_8_dec = nn.Conv2d(1024, 1024, kernel_size=3, padding=1).apply(convert_module_to_f16)
        #18 layers input_blocks
        # torch.Size([1, 3, 256, 256])
        # torch.Size([1, 256, 256, 256])
        # torch.Size([1, 256, 256, 256])#2
        # torch.Size([1, 256, 128, 128])
        # torch.Size([1, 256, 128, 128])
        # torch.Size([1, 256, 128, 128])#5
        # torch.Size([1, 256, 64, 64])
        # torch.Size([1, 512, 64, 64])
        # torch.Size([1, 512, 64, 64])#8
        # torch.Size([1, 512, 32, 32])
        # torch.Size([1, 512, 32, 32])
        # torch.Size([1, 512, 32, 32])#11
        # torch.Size([1, 512, 16, 16])
        # torch.Size([1, 512, 16, 16])
        # torch.Size([1, 512, 16, 16])#14
        # torch.Size([1, 1024, 8, 8])
        # torch.Size([1, 1024, 8, 8])
        # torch.Size([1, 1024, 8, 8])
        # torch.Size([1, 1024, 8, 8])#17

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.unet.convert_to_fp16()
        self.time_aware_enc.convert_to_fp16()

    ####################### without detach()
    def forward(self, x, x_t, timesteps, is_spade = True, emb_text = None):
        # print(emb_text)
        feat_TA_enc = self.time_aware_enc(x, timesteps, emb_text)
        # print('feat_TA_enc',len(feat_TA_enc))
        hs = []
        # with torch.no_grad():
        emb = self.unet.time_embed(timestep_embedding(timesteps, 256))#timesteps = tensor[999]

        # with torch.no_grad():
        h = x_t.type(self.unet.dtype)
        input_block_idx = 0

        for module in self.unet.input_blocks[0:2]:#0,1
            h = module(h, emb)
            hs.append(h)
        #2; torch.Size([1, 256, 256, 256])
        module = self.unet.input_blocks[2]
        h = module(h, emb)#torch.Size([1, 256, 256, 256])
        if is_spade:
            h_res = self.spade_256(h, feat_TA_enc[0])
            h = self.conv_256(h) + h_res
        hs.append(h)

        # with torch.no_grad():
        for module in self.unet.input_blocks[3:5]:#3,4
            h = module(h, emb)
            hs.append(h)
        #5
        module = self.unet.input_blocks[5]
        h = module(h, emb)#torch.Size([1, 256, 128, 128])
        if is_spade:
            h_res = self.spade_128(h, feat_TA_enc[1])
            h = self.conv_128(h) + h_res
        hs.append(h)

        # with torch.no_grad():
        for module in self.unet.input_blocks[6:8]:#6,7
            h = module(h, emb)
            hs.append(h)
        #8
        module = self.unet.input_blocks[8]
        h = module(h, emb)#torch.Size([1, 512, 64, 64])
        if is_spade:
            h_res = self.spade_64(h, feat_TA_enc[2])
            h = self.conv_64(h) + h_res
        hs.append(h)

        # with torch.no_grad():
        for module in self.unet.input_blocks[9:11]:#9,10
            h = module(h, emb)
            hs.append(h)
        #11
        module = self.unet.input_blocks[11]
        h = module(h, emb)#torch.Size([1, 512, 32, 32])
        if is_spade:
            h_res = self.spade_32(h, feat_TA_enc[3])
            h = self.conv_32(h) + h_res
        hs.append(h)

        for module in self.unet.input_blocks[12:14]:#12,13
            h = module(h, emb)
            hs.append(h)
        #14
        module = self.unet.input_blocks[14]
        h = module(h, emb)#torch.Size([1, 512, 32, 32])
        if is_spade:
            h_res = self.spade_16(h, feat_TA_enc[4])
            h = self.conv_16(h) + h_res
        hs.append(h)

        for module in self.unet.input_blocks[15:17]:#12,13
            h = module(h, emb)
            hs.append(h)
        #17
        module = self.unet.input_blocks[17]
        h = module(h, emb)#torch.Size([1, 512, 32, 32])
        if is_spade:
            h_res = self.spade_8(h, feat_TA_enc[5])
            h = self.conv_8(h) + h_res
        hs.append(h)


        # for module in self.unet.input_blocks[12:18]:#12,13,14,15,16,17
        #     h = module(h, emb)
        #     hs.append(h)

        # with torch.no_grad():
        h = self.unet.middle_block(h, emb)
            # for module in self.unet.output_blocks[0:18]:#0,1
            #     h = torch.cat([h, hs.pop()], dim=1)
            #     h = module(h, emb)
            # module = self.unet.output_blocks[6]
            # h = torch.cat([h, hs.pop()], dim=1)
            # h = module(h, emb)#torch.Size([1, 256, 256, 256])
        # h_res = self.spade_32_dec(h, feat_TA_enc[3])
        # h = self.conv_32_dec(h) + h_res
        ############ DECODER
        module = self.unet.output_blocks[0]
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb)#torch.Size([1, 256, 256, 256])
        if is_spade:
            h_res = self.spade_8_dec(h, feat_TA_enc[5])
            h = self.conv_8_dec(h) + h_res

        for module in self.unet.output_blocks[1:3]:#1,2
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        module = self.unet.output_blocks[3]
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb)#torch.Size([1, 256, 256, 256])
        if is_spade:
            h_res = self.spade_16_dec(h, feat_TA_enc[4])
            h = self.conv_16_dec(h) + h_res

        for module in self.unet.output_blocks[4:6]:#4,5
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        module = self.unet.output_blocks[6]
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb)#torch.Size([1, 256, 256, 256])
        if is_spade:
            h_res = self.spade_32_dec(h, feat_TA_enc[3])
            h = self.conv_32_dec(h) + h_res

        # with torch.no_grad():
        for module in self.unet.output_blocks[7:9]:#0,1
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        module = self.unet.output_blocks[9]
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb)
        if is_spade:
            h_res = self.spade_64_dec(h, feat_TA_enc[2])
            h = self.conv_64_dec(h) + h_res

        # with torch.no_grad():
        for module in self.unet.output_blocks[10:12]:#6,7
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        module = self.unet.output_blocks[12]
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb)
        if is_spade:
            h_res = self.spade_128_dec(h, feat_TA_enc[1])
            h = self.conv_128_dec(h) + h_res

        # with torch.no_grad():
        for module in self.unet.output_blocks[13:15]:#9,10
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        module = self.unet.output_blocks[15]
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, emb)
        if is_spade:
            h_res = self.spade_256_dec(h, feat_TA_enc[0])
            h = self.conv_256_dec(h) + h_res

        # with torch.no_grad():
        for module in self.unet.output_blocks[16:18]:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        h = h.type(x_t.dtype)

        # print("h.requires_grad_", h.requires_grad)

        output_unet = self.unet.out(h)
        
        # if is_required_training:
        #     output_unet = output_unet.clone().requires_grad_(True)
        # print("gradddddddddddddddddddd", output_unet.requires_grad)

        ######## TRAINING 
        # mean, log_variance = torch.split(output_unet, x.shape[1], dim=1)
        # nonzero_mask = (
        #     (timesteps != 0).float().view(-1, *([1] * (len(h.shape) - 1)))
        # )  # no noise when t == 0
        # # Gaussianp_mean_variance_(model_output, x_t, t)
        # sample = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
        # return sample
        return output_unet

    #FEATURE VISUALIZATION
    # def forward(self, x, x_t, timesteps, is_spade = True):
    #     # save_path = '/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/feature_maps/spade/'
    #     feat_TA_enc = self.time_aware_enc(x, timesteps)
    #     # print('feat_TA_enc',len(feat_TA_enc))
    #     hs = []
    #     # with torch.no_grad():
    #     emb = self.unet.time_embed(timestep_embedding(timesteps, 256))#timesteps = tensor[999]

    #     # with torch.no_grad():
    #     h = x_t.type(self.unet.dtype)
    #     input_block_idx = 0

    #     for module in self.unet.input_blocks[0:2]:#0,1
    #         h = module(h, emb)
    #         hs.append(h)
    #     #2; torch.Size([1, 256, 256, 256])
    #     module = self.unet.input_blocks[2]
    #     h = module(h, emb)#torch.Size([1, 256, 256, 256])
    #     # FEATURE 256
    #     # h256_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h256_{str(timesteps.item()).zfill(6)}.mat', mdict={'h256': h256_arr}) 
    #     if is_spade:
    #         h_res = self.spade_256(h, feat_TA_enc[0])
    #         # fp256_arr = Torchtensor2Array(torch.mean(feat_TA_enc[0], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp256_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp256': fp256_arr}) 
    #         # h256_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h256_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h256_spade': h256_spade_arr}) 
    #         h = self.conv_256(h) + h_res
    #         # h256_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h256_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h256_hat': h256_hat_arr}) 
    #     hs.append(h)
        

    #     # with torch.no_grad():
    #     for module in self.unet.input_blocks[3:5]:#3,4
    #         h = module(h, emb)
    #         hs.append(h)
    #     #5
    #     module = self.unet.input_blocks[5]
    #     h = module(h, emb)#torch.Size([1, 256, 128, 128])
    #     # FEATURE 128
    #     # h128_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h128_{str(timesteps.item()).zfill(6)}.mat', mdict={'h128': h128_arr}) 
    #     if is_spade:
    #         h_res = self.spade_128(h, feat_TA_enc[1])
    #         # fp128_arr = Torchtensor2Array(torch.mean(feat_TA_enc[1], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp128_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp128': fp128_arr}) 
    #         # h128_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h128_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h128_spade': h128_spade_arr}) 
            
    #         h = self.conv_128(h) + h_res
    #         # h128_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h128_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h128_hat': h128_hat_arr}) 
    #     hs.append(h)

    #     # with torch.no_grad():
    #     for module in self.unet.input_blocks[6:8]:#6,7
    #         h = module(h, emb)
    #         hs.append(h)
    #     #8
    #     module = self.unet.input_blocks[8]
    #     h = module(h, emb)#torch.Size([1, 512, 64, 64])
    #     # FEATURE 64
    #     # h64_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h64_{str(timesteps.item()).zfill(6)}.mat', mdict={'h64': h64_arr}) 
    #     if is_spade:
    #         h_res = self.spade_64(h, feat_TA_enc[2])
    #         # fp64_arr = Torchtensor2Array(torch.mean(feat_TA_enc[2], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp64_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp64': fp64_arr})
    #         # h64_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h64_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h64_spade': h64_spade_arr}) 
            
    #         h = self.conv_64(h) + h_res
    #         # h64_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h64_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h64_hat': h64_hat_arr}) 
    #     hs.append(h)

    #     # with torch.no_grad():
    #     for module in self.unet.input_blocks[9:11]:#9,10
    #         h = module(h, emb)
    #         hs.append(h)
    #     #11
    #     module = self.unet.input_blocks[11]
    #     h = module(h, emb)#torch.Size([1, 512, 32, 32])
    #             # FEATURE 32
    #     # h32_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h32_{str(timesteps.item()).zfill(6)}.mat', mdict={'h32': h32_arr}) 
    #     if is_spade:
    #         h_res = self.spade_32(h, feat_TA_enc[3])
    #         # fp32_arr = Torchtensor2Array(torch.mean(feat_TA_enc[3], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp32_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp32': fp32_arr})
    #         # h32_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h32_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h32_spade': h32_spade_arr}) 
            
    #         h = self.conv_32(h) + h_res
    #         # h32_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h32_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h32_hat': h32_hat_arr}) 
    #     hs.append(h)

    #     for module in self.unet.input_blocks[12:14]:#12,13
    #         h = module(h, emb)
    #         hs.append(h)
    #     #14
    #     module = self.unet.input_blocks[14]
    #     h = module(h, emb)#torch.Size([1, 1028, 16, 16])
    #             # FEATURE 16
    #     # h16_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h16_{str(timesteps.item()).zfill(6)}.mat', mdict={'h16': h16_arr}) 
    #     if is_spade:
    #         h_res = self.spade_16(h, feat_TA_enc[4])
    #         # fp16_arr = Torchtensor2Array(torch.mean(feat_TA_enc[4], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp16_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp16': fp16_arr})
    #         # h16_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h16_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h16_spade': h16_spade_arr}) 
            
    #         h = self.conv_16(h) + h_res
    #         # h16_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h16_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h16_hat': h16_hat_arr}) 
    #     hs.append(h)

    #     for module in self.unet.input_blocks[15:17]:#12,13
    #         h = module(h, emb)
    #         hs.append(h)
    #     #17
    #     module = self.unet.input_blocks[17]
    #     h = module(h, emb)#torch.Size([1, 1028, 8, 8])
    #                     # FEATURE 8
    #     # h8_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h8_{str(timesteps.item()).zfill(6)}.mat', mdict={'h8': h8_arr}) 
    #     if is_spade:
    #         h_res = self.spade_8(h, feat_TA_enc[5])
    #         # fp8_arr = Torchtensor2Array(torch.mean(feat_TA_enc[5], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp8_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp8': fp8_arr})
    #         # h8_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h8_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h8_spade': h8_spade_arr}) 
            
    #         h = self.conv_8(h) + h_res
    #         # h8_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h8_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h8_hat': h8_hat_arr}) 
    #     hs.append(h)
    #     h = self.unet.middle_block(h, emb)
    #     ############ DECODER
    #     module = self.unet.output_blocks[0]
    #     h = torch.cat([h, hs.pop()], dim=1)
    #     h = module(h, emb)#torch.Size([1, 256, 256, 256])
    #      # FEATURE 8
    #     # h8_dec_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h8_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'h8_dec': h8_dec_arr}) 
    #     if is_spade:
    #         h_res = self.spade_8_dec(h, feat_TA_enc[5])
    #         # fp8_dec_arr = Torchtensor2Array(torch.mean(feat_TA_enc[5], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp8_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp8_dec': fp8_dec_arr})
    #         # h8_dec_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h8_dec_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h8_dec_spade': h8_dec_spade_arr}) 

    #         h = self.conv_8_dec(h) + h_res
    #         # h8_dec_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h8_dec_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h8_dec_hat': h8_dec_hat_arr}) 
        
    #     for module in self.unet.output_blocks[1:3]:#1,2
    #         h = torch.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb)
    #     module = self.unet.output_blocks[3]
    #     h = torch.cat([h, hs.pop()], dim=1)
    #     h = module(h, emb)#torch.Size([1, 256, 256, 256])
    #      # FEATURE 16
    #     # h16_dec_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h16_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'h16_dec': h16_dec_arr}) 
    #     if is_spade:
    #         h_res = self.spade_16_dec(h, feat_TA_enc[4])
    #         # fp16_dec_arr = Torchtensor2Array(torch.mean(feat_TA_enc[4], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp16_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp16_dec': fp16_dec_arr})
    #         # h16_dec_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h16_dec_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h16_dec_spade': h16_dec_spade_arr}) 

    #         h = self.conv_16_dec(h) + h_res
    #         # h16_dec_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h16_dec_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h16_dec_hat': h16_dec_hat_arr}) 
        

    #     for module in self.unet.output_blocks[4:6]:#4,5
    #         h = torch.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb)
    #     module = self.unet.output_blocks[6]
    #     h = torch.cat([h, hs.pop()], dim=1)
    #     h = module(h, emb)#torch.Size([1, 256, 256, 256])
    #     # FEATURE 32
    #     # h32_dec_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h32_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'h32_dec': h32_dec_arr}) 
    #     if is_spade:
    #         h_res = self.spade_32_dec(h, feat_TA_enc[3])
    #         # fp32_dec_arr = Torchtensor2Array(torch.mean(feat_TA_enc[3], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp32_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp32_dec': fp32_dec_arr})
    #         # h32_dec_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h32_dec_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h32_dec_spade': h32_dec_spade_arr}) 

    #         h = self.conv_32_dec(h) + h_res
    #         # h32_dec_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h32_dec_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h32_dec_hat': h32_dec_hat_arr}) 
        
    #     # with torch.no_grad():
    #     for module in self.unet.output_blocks[7:9]:#0,1
    #         h = torch.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb)
    #     module = self.unet.output_blocks[9]
    #     h = torch.cat([h, hs.pop()], dim=1)
    #     h = module(h, emb)
    #     # FEATURE 64
    #     # h64_dec_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h64_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'h64_dec': h64_dec_arr}) 
    #     if is_spade:
    #         h_res = self.spade_64_dec(h, feat_TA_enc[2])
    #         # fp64_dec_arr = Torchtensor2Array(torch.mean(feat_TA_enc[2], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp64_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp64_dec': fp64_dec_arr})
    #         # h64_dec_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h64_dec_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h64_dec_spade': h64_dec_spade_arr}) 

    #         h = self.conv_64_dec(h) + h_res
    #         # h64_dec_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h64_dec_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h64_dec_hat': h64_dec_hat_arr}) 
        

    #     # with torch.no_grad():
    #     for module in self.unet.output_blocks[10:12]:#6,7
    #         h = torch.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb)
    #     module = self.unet.output_blocks[12]
    #     h = torch.cat([h, hs.pop()], dim=1)
    #     h = module(h, emb)
    #     # FEATURE 128
    #     # h128_dec_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h128_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'h128_dec': h128_dec_arr}) 
    #     if is_spade:
    #         h_res = self.spade_128_dec(h, feat_TA_enc[1])
    #         # fp128_dec_arr = Torchtensor2Array(torch.mean(feat_TA_enc[1], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp128_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp128_dec': fp128_dec_arr})
    #         # h128_dec_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h128_dec_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h128_dec_spade': h128_dec_spade_arr}) 

    #         h = self.conv_128_dec(h) + h_res
    #         # h128_dec_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h128_dec_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h128_dec_hat': h128_dec_hat_arr}) 
        
    #     # with torch.no_grad():
    #     for module in self.unet.output_blocks[13:15]:#9,10
    #         h = torch.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb)
    #     module = self.unet.output_blocks[15]
    #     h = torch.cat([h, hs.pop()], dim=1)
    #     h = module(h, emb)
    #     # FEATURE 256
    #     # h256_dec_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #     # scipy.io.savemat(save_path + f'h256_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'h256_dec': h256_dec_arr}) 

    #     if is_spade:
    #         h_res = self.spade_256_dec(h, feat_TA_enc[0])
    #         # fp256_dec_arr = Torchtensor2Array(torch.mean(feat_TA_enc[0], dim= 1))
    #         # scipy.io.savemat(save_path + f'fp256_dec_{str(timesteps.item()).zfill(6)}.mat', mdict={'fp256_dec': fp256_dec_arr})
    #         # h256_dec_spade_arr = Torchtensor2Array(torch.mean(h_res, dim= 1))
    #         # scipy.io.savemat(save_path + f'h256_dec_spade_{str(timesteps.item()).zfill(6)}.mat', mdict={'h256_dec_spade': h256_dec_spade_arr}) 

    #         h = self.conv_256_dec(h) + h_res
    #         # h256_dec_hat_arr = Torchtensor2Array(torch.mean(h, dim= 1))
    #         # scipy.io.savemat(save_path + f'h256_dec_hat_{str(timesteps.item()).zfill(6)}.mat', mdict={'h256_dec_hat': h256_dec_hat_arr}) 
        

    #     # with torch.no_grad():
    #     for module in self.unet.output_blocks[16:18]:
    #         h = torch.cat([h, hs.pop()], dim=1)
    #         h = module(h, emb)
        
    #     h = h.type(x_t.dtype)

    #     # print("h.requires_grad_", h.requires_grad)

    #     output_unet = self.unet.out(h)
        
    #     # if is_required_training:
    #     #     output_unet = output_unet.clone().requires_grad_(True)
    #     # print("gradddddddddddddddddddd", output_unet.requires_grad)

    #     ######## TRAINING 
    #     # mean, log_variance = torch.split(output_unet, x.shape[1], dim=1)
    #     # nonzero_mask = (
    #     #     (timesteps != 0).float().view(-1, *([1] * (len(h.shape) - 1)))
    #     # )  # no noise when t == 0
    #     # # Gaussianp_mean_variance_(model_output, x_t, t)
    #     # sample = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
    #     # return sample
    #     return output_unet


class AM(nn.Module):#Attention map genenration
    def __init__(self, nChannel, nFeat, growthRate = 32):
        super(AM, self).__init__()
        # nChannel = args.nChannel
        nDenselayer = 6

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        # self.conv2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        self.att11 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()


    def forward(self, x2, x1):#x2: enc feat; x1: gen feat, modify x1

        F1_ = self.relu(self.conv1(x1))
        F2_ = self.relu(self.conv1(x2))
        # F3_ = self.relu(self.conv1(x3))

        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = nn.functional.sigmoid(F1_A)
        F1_ = F1_ * F1_A

        return F1_, F2_ #gen feat, enc feat

       
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)   

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, is_last=False):
        super(ConvLayer, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)

    def forward(self, x):
        out = self.conv2d(x)
        return out
    
class SFC_Block(nn.Module):
    def __init__(self, in_ch, expansion=2):
        super(SFC_Block, self).__init__()

        exp_ch = int(in_ch * expansion)

        self.se_conv = nn.Conv2d(in_ch, exp_ch, 3, stride=1, padding=1, groups=in_ch)

        self.hd_conv = nn.Conv2d(exp_ch, exp_ch, 3, stride=1, padding=1, groups=in_ch)
        
        self.cp_conv = nn.Conv2d(exp_ch, in_ch, 1, stride=1, padding=0, groups=in_ch)

        self.gelu1 = nn.PReLU()
        self.gelu2 = nn.PReLU()
        
        self.fused = ConvLayer(in_ch*2, in_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, enc_feat, gen_feat):

        enh_result = torch.cat([enc_feat, gen_feat], 1)

        enh_result = self.fused(enh_result)
        x = self.se_conv(enh_result)
        x = self.gelu1(x)
        x = self.hd_conv(x)
        x = self.gelu2(x)
        x = self.cp_conv(x)

        return x  

def Fourier_filter(x, threshold, scale):
    # dtype = x.dtype
    # x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    # x_filtered = x_filtered.type(dtype)
    return x_filtered