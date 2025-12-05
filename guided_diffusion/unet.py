from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numbers
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from einops import rearrange
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        # print(x.shape)
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        print(x.shape)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        # print(x.shape, self.positional_embedding[None, :, :].shape)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                # layer.type(x.dtype)#Le added for SSA
                x = layer(x, emb)
            else:
                # print(x.type())
            
                # print('first',layer.weight.data.type())
                # layer.type(x.dtype)#Le added for SSA
                x = layer(x)#org
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        # self.out_layers = nn.Sequential(
        #     normalization(self.out_channels),
        #     nn.SiLU(),
        #     nn.Dropout(p=dropout),
        #     zero_module(
        #         conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        #     ),
        # )
        if self.out_channels % 32 == 0:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(x.type())
        # print(emb.type())
        
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # emb = emb.type(x.dtype)#Le added
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # print("image_size", image_size)
        # print("in_channels", in_channels)
        # print("model_channels", model_channels)
        # print("out_channels", out_channels)
        # print("num_res_blocks", num_res_blocks)
        # print("attention_resolutions", attention_resolutions)
        # print("dropout", dropout)
        # print("channel_mult", channel_mult)
        # print("conv_resample", conv_resample)
        # print("dims", dims)
        # print("num_classes", num_classes)
        # print("use_checkpoint", use_checkpoint)
        # print("use_fp16", use_fp16)
        # print("num_heads", num_heads)
        # print("num_head_channels", num_head_channels)
        # print("num_heads_upsample", num_heads_upsample)
        # print("use_scale_shift_norm", use_scale_shift_norm)
        # print("resblock_updown", resblock_updown)
        # print("use_new_attention_order", use_new_attention_order)



        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, is_return_feat=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        if is_return_feat:
            return hs

        return self.out(h)
        # return hs, 


class UNetModel_TimeAwareEncoder(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        reso_512 = False,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        if reso_512:
            self.channel_mult = (1, 2, 4, 8, 8)
        else:
            self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        # self._feature_size += ch

        #Le 
        # input_block_chans.append(ch)
        # self._feature_size += ch
        # self.fea_tran = nn.ModuleList([])
        # for i in range(len(input_block_chans)):
        #     self.fea_tran.append(
        #         ResBlock(
        #             input_block_chans[i],
        #             time_embed_dim,
        #             dropout,
        #             out_channels=out_channels,
        #             dims=dims,
        #             use_checkpoint=use_checkpoint,
        #             use_scale_shift_norm=use_scale_shift_norm,
        #         )
        #     )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        # self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        # self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        result_list = []
        # results = {}
        h = x.type(self.dtype)
        # emb = emb.type(self.dtype)
        # print(self.input_blocks[0].weights.type())
        for module in self.input_blocks:
            # print('module',module.weight.data.type())
            
            last_h = h
            h = module(h, emb)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)
        h = self.middle_block(h, emb)
        result_list.append(h)

        # assert len(result_list) == len(self.fea_tran)

        # for i in range(len(result_list)):
        #     results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)

        return result_list

        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        # emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        # # if self.num_classes is not None:
        # #     assert y.shape == (x.shape[0],)
        # #     emb = emb + self.label_emb(y)

        # h = x.type(self.dtype)
        # for module in self.input_blocks:
        #     h = module(h, emb)
        #     hs.append(h)
        # # h = self.middle_block(h, emb)
        # # for module in self.output_blocks:
        # #     h = th.cat([h, hs.pop()], dim=1)
        # #     h = module(h, emb)
        # # h = h.type(x.dtype)
        # return h

class UNetModel_TimeAwareEncoder_Text(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        
        # text_time_embedding_from_dim = 1280

        # self.add_embedding = TextTimeEmbedding(
        #     text_time_embedding_from_dim, time_embed_dim, num_heads=64
        # )

        self.add_embedding = TextTimeEmbedding(
                768, time_embed_dim, num_heads=64#addition_embed_type_num_heads
            )
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        # self._feature_size += ch

        #Le 
        # input_block_chans.append(ch)
        # self._feature_size += ch
        # self.fea_tran = nn.ModuleList([])
        # for i in range(len(input_block_chans)):
        #     self.fea_tran.append(
        #         ResBlock(
        #             input_block_chans[i],
        #             time_embed_dim,
        #             dropout,
        #             out_channels=out_channels,
        #             dims=dims,
        #             use_checkpoint=use_checkpoint,
        #             use_scale_shift_norm=use_scale_shift_norm,
        #         )
        #     )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        # self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        # self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, encoder_hidden_states):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]
        aug_emb = self.add_embedding(encoder_hidden_states)
        emb = emb + aug_emb
        result_list = []
        # results = {}
        h = x.type(self.dtype)
        # emb = emb.type(self.dtype)
        # print(self.input_blocks[0].weights.type())
        for module in self.input_blocks:
            # print('module',module.weight.data.type())
            
            last_h = h
            h = module(h, emb)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)
        h = self.middle_block(h, emb)
        result_list.append(h)

        # assert len(result_list) == len(self.fea_tran)

        # for i in range(len(result_list)):
        #     results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)

        return result_list

        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        # emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        # # if self.num_classes is not None:
        # #     assert y.shape == (x.shape[0],)
        # #     emb = emb + self.label_emb(y)

        # h = x.type(self.dtype)
        # for module in self.input_blocks:
        #     h = module(h, emb)
        #     hs.append(h)
        # # h = self.middle_block(h, emb)
        # # for module in self.output_blocks:
        # #     h = th.cat([h, hs.pop()], dim=1)
        # #     h = module(h, emb)
        # # h = h.type(x.dtype)
        # return h



class UNetModel_TimeAwareEncoder_New(nn.Module): #adding fea_tran
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        # self._feature_size += ch

        #Le 
        input_block_chans.append(ch)
        self._feature_size += ch
        self.input_block_chans = input_block_chans
        self.fea_tran = nn.ModuleList([])
        for i in range(len(input_block_chans)):
            self.fea_tran.append(
                ResBlock(
                    input_block_chans[i],
                    time_embed_dim,
                    dropout,
                    # out_channels=input_block_chans[i],
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.fea_tran.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        # self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        result_list = []
        results = {}
        # print("h",h.type())
        

        h = x.type(self.dtype)
        # emb = emb.type(self.dtype)
        for module in self.input_blocks:
            last_h = h
            h = module(h, emb)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)
                # print("result_list",last_h.size())
        # print("h",h.type())
        # print("emb",emb.type())
        h = self.middle_block(h, emb)
        result_list.append(h)
        # print("result_list",h.size())


        assert len(result_list) == len(self.fea_tran)
        for i in range(len(result_list)):
            # print("result_list[i]",result_list[i].type())
            # emb = emb.type(result_list[i].dtype)
            # print("emb",emb.type())
            results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)
            # print("after fea_tran",results[str(result_list[i].size(-1))].size())
        # print(results)
        # print(results.size())
        return results

        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        # emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        # # if self.num_classes is not None:
        # #     assert y.shape == (x.shape[0],)
        # #     emb = emb + self.label_emb(y)

        # h = x.type(self.dtype)
        # for module in self.input_blocks:
        #     h = module(h, emb)
        #     hs.append(h)
        # # h = self.middle_block(h, emb)
        # # for module in self.output_blocks:
        # #     h = th.cat([h, hs.pop()], dim=1)
        # #     h = module(h, emb)
        # # h = h.type(x.dtype)
        # return h


class UNetModel_TimeAwareEncoder_Transformers(nn.Module): #using transformer blocks instead of resblocks
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        #for tranformers 
        heads = [1,2,4,8,8]
        ffn_expansion_factor = 2.66,

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                # print(level)
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        # ResBlock(
                        #     ch,
                        #     time_embed_dim,
                        #     dropout,
                        #     out_channels=out_ch,
                        #     dims=dims,
                        #     use_checkpoint=use_checkpoint,
                        #     use_scale_shift_norm=use_scale_shift_norm,
                        #     down=True,
                        # )
                        TransformerBlock(
                            channels=ch, 
                            num_heads=heads[level], 
                            ffn_expansion_factor=2.66, 
                            bias=False, 
                            LayerNorm_type='WithBias', 
                            time_embed_dim=time_embed_dim,
                            dims=dims,
                            down=True,) 

                        # if resblock_updown
                        # else Downsample(
                        #     ch, conv_resample, dims=dims, out_channels=out_ch
                        # )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            # ResBlock(
            #     ch,
            #     time_embed_dim,
            #     dropout,
            #     dims=dims,
            #     use_checkpoint=use_checkpoint,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            # ResBlock(
            #     ch,
            #     time_embed_dim,
            #     dropout,
            #     dims=dims,
            #     use_checkpoint=use_checkpoint,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
            TransformerBlock(
                            channels=ch, 
                            num_heads=heads[3], 
                            ffn_expansion_factor=2.66, 
                            bias=False, 
                            LayerNorm_type='WithBias', 
                            time_embed_dim=time_embed_dim,
                            dims=dims,),
            # TransformerBlock(
            #                 channels=ch, 
            #                 num_heads=heads[3], 
            #                 ffn_expansion_factor=2.66, 
            #                 bias=False, 
            #                 LayerNorm_type='WithBias', 
            #                 time_embed_dim=time_embed_dim,
            #                 dims=dims,),
            # TransformerBlock(
            #                 dim=ch, 
            #                 num_heads=heads[3], 
            #                 ffn_expansion_factor=2.66, 
            #                 bias=False, 
            #                 LayerNorm_type='WithBias', 
            #                 time_embed_dim=time_embed_dim)
        )
        # self._feature_size += ch

        #Le 
        # input_block_chans.append(ch)
        # self._feature_size += ch
        # self.fea_tran = nn.ModuleList([])
        # for i in range(len(input_block_chans)):
        #     self.fea_tran.append(
        #         ResBlock(
        #             input_block_chans[i],
        #             time_embed_dim,
        #             dropout,
        #             out_channels=out_channels,
        #             dims=dims,
        #             use_checkpoint=use_checkpoint,
        #             use_scale_shift_norm=use_scale_shift_norm,
        #         )
        #     )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        # self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        # self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        result_list = []
        # results = {}
        h = x.type(self.dtype)
        # emb = emb.type(self.dtype)
        # print(self.input_blocks[0].weights.type())
        for module in self.input_blocks:
            # print('module',module.weight.data.type())
            
            last_h = h
            h = module(h, emb)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)
        h = self.middle_block(h, emb)
        result_list.append(h)

        # assert len(result_list) == len(self.fea_tran)

        # for i in range(len(result_list)):
        #     results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)

        return result_list

        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        # emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        # # if self.num_classes is not None:
        # #     assert y.shape == (x.shape[0],)
        # #     emb = emb + self.label_emb(y)

        # h = x.type(self.dtype)
        # for module in self.input_blocks:
        #     h = module(h, emb)
        #     hs.append(h)
        # # h = self.middle_block(h, emb)
        # # for module in self.output_blocks:
        # #     h = th.cat([h, hs.pop()], dim=1)
        # #     h = module(h, emb)
        # # h = h.type(x.dtype)
        # return h

class UNetModel_TimeAwareEncoder_Transformers_Text(nn.Module): #using transformer blocks instead of resblocks, adding text embedding
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        reso_512 = False,
    ):
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout

        self.channel_mult = channel_mult
        # self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        #for tranformers 
        if image_size == 512:
            heads = [1, 2, 4, 8, 8, 8]
        else:
            heads = [1,2,4,8,8]
        ffn_expansion_factor = 2.66,

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.add_embedding = TextTimeEmbedding(
                768, time_embed_dim, num_heads=64#addition_embed_type_num_heads
            )
        
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )
        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                # print(level)
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        # ResBlock(
                        #     ch,
                        #     time_embed_dim,
                        #     dropout,
                        #     out_channels=out_ch,
                        #     dims=dims,
                        #     use_checkpoint=use_checkpoint,
                        #     use_scale_shift_norm=use_scale_shift_norm,
                        #     down=True,
                        # )
                        TransformerBlock(
                            channels=ch, 
                            num_heads=heads[level], 
                            ffn_expansion_factor=2.66, 
                            bias=False, 
                            LayerNorm_type='WithBias', 
                            time_embed_dim=time_embed_dim,
                            dims=dims,
                            down=True,) 

                        # if resblock_updown
                        # else Downsample(
                        #     ch, conv_resample, dims=dims, out_channels=out_ch
                        # )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            # ResBlock(
            #     ch,
            #     time_embed_dim,
            #     dropout,
            #     dims=dims,
            #     use_checkpoint=use_checkpoint,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
            # AttentionBlock(
            #     ch,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            # ResBlock(
            #     ch,
            #     time_embed_dim,
            #     dropout,
            #     dims=dims,
            #     use_checkpoint=use_checkpoint,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
            TransformerBlock(
                            channels=ch, 
                            num_heads=heads[3], 
                            ffn_expansion_factor=2.66, 
                            bias=False, 
                            LayerNorm_type='WithBias', 
                            time_embed_dim=time_embed_dim,
                            dims=dims,),
            # TransformerBlock(
            #                 channels=ch, 
            #                 num_heads=heads[3], 
            #                 ffn_expansion_factor=2.66, 
            #                 bias=False, 
            #                 LayerNorm_type='WithBias', 
            #                 time_embed_dim=time_embed_dim,
            #                 dims=dims,),
            # TransformerBlock(
            #                 dim=ch, 
            #                 num_heads=heads[3], 
            #                 ffn_expansion_factor=2.66, 
            #                 bias=False, 
            #                 LayerNorm_type='WithBias', 
            #                 time_embed_dim=time_embed_dim)
        )
        # self._feature_size += ch

        #Le 
        # input_block_chans.append(ch)
        # self._feature_size += ch
        # self.fea_tran = nn.ModuleList([])
        # for i in range(len(input_block_chans)):
        #     self.fea_tran.append(
        #         ResBlock(
        #             input_block_chans[i],
        #             time_embed_dim,
        #             dropout,
        #             out_channels=out_channels,
        #             dims=dims,
        #             use_checkpoint=use_checkpoint,
        #             use_scale_shift_norm=use_scale_shift_norm,
        #         )
        #     )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        # self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        # self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, encoder_hidden_states):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        # print("trans and text emb")
        # print('encoder_hidden_states', encoder_hidden_states)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]
        aug_emb = self.add_embedding(encoder_hidden_states)
        emb = emb + aug_emb

        result_list = []
        # results = {}
        h = x.type(self.dtype)
        # emb = emb.type(self.dtype)
        # print(self.input_blocks[0].weights.type())
        for module in self.input_blocks:
            # print('module',module.weight.data.type())
            
            last_h = h
            h = module(h, emb)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)
        h = self.middle_block(h, emb)
        result_list.append(h)

        # assert len(result_list) == len(self.fea_tran)

        # for i in range(len(result_list)):
        #     results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)

        return result_list

        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # hs = []
        # emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]

        # # if self.num_classes is not None:
        # #     assert y.shape == (x.shape[0],)
        # #     emb = emb + self.label_emb(y)

        # h = x.type(self.dtype)
        # for module in self.input_blocks:
        #     h = module(h, emb)
        #     hs.append(h)
        # # h = self.middle_block(h, emb)
        # # for module in self.output_blocks:
        # #     h = th.cat([h, hs.pop()], dim=1)
        # #     h = module(h, emb)
        # # h = h.type(x.dtype)
        # return h



class AttentionPooling(nn.Module):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54

    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter(th.randn(1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def forward(self, x):
        bs, length, width = x.size()

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose(1, 2)
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose(1, 2)
            return x

        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x.dtype)
        x = th.cat([class_token, x], dim=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = th.einsum("bts,bcs->bct", weight, v)

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose(1, 2)

        return a[:, 0, :]  # cls_token
    
class TextTimeEmbedding(nn.Module):
    def __init__(self, encoder_dim: int, time_embed_dim: int, num_heads: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.pool = AttentionPooling(num_heads, encoder_dim)
        self.proj = nn.Linear(encoder_dim, time_embed_dim)
        self.norm2 = nn.LayerNorm(time_embed_dim)

    def forward(self, hidden_states):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states
    
# class UNetModel_TimeAwareEncoder_Transformers_Text(nn.Module): #using transformer blocks instead of resblocks, adding text embedding
#     """
#     The full UNet model with attention and timestep embedding.

#     :param in_channels: channels in the input Tensor.
#     :param model_channels: base channel count for the model.
#     :param out_channels: channels in the output Tensor.
#     :param num_res_blocks: number of residual blocks per downsample.
#     :param attention_resolutions: a collection of downsample rates at which
#         attention will take place. May be a set, list, or tuple.
#         For example, if this contains 4, then at 4x downsampling, attention
#         will be used.
#     :param dropout: the dropout probability.
#     :param channel_mult: channel multiplier for each level of the UNet.
#     :param conv_resample: if True, use learned convolutions for upsampling and
#         downsampling.
#     :param dims: determines if the signal is 1D, 2D, or 3D.
#     :param num_classes: if specified (as an int), then this model will be
#         class-conditional with `num_classes` classes.
#     :param use_checkpoint: use gradient checkpointing to reduce memory usage.
#     :param num_heads: the number of attention heads in each attention layer.
#     :param num_heads_channels: if specified, ignore num_heads and instead use
#                                a fixed channel width per attention head.
#     :param num_heads_upsample: works with num_heads to set a different number
#                                of heads for upsampling. Deprecated.
#     :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
#     :param resblock_updown: use residual blocks for up/downsampling.
#     :param use_new_attention_order: use a different attention pattern for potentially
#                                     increased efficiency.
#     """

#     def __init__(
#         self,
#         image_size,
#         in_channels,
#         model_channels,
#         out_channels,
#         num_res_blocks,
#         attention_resolutions,
#         dropout=0,
#         channel_mult=(1, 2, 4, 8),
#         conv_resample=True,
#         dims=2,
#         num_classes=None,
#         use_checkpoint=False,
#         use_fp16=False,
#         num_heads=1,
#         num_head_channels=-1,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=False,
#         resblock_updown=False,
#         use_new_attention_order=False,
#     ):
#         super().__init__()
        
#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads
        
#         self.image_size = image_size
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         self.out_channels = out_channels
#         self.num_res_blocks = num_res_blocks
#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.num_classes = num_classes
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample
#         #for tranformers 
#         heads = [1,2,4,8,8]
#         ffn_expansion_factor = 2.66,

#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         if self.num_classes is not None:
#             self.label_emb = nn.Embedding(num_classes, time_embed_dim)

#         self.add_embedding = TextTimeEmbedding(
#                 512, time_embed_dim, num_heads=64#addition_embed_type_num_heads
#             )
        
#         # ch = input_ch = int(channel_mult[0] * model_channels)
#         self.input_blocks = nn.ModuleList(
#             [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
#         )
#         self._feature_size = model_channels
#         input_block_chans = []
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for _ in range(num_res_blocks):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=int(mult * model_channels),
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = int(mult * model_channels)
#                 # if ds in attention_resolutions:
#                 #     layers.append(
#                 #         AttentionBlock(
#                 #             ch,
#                 #             use_checkpoint=use_checkpoint,
#                 #             num_heads=num_heads,
#                 #             num_head_channels=num_head_channels,
#                 #             use_new_attention_order=use_new_attention_order,
#                 #         )
#                 #     )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self._feature_size += ch
#                 # input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 # print(level)
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         # ResBlock(
#                         #     ch,
#                         #     time_embed_dim,
#                         #     dropout,
#                         #     out_channels=out_ch,
#                         #     dims=dims,
#                         #     use_checkpoint=use_checkpoint,
#                         #     use_scale_shift_norm=use_scale_shift_norm,
#                         #     down=True,
#                         # )
#                         TransformerBlock(
#                             channels=ch, 
#                             num_heads=heads[level], 
#                             ffn_expansion_factor=2.66, 
#                             bias=False, 
#                             LayerNorm_type='WithBias', 
#                             time_embed_dim=time_embed_dim,
#                             dims=dims,
#                             down=True,) 

#                         # if resblock_updown
#                         # else Downsample(
#                         #     ch, conv_resample, dims=dims, out_channels=out_ch
#                         # )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 ds *= 2
#                 self._feature_size += ch

#         self.middle_block = TimestepEmbedSequential(
#             TransformerBlock(
#                             channels=ch, 
#                             num_heads=heads[3], 
#                             ffn_expansion_factor=2.66, 
#                             bias=False, 
#                             LayerNorm_type='WithBias', 
#                             time_embed_dim=time_embed_dim,
#                             dims=dims,),
#         )
        
#     def convert_to_fp16(self):
#         """
#         Convert the torso of the model to float16.
#         """
#         self.input_blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)
#         # self.output_blocks.apply(convert_module_to_f16)

#     def convert_to_fp32(self):
#         """
#         Convert the torso of the model to float32.
#         """
#         self.input_blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)
#         # self.output_blocks.apply(convert_module_to_f32)

#     def forward(self, x, timesteps, encoder_hidden_states): #encoder_hidden_states for text embedding
#         """
#         Apply the model to an input batch.

#         :param x: an [N x C x ...] Tensor of inputs.
#         :param timesteps: a 1-D batch of timesteps.
#         :param y: an [N] Tensor of labels, if class-conditional.
#         :return: an [N x C x ...] Tensor of outputs.
#         """
#         # assert (y is not None) == (
#         #     self.num_classes is not None
#         # ), "must specify y if and only if the model is class-conditional"

#         # hs = []
#         emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))#timesteps = tensor[999]
        
#         #adding text embeeding
#         aug_emb = self.add_embedding(encoder_hidden_states)
#         emb = emb + aug_emb

#         result_list = []
#         # results = {}
#         h = x.type(self.dtype)
#         # emb = emb.type(self.dtype)
#         # print(self.input_blocks[0].weights.type())
#         for module in self.input_blocks:
#             # print('module',module.weight.data.type())
            
#             last_h = h
#             h = module(h, emb)
#             if h.size(-1) != last_h.size(-1):
#                 result_list.append(last_h)
#         h = self.middle_block(h, emb)
#         result_list.append(h)

#         # assert len(result_list) == len(self.fea_tran)

#         # for i in range(len(result_list)):
#         #     results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)

#         return result_list

class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear", align_corners=True)
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        # print(self.num_head_channels)
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = th.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered

class Free_UNetModel(UNetModel):
    """
    :param b1: backbone factor of the first stage block of decoder.
    :param b2: backbone factor of the second stage block of decoder.
    :param s1: skip factor of the first stage block of decoder.
    :param s2: skip factor of the second stage block of decoder.
    """

    def __init__(
        self,
        b1,
        b2,
        s1,
        s2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.b1 = b1 
        self.b2 = b2
        self.s1 = s1
        self.s2 = s2

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            hs_ = hs.pop()

            # --------------- FreeU code -----------------------
            # Only operate on the first two stages
            if h.shape[1] == 1280:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = th.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = th.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:640] = h[:,:640] * ((self.b1 - 1 ) * hidden_mean + 1)
                hs_ = Fourier_filter(hs_, threshold=1, scale=self.s1)
            if h.shape[1] == 640:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = th.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = th.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                h[:,:320] = h[:,:320] * ((self.b2 - 1 ) * hidden_mean + 1)
                hs_ = Fourier_filter(hs_, threshold=1, scale=self.s2)
            # ---------------------------------------------------------

            h = th.cat([h, hs_], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


#copy from transformerBLock,py file in BFRffusion algorithm
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def reshape(x, emb_out):
    emb_list = []
    for emb in emb_out:
        while len(emb.shape) < len(x.shape):
            emb = emb[..., None]
        emb_list.append(emb)
    return emb_list

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = th.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(th.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / th.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = th.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(th.ones(normalized_shape))
        self.bias = nn.Parameter(th.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / th.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*3, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.pwconw = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2, x3 = self.project_in(x).chunk(3, dim=1)
        x2, x3 = self.dwconv(th.cat((x2,x3),dim=1)).chunk(2, dim=1)
        x = F.gelu(x1)*x3 + F.gelu(x2)*x3 + self.pwconw(x3)
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(th.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = th.nn.functional.normalize(q, dim=-1)
        k = th.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(TimestepBlock):
    def __init__(self, channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type, time_embed_dim,dims=2, up=False,
        down=False,):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(channels, LayerNorm_type).half()
        self.attn = Attention(channels, num_heads, bias).half()
        self.norm2 = LayerNorm(channels, LayerNorm_type).half()
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias).half()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 6 * channels, bias=True)
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
    def forward(self, x, emb):
        x_type = x.dtype
        # emb = emb.type(x.dtype)#Le added
        emb_out = self.adaLN_modulation(emb).chunk(6, dim=1)
        # emb_out = emb_out.type(x.dtype)#Le added
        # emb_out = tuple(th.tensor(val, dtype=x.dtype).item() for val in emb_out)#Le added
        # emb_out = tuple(emb_out.to(x.dtype).tolist())
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = reshape(x, emb_out)
        x = x.to(x_type)
        shift_msa = shift_msa.to(x_type)
        scale_msa = scale_msa.to(x_type)
        gate_msa = gate_msa.to(x_type)
        shift_ffn = shift_ffn.to(x_type)
        scale_ffn = scale_ffn.to(x_type)
        gate_ffn = gate_ffn.to(x_type)
        # print('reshape', x.type())
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # print('gate_msa', x.type())
        # x = x.to(x_type)
        x = x + gate_ffn * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        # print('gate_ffn', x.type())
        if self.updown:
            x = self.x_upd(x)
        return x.to(x_type)
    
class MFEM(nn.Module):
    def __init__(self, 
        in_channels=4, 
        control_channels = 320,
        time_embed_dim = 1280,
        heads = [1,2,4,8],
        conv_resample=True,
        dims=2,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   
    ):

        super().__init__()
        self.control_channels = control_channels
        self.dims = dims
        self.time_embed = nn.Sequential(
            linear(control_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, 3, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, control_channels, 3, padding=1))
        )
        self.input_blocks = TimestepEmbedSequential(conv_nd(dims, in_channels, control_channels, 3, padding=1))


        # self.conv3x3 = nn.Conv2d(in_channels, control_channels,3, padding=1)      # 4,64,64 ->320,64,64
        self.resblock = ResBlock(control_channels, time_embed_dim, dropout=0, out_channels=control_channels)
        
        self.encoder1 = TransformerBlock(dim=control_channels, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.down1 = Downsample(control_channels, conv_resample, dims, control_channels*2) ## From 320,64,64 to 640,32,32
        
        self.encoder2 = TransformerBlock(dim=int(control_channels*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.down2 = Downsample(control_channels*2, conv_resample, dims, control_channels*4) ## From 640,32,32 -> 1280,16,16
        
        self.encoder3 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.down3 = Downsample(control_channels*4, conv_resample, dims, control_channels*4) ## From 1280,16,16 -> 1280,8,8
            
        self.mid1 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.mid2 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 
        self.mid3 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim) 

        self.up3 = Upsample(control_channels*4, conv_resample, dims, control_channels*4) ## From 1280,8,8 -> 1280,16,16
        self.reduce_chan_level3 = nn.Conv2d(int(control_channels*8), int(control_channels*4), kernel_size=1, bias=bias)
        self.decoder3 = TransformerBlock(dim=int(control_channels*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim)

        self.up2 = Upsample(control_channels*4, conv_resample, dims, control_channels*2) ## From 1280,16,16 -> 640,32,32
        self.reduce_chan_level2 = nn.Conv2d(int(control_channels*4), int(control_channels*2), kernel_size=1, bias=bias)
        self.decoder2 = TransformerBlock(dim=int(control_channels*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim)
        
        self.up1 = Upsample(control_channels*2, conv_resample, dims, control_channels)  ## From 640,32,32 -> 320,64,64
        self.reduce_chan_level1 = nn.Conv2d(int(control_channels*2), int(control_channels), kernel_size=1, bias=bias)
        self.decoder1 = TransformerBlock(dim=int(control_channels), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, time_embed_dim=time_embed_dim)
        
        self.zero_convs_module = nn.ModuleList([TimestepEmbedSequential(conv_nd(self.dims, control_channels, control_channels, 1, padding=0)), 
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*2, control_channels*2, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*4, control_channels*4, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels*2, control_channels*2, 1, padding=0)),
                                                TimestepEmbedSequential(conv_nd(self.dims, control_channels, control_channels, 1, padding=0))])

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.control_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        hint = self.input_hint_block(hint, emb, context)
        x = self.input_blocks(x, emb, context) + hint
 

        # x = self.conv3x3(x)
        x = self.resblock(x, emb)
        en1 = self.encoder1(x, emb)    
        dn1 = self.down1(en1)

        en2 = self.encoder2(dn1, emb)
        dn2 = self.down2(en2)

        en3 = self.encoder3(dn2, emb) 
        dn3 = self.down3(en3)
        
        mid1 = self.mid1(dn3, emb)
        mid2 = self.mid2(mid1, emb)
        mid3 = self.mid3(mid2, emb)

        up3 = self.up3(mid3)
        up3 = th.cat([up3, en3], 1)
        up3 = self.reduce_chan_level3(up3)
        de3 = self.decoder3(up3, emb) 

        up2 = self.up2(de3)
        up2 = th.cat([up2, en2], 1)
        up2 = self.reduce_chan_level2(up2)
        de2 = self.decoder2(up2, emb) 

        up1 = self.up1(de2)
        up1 = th.cat([up1, en1], 1)
        up1 = self.reduce_chan_level1(up1)
        de1 = self.decoder1(up1, emb)

        out = [en1,en2,en3,mid1,mid2,mid3,de3,de2,de1]

        assert len(out) == len(self.zero_convs_module)

        for i,module in enumerate(self.zero_convs_module):
            out[i] = module(out[i],emb, context)
        return out