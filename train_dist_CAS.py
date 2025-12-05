from __future__ import print_function

import argparse
# from cv2 import normalize
import os

#################### Import Pytorch libraries
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch import distributed as dist, nan_to_num_
#################### Import Python libraries
import math
import numpy as np
import random
import glob
from Network.Proposed import *
from diffusers.schedulers import DDIMScheduler
# from Network.losses import Perceptual
# 
from utils.LoadData import Load_ImagesDataset
import library.custom_train_functions as custom_train_functions
#################### Parameters
def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default = 100, type = int)
    parser.add_argument('--is_cuda', default = 'cuda', type = str)
    parser.add_argument('--batch_size', default =4, type = int) # default from StyleGAN2
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate") # layout2img-openimages256/config.yaml
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization") # default from StyleGAN2
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization") # default from StyleGAN2
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization") # default from StyleGAN2
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization") # default from StyleGAN2
    parser.add_argument('--lr_decay_interval', type=int, default=50,
                            help='decay learning rate every N epochs(default: 100)')
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    return parser


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0
    
def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()

def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor

def reduce_loss_dinputt(loss_dinputt):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dinputt

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dinputt.keys()):
            keys.append(k)
            losses.append(loss_dinputt[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

# def accumulate(model1, model2, decay=0.999):
#     par1 = dinputt(model1.named_parameters())
#     par2 = dinputt(model2.named_parameters())

#     for k in par1.keys():
#         par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def d_logistinput_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    This matches the implementation in Denoising Diffusion Probabilistinput Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

#cmd : CUDA_VISIBLE_deviceS="0,1" python -m torch.distributed.launch --nproc_per_node=2 train_dist.py

#################### Main code
if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    parser = setup_parser()
    opt = parser.parse_args()
    # opt = train_util.read_config_from_file(opt, parser)
    NUM_EPOCHS = opt.num_epochs
    BATCH = opt.batch_size
    device = opt.is_cuda
    LR = opt.lr
    weight_dtype = torch.float32
    # ### Distributed
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print(world_size)
    
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    dist.barrier()

    ### Train DATA path
    input_path = 'data/synthesis_COCO/*.png'
    mask_path = 'data/mask_COCO/*.png'
    gt_path = 'data/gt_COCO/*.png'
    INPUT = sorted(glob.glob(input_path)) 
    MASK = sorted(glob.glob(mask_path)) 
    GT =  sorted(glob.glob(gt_path)) 
    data_train = Load_ImagesDataset(INPUT, GT, MASK, is_trained=True) 

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size = BATCH,
                                                    sampler=torch.utils.data.distributed.DistributedSampler(data_train, shuffle=True),
                                                    drop_last=True,
                                                    num_workers=8)


    # text_encoder, vae, unet_ldm, load_stable_diffusion_format = train_util.load_target_model(opt, torch.float32)
    
    noise_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # tokenizer = train_util.load_tokenizer(opt)

    content_adaptor = Content_Adaptor_SubSpace().cuda()
    content_adaptor = nn.parallel.DistributedDataParallel(content_adaptor, broadcast_buffers=False, find_unused_parameters=True)


       
    
    ### Define Optimizers
   
    g_optim = optim.Adam(content_adaptor.parameters(), lr = LR,  betas=(0.9, 0.999))
    ### Continue training ...
    # g_optim.load_state_dict(state_dict["g_optim"])
    # d_optim.load_state_dict(state_dict1["d_optim"])

    ### Training and Testing section
    mean_path_length = 0
    mean_path_length_avg = 0
    ########## loss function
    rec_loss = nn.MSELoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # inception = InceptionScore()
    r1_loss = torch.tensor(0.0).cuda()
    path_loss = torch.tensor(0.0).cuda()
    path_lengths = torch.tensor(0.0).cuda()

    loss_dinputt = {}
    iteration = 0

    CUDA_LAUNCH_BLOCKING = 1

    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                content_adaptor.train()

                count = 0

                data_train_loader.sampler.set_epoch(epoch)
                total_train_loss = []
                
                # print("test")
                for i, data in enumerate(data_train_loader, 0):
                # Load images
                    input = data[0].cuda()
                    gt = data[1].cuda()                            
                    mask = data[2].cuda()
                    # img_52 = data[3].cuda()
                    roi_img = input * mask
                    
                    inv_mask = 1. - mask
                    bg_img = input * inv_mask
                    
                    b_size = gt.shape[0]
                    
                    
                    # Sample a random timestep for each image
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(gt, device=gt.device)

                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=gt.device)
                    timesteps = timesteps.long()

                    
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    gt_t = noise_scheduler.add_noise(gt, noise, timesteps)
                    input_t = noise_scheduler.add_noise(input, noise, timesteps)
                    

                    input_t_tilde = content_adaptor(roi_img, bg_img, input_t)
                    input_unet_ldm = torch.cat((input_t_tilde, mask, input),1)
                    loss = torch.nn.functional.mse_loss(input_t_tilde.float(), gt_t.float(), reduction="mean")
                    
                    tot_loss = loss 

                    total_train_loss = total_train_loss + [tot_loss.item()]
                    
                    # Save loss
                    loss_dinputt["g_total"] = tot_loss
                    with torch.enable_grad():
                        tot_loss.backward()

                    g_optim.step()
                    g_optim.zero_grad(set_to_none=True)

                    loss_reduced = reduce_loss_dinputt(loss_dinputt)


                    if is_main_process():
                        if count % 50 == 0:
                            print('[%d/%d][%d/%d]---------------------------------------------' %(epoch, NUM_EPOCHS, count, len(data_train_loader)))

                            print('-----UNet training-----')
                            print('time_step: (%d, %d, %d, %d) \t total loss: %.4f' %(
                                    timesteps[0],timesteps[1],timesteps[2],timesteps[3], tot_loss))
                            content_adaptor.train()
                    iteration = iteration + 1
                    count = count + 1

            
                if is_main_process():
                    # Save after each epoch
                    torch.save(
                        {
                        "g": content_adaptor.module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        }, 
                        "Weights/CAS_net_ep_%d_iter_%d.pth" %(epoch, iteration)
                    )
            