"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from Network.Proposed import *
import sys
from PIL import Image
import random
from guided_diffusion import logger
from diffusers import DDIMScheduler
from multiprocessing.pool import ThreadPool
from guided_diffusion.script_util_x0_modified import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import torchvision.transforms as transforms
import glob
from utils.LoadData import Load_TestImagesDataset
import torchvision.transforms.functional as TF
sys.path.insert(1, 'VVC/') 
import VVC
from transformers.utils.logging import set_verbosity_error
from transformers import CLIPTokenizer, CLIPTextModel
torch.use_deterministic_algorithms(True, warn_only=True)
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

input_path = '/media/ssd1/daole/VCM_Proposed/data/test_in_COCO/*.png'
mask_path = '/media/ssd1/daole/VCM_Proposed/data/test_mask_COCO/*.png'
gt_path = '/media/ssd1/daole/VCM_Proposed/data/test_gt_COCO/*.png'
INPUT = sorted(glob.glob(input_path))
GT = sorted(glob.glob(gt_path))
MASK = sorted(glob.glob(mask_path))
data_val = Load_TestImagesDataset(INPUT, GT, MASK)    

noise_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

deg = 'TGDR_results'
logger.log("mse between full jpeg(output) and full input")

image_size = 256
channels = 3
device = 'cuda:2'
H_funcs = None
def main():
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    torch.cuda.manual_seed_all(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = create_argparser().parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = th.device('cuda')
    
    
    save_dir = args.save_dir if len(args.save_dir)>0 else None

    # dist_util.setup_dist()
    logger.configure(dir = save_dir)

    logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # diffusion.training_losses

    model_dual = Dual_UNet_TTAFE(args).cuda()
    model_dual.unet.convert_to_fp16()
    model_dual.time_aware_enc.convert_to_fp16()

    ################# loading pt file
    model_path = '/media/ssd1/daole/NAS/sd_scripts/GenerativeDiffusionPrior/scripts/Weights/Time_Aware_Encoder/Transformers_TextGuided_TextLoss/model610000.pt'
    model_dual.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    model_path = "/media/ssd1/daole/sd_scripts/GenerativeDiffusionPrior/scripts/models/256x256_diffusion_uncond.pt"
    model_dual.unet.load_state_dict(
        th.load(model_path, map_location="cpu")
    )
    model_dual.eval()
    BATCH = args.batch_size
    data_val = Load_TestImagesDataset(INPUT, GT, MASK)    
    data_val_loader = torch.utils.data.DataLoader(data_val,
                                                batch_size = BATCH, shuffle = False, num_workers = 4)
    for batchIdx in range(BATCH):
        save_dir_ = os.path.join(logger.get_dir(), 'vvc',  'batch_' + str(batchIdx))
        if not os.path.exists(save_dir_):
            os.makedirs(save_dir_)
        for i in range(3):
            child = os.path.join(save_dir_, str(i))
            if not os.path.exists(child):
                os.makedirs(child)
    thread_pool = ThreadPool(processes=12)

    #input x should be in range[-1, 1]
    def grad_VVC(x_in , x_lr, maskBG, batchIdx, curStep, fast=False):
        x_in = x_in[batchIdx]
        maskBG = maskBG[batchIdx]
        x_lr   = x_lr[batchIdx]

        h = 16/255
        perturb =torch.ones_like(x_in).cuda()
        x_in_h_fw = x_in + h*perturb
        x_in_h_fw = torch.clamp(x_in_h_fw, -1, 1)

        x_in_h_bw = x_in - h*perturb
        x_in_h_bw = torch.clamp(x_in_h_bw, -1, 1)
        save_dir_ = os.path.join(logger.get_dir(), 'vvc',  'batch_' + str(batchIdx))

        x_in_degarded = VVC.vvc_func(((x_in+1)/2), curStep, save_dir_)
        x_in_lr_h_fw = VVC.vvc_func_h(((x_in_h_fw+1)/2), curStep, save_dir_)
        x_in_lr_h_bw = VVC.vvc_func_h(((x_in_h_bw+1)/2), curStep, save_dir_)

        grad_Dx = 2*maskBG*(x_in_degarded - ((x_lr+1)/2))
        grad_x = (x_in_lr_h_fw - x_in_lr_h_bw)*perturb/(2*h)
        grad_BG = grad_Dx*grad_x

        return  grad_BG, (x_in_degarded*2 - 1) #to range [-1, 1] 
    
    def grad_VVC_batch_parallel(x_in , x_lr, maskBG, curStep):
        # start = time.time()
        N_batches = x_in.shape[0]
        x_in_degraded_batch = N_batches*[None]
        grad_BG_batch = N_batches*[None]
        task = N_batches*[None]
        # print("N batches = ", N_batches)

        for batchIdx in range(N_batches):
            task[batchIdx] = thread_pool.apply_async(grad_VVC, args=(x_in, x_lr, maskBG, batchIdx, curStep, False))
        
        for batchIdx in range(N_batches):
            curGrad, degraded_x = task[batchIdx].get()
            grad_BG_batch[batchIdx] = curGrad
            x_in_degraded_batch[batchIdx] = degraded_x

        grad_BG_batch       = torch.stack(grad_BG_batch, dim=0)
        x_in_degraded_batch = torch.stack(x_in_degraded_batch, dim=0)

        # print("Parallel running time:", time.time() - start)
        return grad_BG_batch, x_in_degraded_batch
    
    def grad_VVC_batch_parallel(x_in , x_lr, maskBG, curStep):
        # start = time.time()
        N_batches = x_in.shape[0]
        x_in_degraded_batch = N_batches*[None]
        grad_BG_batch = N_batches*[None]
        task = N_batches*[None]
        # print("N batches = ", N_batches)

        for batchIdx in range(N_batches):
            task[batchIdx] = thread_pool.apply_async(grad_VVC, args=(x_in, x_lr, maskBG, batchIdx, curStep, False))
        
        for batchIdx in range(N_batches):
            curGrad, degraded_x = task[batchIdx].get()
            grad_BG_batch[batchIdx] = curGrad
            x_in_degraded_batch[batchIdx] = degraded_x

        grad_BG_batch       = torch.stack(grad_BG_batch, dim=0)
        x_in_degraded_batch = torch.stack(x_in_degraded_batch, dim=0)

        # print("Parallel running time:", time.time() - start)
        return grad_BG_batch, x_in_degraded_batch
    
    def general_cond_fn(x, t, scale, x_lr=None, mask = None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None):
        # assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True) #output of previous t_step
            ######## GRADIENT FOR ROI
            mse_roi = torch.sum((((x_lr+1)/2- (x_in+1)/2)*mask)**2)
            grad_ROI = th.autograd.grad(mse_roi, x_in, retain_graph=True)[0]

            ######## GRADIENT FOR BG
            #applying VVC 
            maskBG = 1 - mask
            gradBG, x_degraded = grad_VVC_batch_parallel(x_in, x_lr, maskBG, t[0].item())
# 
            mse_bg = (x_degraded - x_lr) ** 2
            mse_bg = mse_bg*maskBG
            mse_bg = mse_bg.mean(dim=(1,2,3))
            mse_bg = mse_bg.sum()

            print('step t %d img guidance has been used, mse bg is %.4f' % (t[0], mse_bg))
            
            return -args.img_guidance_scale*(grad_ROI + gradBG) 
     
    #LE MODIFIED
    def model_fn(x, x_t, t):
        return model_dual(x, x_t, t, is_spade= True, emb_text = emb_text)#t = tensor[999]
    

    logger.log("loading dataset...")
    logger.log("input x900; denoising t = 880, 860, ..., 0")
    if args.save_png_files:
        print(logger.get_dir())
        os.makedirs(os.path.join(logger.get_dir(), 'images'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'vvc'), exist_ok=True)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    content_adaptor = Content_Adaptor_SubSpace().cuda()
    net_path1 = '/media/ssd1/daole/sd_scripts/GenerativeDiffusionPrior/scripts/Weights/CAS_net.pth'
    # net_path = '/media/ssd1/daole/sd_scripts/Weights/Weights_Content_Adaptor/Gated_Bottleneck_Image_BG_ROI/net_ep_139_iter_495370.pth'
    state_dict1 = torch.load(net_path1, map_location = lambda s, l: s)
    content_adaptor.load_state_dict(state_dict1["g"]) 
    content_adaptor = content_adaptor.eval()
    saved_img = 0

    tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir= "/media/ssd1/daole/NAS/pretrained")
    # f = open("/media/ssd1/daole/NAS/COCO_dataset/annotations/captions_test_dataset.txt", "r")
    f = open("captions_test_COCO_chatGPT.txt", "r")
    list_of_lines  = f.readlines()
    for i, data in enumerate(data_val_loader, 0):
        # if i > 664:
        if 1:
            
            # print(len(list_of_lines))
            prompts_list = []
            for j in range(BATCH):
                prompts = list_of_lines[i*BATCH:(i+1)*BATCH]
                prompt = (' '.join(prompts[j].split()[1:]))
                # print(prompt)
                prompts_list.append(prompt)
            text_inputs = tokenizer(prompts_list, padding='max_length', truncation=True, 
                        max_length=77, return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompts_list, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not th.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )
            
            
            emb_text = text_encoder(
                text_input_ids,
                attention_mask=None,
            )
            emb_text = emb_text[0].cuda()

            image_lr = data[0].cuda() #input
            image_lr_tensor = image_lr
            image = data[1].cuda()#gt
            mask_image = data[2].cuda()
            roi_img = (image_lr * mask_image).cuda() 
            bg_img = (image_lr * (1 - mask_image)).cuda()
            label = torch.Tensor([0])
            
            cond_fn = lambda x,t,scale : general_cond_fn(x, t,scale, x_lr=image_lr, mask = mask_image, sample_noisy_x_lr=args.sample_noisy_x_lr, diffusion=diffusion, sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred)

            shape = (image.shape[0], 3, args.image_size, args.image_size)
            classes = label.cuda().long()
            image = image.cuda()
            noise = torch.randn_like(image, device=image.device)
            xT = noise_scheduler.add_noise(image_lr, noise, torch.LongTensor([900]))
            input_t_tilde = content_adaptor(roi_img, bg_img, xT)
            model_kwargs = {}
            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop_modified
            )
            if args.start_from_scratch:
                sample = sample_fn(
                    model_fn,
                    shape,
                    noise = input_t_tilde,################ modified
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    mask_image = mask_image,
                    IC = image_lr,
                )
            else:
                sample = sample_fn(
                    model_fn,
                    shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=device,
                    noise=image,
                    denoise_steps=args.denoise_steps
                )
            sample_norm = sample
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8) #[-1;1] -> 0;255
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            image_lr = ((image_lr + 1) * 127.5).clamp(0, 255).to(th.uint8)
            image_lr = image_lr.permute(0, 2, 3, 1)
            image_lr = image_lr.contiguous()

            if args.save_png_files:
                sample_img = torch.Tensor(sample_norm).cuda()
            
                sample_saving = ((sample_img + 1)/2) *(1-mask_image) + ((image_lr_tensor+1)/2)*mask_image
                # # mask_image.min
                save_dir_ = os.path.join(logger.get_dir(), 'images') 
                for j in range(sample_saving.shape[0]):
                    # filename = os.path.join(save_dir_, f"{str(i*BATCH+j).c}.png")
                    filename = os.path.join(save_dir_, f"{str(i*BATCH+j).zfill(6)}.png")
                    utils.save_image(sample_saving[j:j+1,:,:,:], filename, nrow=1, normalize=False)
                    
            all_images.append(sample)
            all_labels.append(classes)
            logger.log(f"created {len(all_images) * args.batch_size} samples")

    if args.save_numpy_array:
        arr = np.concatenate(all_images, axis=0)
        label_arr = np.concatenate(all_labels, axis=0)

        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_rank_{args.global_rank}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    # dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=20,
        batch_size=5,
        use_ddim=False,
        model_path="/media/ssd1/daole/sd_scripts/GenerativeDiffusionPrior/scripts/models/256x256_diffusion_uncond.pt"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)


    save_dir  = os.path.join('/media/ssd1/daole/TGDR/generate_images', (deg))
    parser.add_argument("--device", default=0, type=int, help='the cuda device to use to generate images')
    parser.add_argument("--global_rank", default=0, type=int, help='global rank of this process')
    parser.add_argument("--world_size", default=1, type=int, help='the total number of ranks')
    parser.add_argument("--save_dir", default=save_dir, type=str, help='the directory to save the generate images')
    parser.add_argument("--save_png_files", action='store_true', help='whether to save the generate images into individual png files')
    parser.add_argument("--save_numpy_array", action='store_true', help='whether to save the generate images into a single numpy array')
    
    # these two arguments are only valid when not start from scratch
    parser.add_argument("--denoise_steps", default=20, type=int, help='number of denoise steps')
    parser.add_argument("--dataset_path", default='', type=str, help='path to the generated images. Could be an npz file or an image folder')

    parser.add_argument("--use_img_for_guidance", action='store_true', help='whether to use a (low resolution) image for guidance. If true, we generate an image that is similar to the low resolution image')
    parser.add_argument("--img_guidance_scale", default=0.07, type=float, help='guidance scale')
    parser.add_argument("--text_guidance_scale", default=0.1, type=float, help='guidance scale')
    parser.add_argument("--sample_noisy_x_lr", action='store_true', help='whether to first sample a noisy x_lr, then use it for guidance. ')
    parser.add_argument("--sample_noisy_x_lr_t_thred", default=1e8, type=int, help='only for t lower than sample_noisy_x_lr_t_thred, we add noise to lr')
    
    parser.add_argument("--start_from_scratch", action='store_true', help='whether to generate images purely from scratch, not use gan or vae generated samples')
    parser.add_argument("--deg", default='inp', type=str, help='the chosen of degradation model')

    return parser

import pdb
if __name__ == "__main__":
    main()