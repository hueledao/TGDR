"""
Train a diffusion model on images.
"""

import argparse
import os
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util_x0_modified import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from Network.Proposed import *
import glob
from utils.LoadData import Load_ImagesDataset
import datetime
# import torch.utils.tensorboard as tfboard

from torch import distributed as dist
def main():

    args = create_argparser().parse_args()
    save_path = "Weights/TTAFE/"
    # tensorboad_name ="TAFE_Transformer_TextGuided_TextLoss_ft"
    ########### Distributed
    dist_util.setup_dist()
    logger.configure(dir = save_path)

    logger.log("creating diffusion...")
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model = Dual_UNet_TTAFE(args)
    # model = Dual_UNet_semantic(args)
    model.to(dist_util.dev())
    model_path = "/media/ssd1/daole/sd_scripts/GenerativeDiffusionPrior/scripts/models/256x256_diffusion_uncond.pt"
    model.unet.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    # model.unet.convert_to_fp16()
    # model.unet.weight.requires_grad = False
    # model.unet.bias.requires_grad = False
    model.unet.eval()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    input_path = '/media/ssd1/daole/VCM_Proposed/data/New/synthesis_COCO/*.png'
    mask_path = '/media/ssd1/daole/VCM_Proposed/data/New/mask_COCO/*.png'
    gt_path = '/media/ssd1/daole/VCM_Proposed/data/New/gt_COCO/*.png'
    INPUT = sorted(glob.glob(input_path)) 
    MASK = sorted(glob.glob(mask_path)) 
    GT =  sorted(glob.glob(gt_path)) 
    data_train = Load_ImagesDataset(INPUT, GT, MASK, is_trained=True) 

    data = torch.utils.data.DataLoader(data_train,
                                                    batch_size = args.batch_size,
                                                    drop_last=True,
                                                    shuffle=True,
                                                    num_workers=8)

     #VALIDATION DATALOADER
    # input_val_path = '/media/ssd1/daole/VCM_Proposed/Validation_Results/test_in_COCO/*.png'
    # mask_val_path = '/media/ssd1/daole/VCM_Proposed/Validation_Results/test_mask_COCO/*.png'
    # gt_val_path = '/media/ssd1/daole/VCM_Proposed/Validation_Results/test_gt_COCO/*.png'
    # # input_val_path = '/media/ssd1/daole/VCM_Proposed/data/testing_20/test_in_COCO/*.png'
    # # mask_val_path = '/media/ssd1/daole/VCM_Proposed/data/testing_20/test_mask_COCO/*.png'
    # # gt_val_path = '/media/ssd1/daole/VCM_Proposed/data/testing_20/test_gt_COCO/*.png'
    # INPUT_VAL = sorted(glob.glob(input_val_path)) 
    # MASK_VAL = sorted(glob.glob(mask_val_path)) 
    # GT_VAL =  sorted(glob.glob(gt_val_path)) 
    # data_val = Load_ImagesDataset(INPUT_VAL, GT_VAL, MASK_VAL, is_trained=False) 

    # data_val_loader = torch.utils.data.DataLoader(data_val,
                                                    # batch_size = 1)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        text_guided = True,
        both_text = True, #both_text means using text embedding and text loss
    ).run_loop(save_path= save_path, structual_loss = True, text_loss = True)

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=2e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        clip_denoised=True,
        num_samples=20,
        use_ddim=False,
        model_path="/media/ssd1/daole/sd_scripts/GenerativeDiffusionPrior/scripts/models/256x256_diffusion_uncond.pt"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    return parser


if __name__ == "__main__":
    # tf_board_logs = 'TensorBoard/'
    # time_before = datetime.datetime.now()
    # tfb_train_writer = tfboard.SummaryWriter(log_dir=os.path.join(tf_board_logs, 'Time_Aware_Encoder_GD'))
    main()

    # CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="5" mpiexec -n 1 python train_dist_TTAFE.py --attention_resolutions 32,16,8 --class_cond False --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --rescale_learned_sigmas False