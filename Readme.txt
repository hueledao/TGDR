NOTE:
- Download pretrained diffusion model (256x256_diffusion_uncond.pt) from https://github.com/openai/guided-diffusion
- Set up environment: environment_guided_diff.yml

######################### Training CAS block #################################

CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.run --nproc_per_node=2 train_dist_CAS.py

######################### Training T-TAFE block ################################

CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="5" mpiexec -n 1 python train_dist_TTAFE.py --attention_resolutions 32,16,8 --class_cond False --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --rescale_learned_sigmas False

############################### Inference ####################################
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="5" python TGDR.py --attention_resolutions 32,16,8 --class_cond False --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_img_for_guidance --start_from_scratch --save_png_files --timestep_respacing ddim50 --img_guidance_scale 0.09


