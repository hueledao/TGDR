import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import datetime
import torch.utils.tensorboard as tfboard
from diffusers import AutoencoderKL
import time
import pyiqa
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import sys
sys.path.insert(0, '/media/ssd1/daole/sd_scripts/GenerativeDiffusionPrior/scripts/clip/clip/') 
# import clip_custom
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor, CLIPModel
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
    
class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        data_val=None,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        text_guided = False,
        both_text = False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.data_val = data_val
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.text_guided = text_guided
        self.both_text = both_text
        if self.text_guided:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir= "/media/ssd1/daole/NAS/pretrained")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            # print("clip_model =", self.clip_model)

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,#org = False
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=th.float16)
        self.vae.to(dist_util.dev())
        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # print(self.resume_step)
            # if dist.get_rank() == 0: #LE MODIFIED
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
    
    def run_loop(self, save_path, structual_loss = False, text_loss = False):
        num_iterations = 200000
        while self.step < num_iterations:
            if self.text_guided:
                f = open("captions_train_COCO_chatGPT.txt", "r")
                list_of_lines  = f.readlines()
            
            for i, data in enumerate(self.data, 0):
            # self.run_step(batch, cond)
                if self.text_guided:
                    prompts_list = []
                    for j in range(self.batch_size):
                        prompts = list_of_lines[i*self.batch_size:(i+1)*self.batch_size]
                        prompt = (' '.join(prompts[j].split()[1:]))
                        prompts_list.append(prompt)
                    text_inputs = self.tokenizer(prompts_list, padding='max_length', truncation=True, 
                        max_length=77, return_tensors='pt')
                    text_input_ids = text_inputs.input_ids
                    untruncated_ids = self.tokenizer(prompts_list, padding="longest", return_tensors="pt").input_ids
                    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not th.equal(
                        text_input_ids, untruncated_ids
                    ):
                        removed_text = self.tokenizer.batch_decode(
                            untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                        )
                        logger.warning(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                        )
                    
                    
                    emb_text = self.text_encoder(
                        text_input_ids,
                        attention_mask=None,
                    )
                    emb_text = emb_text[0]
                self.run_step(data[0], data[1], self.step, structual_loss, mask = data[2], emb_text=emb_text, text_loss = text_loss, processor = self.processor, clip_model = self.clip_model)
                
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save(save_path)
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                    
                self.step += 1
            
                
        # Save the last checkpoint if it wasn't already saved.
        if ( self.step - 1) % self.save_interval!= 0:

            self.save(self.step, save_path)

    # def run_step(self, batch, cond):
    def run_step(self, input, gt, step, structual_loss = False, CL_loss = False, I_52 = None, use_LPIPS_loss = False, LPIPS_cal = None, mask = None, emb_text = None, text_loss = False, processor = None, clip_model = None):
        self.forward_backward(input, gt, structual_loss, CL_loss, I_52, use_LPIPS_loss, LPIPS_cal, mask, emb_text, text_loss, processor, clip_model)
        took_step = self.mp_trainer.optimize(self.opt, step)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        # print(self.lr_anneal_steps)
        self.log_step()

    def forward_backward_val(self, input, batch, mask, lpips_metric, emb_text = None, text_loss = None):
        # self.mp_trainer.zero_grad()
        lpips_sum = 0
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_input = input[i : i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            with th.no_grad():
                zs = self.vae.encode(micro_input.to(dtype=th.float16)).latent_dist.sample()

            lpips_score = functools.partial(
                self.diffusion.validation_cal,
                self.ddp_model,
                micro_input,
                micro,
                t,
                zs,
                mask,
                lpips_metric,
                model_kwargs=None,
                emb_text= emb_text,
                both_text = self.both_text,
                text_loss = text_loss,
            )
            lpips_sum += lpips_score()
        return lpips_sum

    # def forward_backward(self, batch, cond):
    def forward_backward(self, input, batch, structual_loss = False, CL_loss = False, I_52 = None, use_LPIPS_loss = False, LPIPS_cal = None, mask = None, emb_text = None, text_loss = False, processor = None, clip_model  = None):#input, gt
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())#gt
            # micro_cond = {
            #     k: v[i : i + self.microbatch].to(dist_util.dev())
            #     for k, v in cond.items()
            # }
            micro_input = input[i : i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            if CL_loss:
                I_52=I_52.to(dist_util.dev())
            if mask is not None:
                mask=mask.to(dist_util.dev())

            # with th.no_grad():
            #     zs = self.vae.encode(micro_input.to(dtype=th.float16)).latent_dist.sample()
            if structual_loss:
                print("structual_loss structual_loss structual_loss structual_loss")
                compute_losses = functools.partial(
                    self.diffusion.training_losses_dual_unet_structural,
                    self.ddp_model,
                    micro_input,#I^D
                    micro,#gt
                    t,
                    structual_loss,
                    CL_loss = CL_loss,
                    I_52 = I_52,
                    use_LPIPS_loss = use_LPIPS_loss,
                    LPIPS_cal = LPIPS_cal,
                    mask = mask,
                    emb_text = emb_text,
                    text_loss = text_loss,
                    processor = processor,
                    # img_encoder = img_encoder.to(dist_util.dev()),
                    clip_model = clip_model.to(dist_util.dev()) if self.text_guided else None,
                    both_text = self.both_text,
                    model_kwargs=None
                )
            else:
                compute_losses = functools.partial(
                self.diffusion.training_losses_dual_unet,
                self.ddp_model,
                micro_input,
                micro,
                t,
                model_kwargs=None,
            )
                
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, save_path):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            # with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                with bf.BlobFile(bf.join(save_path, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)

        if self.step % (5*self.save_interval) == 0:
            for rate, params in zip(self.ema_rate, self.ema_params):
                save_checkpoint(rate, params)       

        # if dist.get_rank() == 0:
        if self.step % (5*self.save_interval) == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)