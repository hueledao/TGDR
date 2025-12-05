"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
from PIL import Image
import numpy as np
import torch as th
import torch.nn as nn
from einops import rearrange
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from torchvision import utils
from diffusers import DDIMScheduler
# from functions.jpeg_torch import jpeg_decode, jpeg_encode
import torch.nn.functional as F
from Network.lpips_network import *
# from Network.wavelet_color_fix import wavelet_reconstruction
# import pyiqa
# from Network.Proposed_v1 import LPIPS
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL
    

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.noise_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.noise_scheduler.set_timesteps(self.num_timesteps)
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        t could range from 0 to self.num_timesteps-1
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def noise_like(self, shape, device, repeat=False):
        repeat_noise = lambda: th.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
        noise = lambda: th.randn(shape, device=device)
        return repeat_noise() if repeat else noise()

    def sample_from_eps(self, x, e_t, model_ldm, sampler, index):
        b =1
        alphas = sampler.ddim_alphas
        alphas_prev =  sampler.ddim_alphas_prev
        sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas 
        sigmas = sampler.ddim_sigmas
        device = model_ldm.betas.device
        a_t = th.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = th.full((b, 1, 1, 1), alphas_prev[index], device=device)
        # beta_t = a_t / a_prev
        sigma_t = th.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = th.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        ############# UPDATE e_t
        # new_et = self.condition_sample(cond_fn, e_t.detach(), pred_x0, t)
        # pred_x0 = (x - sqrt_one_minus_at * new_et) / a_t.sqrt()
        if 1:
            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            prev_sample = c1 * pred_x0 + c2 * x + c3 * th.randn_like(pred_x0)#Ep. (13) in Steered diffusion
        else:
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * self.noise_like(x.shape, device, False)  
            prev_sample = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return prev_sample, pred_x0
    
    def est_mean_from_eps(self, x, e_t, model_ldm, sampler, index):
        b =1
        alphas = sampler.ddim_alphas
        alphas_prev =  sampler.ddim_alphas_prev
        sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas 
        sigmas = sampler.ddim_sigmas
        device = model_ldm.betas.device
        a_t = th.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = th.full((b, 1, 1, 1), alphas_prev[index], device=device)
        # beta_t = a_t / a_prev
        sigma_t = th.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = th.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        ############# UPDATE e_t
        # new_et = self.condition_sample(cond_fn, e_t.detach(), pred_x0, t)
        # pred_x0 = (x - sqrt_one_minus_at * new_et) / a_t.sqrt()
        beta_t = 1 - a_t
        mean = (1/a_t.sqrt())*(x - e_t*(beta_t/(1-a_prev)))
        return mean
    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, IC = None, zs = None, vae = None, is_latentspace = None, text_embedding = None, freedom =None, model_ldm = None, index = None, sampler=None, unet_grad = None, gt_t = None, text_steer = False,
        
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        if is_latentspace:
            model_output = model(x, self._scale_timesteps(t), text_embedding) #z_hat 
        # if emb_text is not None:
        #     model_output = model(IC, x, self._scale_timesteps(t), emb_text)
        else:
            model_output = model(IC, x, self._scale_timesteps(t)) #mean, variance
        # model_output, feat = model(x, self._scale_timesteps(t), is_return_feat = True)
        

        # model_output = model(IC, x, self._scale_timesteps(t))
        # model_output = model(IC, x, self._scale_timesteps(t), **model_kwargs)
        # model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        # model_output, feat = model(x, self._scale_timesteps(t), is_return_feat = True)
        if is_latentspace:
            if freedom:
                # b =1
                # e_t = model_output
                # noise_pred = e_t
                # alphas = sampler.ddim_alphas
                # alphas_prev =  sampler.ddim_alphas_prev
                # sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas 
                # sigmas = sampler.ddim_sigmas

                # device = model_ldm.betas.device
                # a_t = th.full((b, 1, 1, 1), alphas[index], device=device)
                # a_prev = th.full((b, 1, 1, 1), alphas_prev[index], device=device)
                # beta_t = a_t / a_prev
                # sigma_t = th.full((b, 1, 1, 1), 0, device=device)
                # sqrt_one_minus_at = th.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

                # # current prediction for x_0
                # pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                # ############# UPDATE e_t
                # # new_et = self.condition_sample(cond_fn, e_t.detach(), pred_x0, t)
                # # pred_x0 = (x - sqrt_one_minus_at * new_et) / a_t.sqrt()
                
                # c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
                # c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
                # c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
                # c3 = (c3.log() * 0.5).exp()
                # prev_sample = c1 * pred_x0 + c2 * x + c3 * th.randn_like(pred_x0)#Ep. (13) in Steered diffusion
                # pred_xstart = pred_x0
                noise_pred = model_output
                prev_sample, pred_xstart = self.sample_from_eps(x, model_output, model_ldm, sampler, index)
            else:
                noise_pred = model_output.sample
                prev_sample = self.noise_scheduler.step(noise_pred, self._scale_timesteps(t), x).prev_sample
                pred_xstart = self.noise_scheduler.step(noise_pred, self._scale_timesteps(t), x).pred_original_sample
            # img_ = vae.decode(1 / 0.18215 * prev_sample).sample 
            # utils.save_image(img_, 'Imgs/step_%d.png' % (t), nrow=1, normalize=True)
        else:
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                assert model_output.shape == (B, C * 2, *x.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                if self.model_var_type == ModelVarType.LEARNED:
                    model_log_variance = model_var_values
                    model_variance = th.exp(model_log_variance)
                else:
                    min_log = _extract_into_tensor(
                        self.posterior_log_variance_clipped, t, x.shape
                    )
                    max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                    # The model_var_values is [-1, 1] for [min_var, max_var].
                    frac = (model_var_values + 1) / 2
                    model_log_variance = frac * max_log + (1 - frac) * min_log
                    model_variance = th.exp(model_log_variance)
            else:
                model_variance, model_log_variance = {
                    # for fixedlarge, we set the initial (log-)variance like so
                    # to get a better decoder log likelihood.
                    ModelVarType.FIXED_LARGE: (
                        np.append(self.posterior_variance[1], self.betas[1:]),
                        np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                    ),
                    ModelVarType.FIXED_SMALL: (
                        self.posterior_variance,
                        self.posterior_log_variance_clipped,
                    ),
                }[self.model_var_type]
                model_variance = _extract_into_tensor(model_variance, t, x.shape)
                model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

            def process_xstart(x):
                if denoised_fn is not None:
                    x = denoised_fn(x)
                    # print("denoised_fn", denoised_fn)
                if clip_denoised:
                    # print("clip_denoised", clip_denoised)
                    return x.clamp(-1, 1)
                return x

            if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
                )
                model_mean = model_output
            elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart = process_xstart(model_output)
                    # print(pred_xstart.size())
                else:
                    pred_xstart = process_xstart(
                        self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                    )
                model_mean, _, _ = self.q_posterior_mean_variance(
                    x_start=pred_xstart, x_t=x, t=t
                )
            else:
                raise NotImplementedError(self.model_mean_type)
            assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
            )
        if is_latentspace:
            return {
                "noise_pred": noise_pred,
                "prev_sample": prev_sample,
                "pred_xstart": pred_xstart,
                "mean": None,
            }
        else:
            return {
                "mean": model_mean,
                "variance": model_variance,
                "log_variance": model_log_variance,
                "pred_xstart": pred_xstart,
            }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        # print(t)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, pred_xstart, t, scale=None, model_kwargs=None, dynamic_s = False):
        
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        # writer = SummaryWriter('abcdef')
        # gradient = cond_fn(pred_xstart, self._scale_timesteps(t), **model_kwargs)
        
        optimizer = th.optim.Adam([p_mean_var["mean"]], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        new_mean = p_mean_var["mean"].float()
        # print("old_mean", new_mean)'
        old_mean = new_mean
        for i in range(1):
            # gradient = cond_fn(pred_xstart, self._scale_timesteps(t), **model_kwargs)
            
            if scale is None:
                gradient = cond_fn(pred_xstart, self._scale_timesteps(t))
            else:
                gradient = cond_fn(pred_xstart, self._scale_timesteps(t), scale.mean())
                #PGDiff
            #coefficient = th.linalg.norm(x-sample)/th.linalg.norm(gradient) * model_kwargs["scale"]
            # print(new_mean.grad_fn)
            # print('----------------------->>>>>>>>>>>>>>>', th.unique(gradient))
            # if dynamic_s:
            #     # print('awlkerjlawekjr')
            #     noise = th.randn_like(x)
            #     nonzero_mask = (
            #         (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            #     )  # no noise when t == 0
            #     sample = p_mean_var["mean"] + nonzero_mask * th.exp(0.5 * p_mean_var["log_variance"]) * noise
            #     coefficient = th.linalg.norm(x-sample)/th.linalg.norm(gradient)
                
            #     new_mean = (p_mean_var["mean"].float() +  gradient.float()*coefficient)
            # else:
            new_mean = (p_mean_var["mean"].float() +  gradient.float())
            new_mean.requires_grad_(True)
            # print(new_mean.grad_fn)
            optimizer.zero_grad()
            # print(new_mean.grad_fn)
            new_mean.backward(th.ones_like(new_mean))
            
            # gradient.backward()
            # print(gradient.grad_fn)
            optimizer.step()
            # for tag, parm in p_mean_var["mean"].named_parameters:
            # writer.add_histogram('hist', new_mean.grad.data.cpu().numpy(), 0)
                # writer.add_graph(p_mean_var, x)
            # writer.close()
            # sys.exit(0)
        # print("diff_mean", gradient)
        if t[0]>20:
            return new_mean
        else:
            return old_mean
    
    def condition_grad(self, cond_fn, p_mean_var, x, pred_xstart, t, scale=None, model_kwargs=None, dynamic_s = False):
        
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        # writer = SummaryWriter('abcdef')
        # gradient = cond_fn(pred_xstart, self._scale_timesteps(t), **model_kwargs)
        
        optimizer = th.optim.Adam([p_mean_var["mean"]], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        new_mean = p_mean_var["mean"].float()
        # print("old_mean", new_mean)'
        old_mean = new_mean
        for i in range(1):
            # gradient = cond_fn(pred_xstart, self._scale_timesteps(t), **model_kwargs)
            
            if scale is None:
                gradient = cond_fn(pred_xstart, self._scale_timesteps(t))
            else:
                gradient = cond_fn(pred_xstart, self._scale_timesteps(t), scale.mean())
                #PGDiff
            #coefficient = th.linalg.norm(x-sample)/th.linalg.norm(gradient) * model_kwargs["scale"]
            # print(new_mean.grad_fn)
            # print('----------------------->>>>>>>>>>>>>>>', th.unique(gradient))
            # if dynamic_s:
            #     # print('awlkerjlawekjr')
            #     noise = th.randn_like(x)
            #     nonzero_mask = (
            #         (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            #     )  # no noise when t == 0
            #     sample = p_mean_var["mean"] + nonzero_mask * th.exp(0.5 * p_mean_var["log_variance"]) * noise
            #     coefficient = th.linalg.norm(x-sample)/th.linalg.norm(gradient)
                
            #     new_mean = (p_mean_var["mean"].float() +  gradient.float()*coefficient)
            # else:
            new_mean = (p_mean_var["mean"].float() +  gradient.float())
            new_mean.requires_grad_(True)
            # print(new_mean.grad_fn)
            optimizer.zero_grad()
            # print(new_mean.grad_fn)
            new_mean.backward(th.ones_like(new_mean))
            
            # gradient.backward()
            # print(gradient.grad_fn)
            optimizer.step()
            # for tag, parm in p_mean_var["mean"].named_parameters:
            # writer.add_histogram('hist', new_mean.grad.data.cpu().numpy(), 0)
                # writer.add_graph(p_mean_var, x)
            # writer.close()
            # sys.exit(0)
        # print("diff_mean", gradient)
        if t[0]>20:
            return gradient, new_mean
        else:
            return gradient, old_mean

    def condition_eps(self, cond_fn, noise_pred, x, pred_xstart, t, model_kwargs=None):

        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        # writer = SummaryWriter('abcdef')
        # gradient = cond_fn(pred_xstart, self._scale_timesteps(t), **model_kwargs)
        optimizer = th.optim.Adam([noise_pred], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        new_mean = noise_pred.float()
        # print("old_mean", new_mean)'
        old_mean = new_mean
        for i in range(1):
            # gradient = cond_fn(pred_xstart, self._scale_timesteps(t), **model_kwargs)
            gradient = cond_fn(pred_xstart, self._scale_timesteps(t))
            # print('----------------------->>>>>>>>>>>>>>>', th.unique(gradient))
            # new_mean = (noise_pred.float() +  0.001*gradient.float())
            B,C,_,_ = noise_pred.size()
            # min_z = th.min(gradient.view(B, C, -1), dim = 2, keepdim = True)[0].unsqueeze(dim = 2) # B, C, 1, 1
            # max_z = th.max(gradient.view(B, C, -1), dim = 2, keepdim = True)[0].unsqueeze(dim = 2) # B, C, 1, 1
            # gradient = (gradient - min_z)/(max_z - min_z)
            # gradient = th.group_norm(gradient, 4)
            
            new_mean = (noise_pred.float() + gradient.float())

            new_mean.requires_grad_(True)
            optimizer.zero_grad()
            new_mean.backward(th.ones_like(new_mean))
            # gradient.backward()
            # print(gradient.grad_fn)
            optimizer.step()
            # for tag, parm in p_mean_var["mean"].named_parameters:
            # writer.add_histogram('hist', new_mean.grad.data.cpu().numpy(), 0)
                # writer.add_graph(p_mean_var, x)
            # writer.close()
            # sys.exit(0)
        # print("diff_mean", gradient)
        return new_mean
    
    # #Le 
    # def get_timesteps(self, num_inference_steps, strength):
        
    #     # get the original timestep using init_timestep
    #     init_timestep = min(int(num_inference_steps * strength), num_inference_steps) #+ self.scheduler.config.get("steps_offset", 0)

    #     t_start = max(num_inference_steps - init_timestep, 0)
    #     timesteps = self.noise_scheduler.timesteps[t_start:]

    #     return timesteps, num_inference_steps - t_start
    
    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out
    
    def condition_score_sd(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        # eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = p_mean_var["noise_pred"]
        gradient = cond_fn(
            x, self._scale_timesteps(t)
        )
        scale = (1 - alpha_bar).sqrt()
        eps = eps -  scale*gradient 
        # correction = gradient - eps
        # eps = eps + 7.5 * correction

        # out = p_mean_var.copy()
        # out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        # out["mean"], _, _ = self.q_posterior_mean_variance(
        #     x_start=out["pred_xstart"], x_t=x, t=t
        # )
        return eps

    def pred_sample(self, x, t, start):
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        eps = self._predict_eps_from_xstart(x, t, start)
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eta = 1

        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = start * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps

        sample = mean_pred + nonzero_mask * sigma * noise

        return sample
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        IC = None,
        zs = None,
        vae = None,
        is_latentspace = None,
        text_embedding = None,
        freedom=None,
        model_ldm = None,
        index = None,
        sampler=None,
        approach2 = False,
        use_AdaIN = False,
        unet_grad = None,
        gt_t = None,
        dynamic_s=None,
        writer = None,
        ddim_num = None,
        steer_fn = None,
        # text_steer = False,
        # clip_encoder = None,
        # prompt = None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        if steer_fn is not None:
            with th.enable_grad():
                x = x.detach().requires_grad_()
                # x_in = x.detach().requires_grad_(True)

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            IC = IC,
            zs = zs,
            vae = vae,
            is_latentspace= is_latentspace,
            text_embedding= text_embedding,
            freedom=freedom,
            model_ldm = model_ldm,
            index = index,
            sampler=sampler,
            unet_grad = unet_grad,
            gt_t = gt_t,
            # text_steer = text_steer,
        )

        if is_latentspace:
            def process_xstart(x):
                return x.clamp(-1, 1)
            ##################conditional generation
            #############update e_t
            # new_eps = self.condition_eps(
            #         cond_fn, out["noise_pred"], x, out["pred_xstart"], t, model_kwargs=model_kwargs
            #     )
            # sample, pred_xstart = self.sample_from_eps(x, new_eps, model_ldm, sampler, index)
            # # new_eps = out["noise_pred"]
            # sample, pred_xstart = self.sample_from_eps(x, new_eps, model_ldm, sampler, index)
            # return {"sample": sample, "pred_xstart": pred_xstart}

            # ##############update previous sample x{t-1} on image space
            # x_prev = model_ldm.decode_first_stage(out["prev_sample"])

            # sample = self.condition_eps(
            #         cond_fn, x_prev, x, out["pred_xstart"], t, model_kwargs=model_kwargs
            #     )
            # z_prev = model_ldm.encode_first_stage(sample).sample()
            # z_prev = z_prev * 0.18215

            # return {"sample": z_prev, "pred_xstart": out["pred_xstart"]}
            
            # ##############update previous sample z{t-1} on latent space
            sample = self.condition_eps(
                    cond_fn, out["prev_sample"], x, out["pred_xstart"], t, model_kwargs=model_kwargs
                )
            
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}
            #################### update mean
            # _, _, posterior_log_variance = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=self._scale_timesteps(t))
            # pred_xstart = process_xstart(
            #             self._predict_xstart_from_eps(x_t=x, t=t, eps=out["noise_pred"])
            #         )
            # model_mean, _, posterior_log_variance = self.q_posterior_mean_variance(
            #     x_start=pred_xstart, x_t=x, t=t
            # )
            # # eps = self._predict_eps_from_xstart(x, t, pred_xstart)

            # # alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            # # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
            # # sigma = (
            # #     0.0
            # #     * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            # #     * th.sqrt(1 - alpha_bar / alpha_bar_prev)
            # # )
            # # # Equation 12.
            # # noise = th.randn_like(x)
            # # model_mean = (
            # #     pred_xstart * th.sqrt(alpha_bar_prev)
            # #     + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            # # )
            # # model_mean = self.condition_eps(
            # #     cond_fn, model_mean, x, model_mean, t, model_kwargs=model_kwargs
            # # )
            # b, _, _, _ = x.shape
            # noise = th.randn_like(x)
            # nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
            # sample = model_mean + nonzero_mask * (0.5 * posterior_log_variance).exp() * noise
           
            # return {"sample": sample, "pred_xstart": pred_xstart}

            

            ######################unconditional generation
            
            # return {"sample": out["prev_sample"], "pred_xstart": out["pred_xstart"]}

            
        else:
            # if t != 0:
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            scale = (1 - alpha_bar).sqrt()
            # scale = t[0]/self.num_timesteps
            # scale = t[0]/ddim_num
            # print(self.num_timesteps)
            # scale = None
            if approach2 & t[0]!= 0:
            # if approach2:#multiple GPUs
                # print('approach2...........................')
                # LE modified. using x_0|{t-1} approach 2
                # scale = None
                noise = th.randn_like(x)
                nonzero_mask = (
                    ((t - 1 ) != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                )  # no noise when t == 0
                x_prev = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise #x_{t-1}^u 
                out_x_prev = self.p_mean_variance(
                model,
                x_prev,
                t-1,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                IC = IC,
                zs = zs,
                vae = vae,
                is_latentspace= is_latentspace,
                text_embedding= text_embedding,
                freedom=freedom,
                model_ldm = model_ldm,
                index = index,
                sampler=sampler,
                )
                x0_prev = out_x_prev["pred_xstart"]

                noise = th.randn_like(x)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                )  # no noise when t == 0
                if cond_fn is not None:
                    # out["mean"] = self.condition_mean(
                    #     cond_fn, out, x, out["pred_xstart"], t, model_kwargs=model_kwargs
                    # )
                    if scale is not None:
                        out["mean"] = self.condition_mean(
                            cond_fn, out, x_prev, x0_prev, t, scale, model_kwargs=model_kwargs
                        )
                    else:
                        out["mean"] = self.condition_mean(
                            cond_fn, out, x_prev, x0_prev, t, model_kwargs=model_kwargs
                        )
                    out["pred_xstart"] = x0_prev

                    #using both x_0|{t-1} and x_0|t
                    # out["mean"] = self.condition_mean(
                    #     cond_fn, out, x, out["pred_xstart"], t, model_kwargs=model_kwargs
                    # )
            else: #app 1
                
                noise = th.randn_like(x)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                )  # no noise when t == 0
                # if use_AdaIN and t<500:
                #     out["pred_xstart"] = wavelet_reconstruction(out["pred_xstart"], IC)
                if steer_fn is not None:
                    grad_val_new = steer_fn(out["pred_xstart"])
                    # grad_val_new = -th.autograd.grad(loss_new, x, allow_unused=True)[0]
                    # scale =1 
                    out["pred_xstart"] = out["pred_xstart"] + grad_val_new
                    # sample = self.pred_sample(x, t, start)

                if cond_fn is not None:
                    if scale is not None:
                        out["mean"] = self.condition_mean(
                            cond_fn, out, x, out["pred_xstart"], t, scale, model_kwargs=model_kwargs
                        )
                        if writer is not None:
                            # writer.writerow([t[0].item(), format(scale.mean(dim=(0,1,2,3)).item(), ".4f")])
                            writer.writerow([t[0].item(), scale.item()])
                    else:
                        
                        out["mean"] = self.condition_mean(
                            cond_fn, out, x, out["pred_xstart"], t, model_kwargs=model_kwargs
                        )
                
            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

            
            # if text_steer:
            #     print(x_in.requires_grad) 
            #     print(x_in.grad_fn) 
            #     x_in.requires_grad_(True)
            #     residual = clip_encoder.get_residual(x, prompt)
            #     norm = torch.linalg.norm(residual)
            #     norm_grad = torch.autograd.grad(outputs=norm, inputs=x_in)[0]
            #     # l2 = l1 * 0.02
            #     # rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            #     rho = 1
            #     sample = sample -rho*norm_grad

            # if dynamic_s:
            #     coefficient = th.linalg.norm(x-sample)/th.linalg.norm(grad)
            #     sample -= (grad*coefficient)
            if unet_grad is not None:
                # print(sample.grad_fn )
                # sample.requires_grad_(True)
                loss = torch.nn.functional.mse_loss(sample.float(), gt_t.float(), reduction="mean")
                loss.requires_grad_(True)
                # print(sample.grad_fn)
                # print(gt_t.grad_fn)
                loss.backward()
            
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        

    

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        denoise_steps=None,
        mask_image =None,
        IC =None,
        roi_img = None,
        bg_img = None,
        CAS_fn = None,
        zs = None,
        vae = None,
        is_latentspace = False,
        text_embedding = None,
        freedom=None,
        model_ldm =None,
        sampler = None,
        approach2 = False,
        use_AdaIN = False,
        unet_grad = None,
        gt_img = None,
        dynamic_s = False,
        writer = None,
        steer_fn = None,
        # clip_encoder = None,
        # prompt = None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            denoise_steps=denoise_steps,
            mask_image = mask_image,
            IC = IC,
            roi_img = roi_img,
            bg_img = bg_img,
            CAS_fn = CAS_fn,
            zs = zs,
            vae = vae,
            is_latentspace = is_latentspace,
            text_embedding = text_embedding,
            freedom = freedom,
            model_ldm = model_ldm,
            sampler = sampler,
            approach2 = approach2,
            use_AdaIN = use_AdaIN,
            unet_grad = unet_grad,
            gt_img = gt_img,
            dynamic_s = dynamic_s,
            writer = writer,
            steer_fn = steer_fn,
            # text_steer = text_steer,
            # clip_encoder = clip_encoder,
            # prompt = prompt,
        ):
            final = sample
            # print(final["sample"].min())
            # print(final["sample"].max())
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        denoise_steps = None,
        mask_image = None,
        IC = None,
        roi_img = None,
        bg_img = None,
        CAS_fn = None,
        zs = None,
        vae = None,
        is_latentspace = False,
        text_embedding = None,
        freedom=None,
        model_ldm = None,
        sampler =None,
        approach2 =False,
        use_AdaIN = False,
        unet_grad = None,
        gt_img = None,
        dynamic_s = None,
        writer = None,
        steer_fn = None,
        # text_steer = False,
        # clip_encoder = None,
        # prompt = None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().

        denoise_steps is the manually set number of remaining denoising steps
        when denoise_steps is not None, noise serves as the intial x_0
        we first diffusion it to x_t, where t = denoise_steps,
        then we reversely denoise it t steps back to x_0 
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
            # print('img = noise')
        else:
            img = th.randn(*shape, device=device)
            # print('img = randn')

        if denoise_steps is None:
            indices = list(range(self.num_timesteps))[::-1] #49...0
            if noise is not None:
                # 100/(1000/self.num_timesteps)
                # print(self.num_timesteps)
                idx = int(self.num_timesteps/10)
                # print(idx)
                indices = indices[idx:] #44...0 #ddim50
                # print('self.num_timesteps', len(indices))
        else:
            assert noise is not None
            print('denoise_steps: ', denoise_steps)
            t_steps = th.ones(noise.shape[0], device=noise.device).long()
            t_steps = t_steps * (denoise_steps-1)
            img = self.q_sample(noise, t_steps) #forward
            
            indices = list(range(denoise_steps))[::-1]
        if freedom:
            sampler.make_schedule(ddim_num_steps=len(indices), ddim_eta=0.0, verbose=False)
        # indices from num_timesteps-1 to 0

        # when forward t = num_timesteps-1
        # backward indices start from num_timesteps-1 to 0
        # when forward t = denoise_steps
        # backward indices start from denoise_steps to 0

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)#44...0
        index = len(indices)-1
        for i in indices:
            
            t = th.tensor([i] * shape[0], device=device)
            # print("ttttttttttttttttttttttttttt", t)
            #le modified
            # rd_noise = th.randn_like(img, device=img.device)
            # # xT = noise_scheduler.add_noise(image_lr, noise, torch.LongTensor([900]))
            # IC_t = self.noise_scheduler.add_noise(IC, rd_noise, t)
            # img = img*(1 - mask_image) + IC_t*mask_image
            if unet_grad is not None:
                gt_t = self.q_sample(gt_img, t[0]-1)
            else:
                gt_t = None
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    IC = IC,
                    zs = zs,
                    vae = vae,
                    is_latentspace = is_latentspace,
                    text_embedding = text_embedding,
                    freedom=freedom,
                    model_ldm = model_ldm,
                    index = index,
                    sampler=sampler,
                    approach2 = approach2,
                    use_AdaIN = use_AdaIN,
                    unet_grad= unet_grad,
                    gt_t = gt_t,
                    dynamic_s = dynamic_s,
                    writer = writer,
                    ddim_num = indices[0],
                    steer_fn = steer_fn,
                    # text_steer = text_steer,
                    # clip_encoder = clip_encoder,
                    # prompt = prompt,
                )
                yield out
                img = out["sample"]
                # img_ = vae.decode(1 / 0.18215 * img).sample 
                # utils.save_image(img_, 'Imgs/step_%d.png' % (t), nrow=1, normalize=True)
                
                # Le modified - blending
                IC_t = self.q_sample(IC, t[0]-1)
                # img = CAS_fn(roi_img, bg_img, img)
                img = img*(1 - mask_image) + IC_t*mask_image
                
                
                # Le modified: using ROI as guidance in image space
                # with th.enable_grad():
                #     x_in_ = img.detach().requires_grad_(True)
                #     x_in_roi_ = x_in_*mask_image
                #     x_lr_roi_ = IC*mask_image
                #     mse = ((x_in_roi_+1)/2 - (x_lr_roi_+1)/2) ** 2
                #     mse = mse.mean(dim=(1,2,3))
                #     mse = mse.sum()
                #     loss_ = - mse * 20000
                #     grad_ = th.autograd.grad(loss_, x_in_)[0]
                # img = img - grad_/100
                # print(img.min())
                # print(img.max())
                # print(IC.min())
                # print(IC.max())
                # print(grad_)
                # print('----------------------->>>>>>>>>>>>>>>', th.unique(grad_))
                # print(mask_image.max())
                # utils.save_image(img, 'step_%d.png' % (t), nrow=1, normalize=False)
            index = index -1
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]


    def ddim_sample_loop_modified(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
        ):
            """
            Generate samples from the model using DDIM.

            Same usage as p_sample_loop().
            """
            final = None
            count =0
            for sample in self.ddim_sample_loop_progressive_modified(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
            ):
                final = sample
                count += 1
                # saving
                sample_saving = final["sample"]
                sample_saving = ((sample_saving + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample_saving = sample_saving.permute(0, 2, 3, 1)
                sample_saving = sample_saving.contiguous()

                sample_saving = sample_saving.detach().cpu().numpy()
                # classes = classes.detach().cpu().numpy()
                # image_lr = image_lr.detach().cpu().numpy()
                # print(sample.dtype)
                sample_img = th.Tensor(sample_saving)
                # print(sample_img.type())
                
                sample_img = rearrange(sample_img, 'a b c d -> a d b c').cuda()
                # print(sample_img.type())
                # # sample = image_lr_tensor
                sample_saving = sample_img 
                # mask_image.min
                save_dir_ = "/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/generate_images/generated_image_x0_GDP_inp_box_jpeg_s900/images/"
                filename = os.path.join(save_dir_, 'step_%d.png' % (count) )
                # sample_img = (sample_img+1)/2
                # Image.fromarray(sample[0]).save(filename)
                # print(filename)
                utils.save_image(sample_saving, filename, nrow=1, normalize=True)
            return final["sample"]


    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        
        indices = list(range(self.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def ddim_sample_loop_progressive_modified(# Le modified
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        cal_ts = False
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        # if cal_ts:
        #     self.noise_scheduler.set_timesteps(self.num_timesteps, device=device)
        #     timesteps, _ = self.get_timesteps(self.num_timesteps, strength = 0.9) #tensor [999...0]
        #     indices = timesteps.tolist() #[999...0]
        # else:
        indices = list(range(self.num_timesteps))[::-1]
        # indices_ = list(range(timesteps))[::-1]
        # indices = tqdm(indices_)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None, IC = None, zs = None,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, IC = IC, zs = zs,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
    
    #LE MODIFIED
    def training_losses_dual_unet(self, model, input_IC, x_start, t, zs=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            # print("LossType.KL")
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
                # print("LossType.RESCALED_KL")
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # model_output = model(input_IC, x_t, self._scale_timesteps(t), **model_kwargs)
        #     out = self.p_mean_variance(
        #     model.unet,
        #     x_t,
        #     t,
        #     clip_denoised=True,
        #     model_kwargs=model_kwargs,
        #     IC = input_IC,
        #     zs = zs,
        # )
            # print(out["pred_xstart"].size())
            # model_output = model(input_IC, x_t, self._scale_timesteps(t), zs, x_0 = out["pred_xstart"])
            # model_output = model(input_IC, x_t, self._scale_timesteps(t), zs)
            # print('zsssssssssssssssss', zs)
            model_output = model(input_IC, x_t, self._scale_timesteps(t))
            # print("LossType.MSE")#used
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    IC = input_IC,
                    zs = zs,
                )["output"]#used
                # print("LossType.LEARNED")
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
                    # print("LossType.RESCALED_MSE")

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                # print('mse',terms["mse"])
                # print('vb',terms["vb"])
                terms["loss"] = terms["mse"] + terms["vb"]#used
                # print("vb")
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
    
    def training_losses_dual_unet_structural(self, model, input_IC, x_start, t, structural_loss = False, CL_loss = False, model_kwargs=None, noise=None, I_52 = None, use_LPIPS_loss = False, LPIPS_cal = None, mask = None, emb_text = None, text_loss = False, processor = None, clip_model=None, both_text = False):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            # print("LossType.KL")
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
                # print("LossType.RESCALED_KL")
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # model_output = model(input_IC, x_t, self._scale_timesteps(t), **model_kwargs)
        #     out = self.p_mean_variance(
        #     model.unet,
        #     x_t,
        #     t,
        #     clip_denoised=True,
        #     model_kwargs=model_kwargs,
        #     IC = input_IC,
        #     zs = zs,
        # )
            # print(out["pred_xstart"].size())
            # model_output = model(input_IC, x_t, self._scale_timesteps(t), zs, x_0 = out["pred_xstart"])
            # model_output = model(input_IC, x_t, self._scale_timesteps(t), zs)
            # print('zsssssssssssssssss', zs)

            if both_text or (not text_loss and emb_text is not None):
                model_output = model(input_IC, x_t, self._scale_timesteps(t), emb_text=emb_text)
            else:
                model_output = model(input_IC, x_t, self._scale_timesteps(t))
            # model_output = model(x_t, self._scale_timesteps(t))
            #model_output size B,6,256,256
            # print("LossType.MSE")#used
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1) #noise, 
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    IC = input_IC,
                )["output"]#used
                # print("LossType.LEARNED")
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
                    # print("LossType.RESCALED_MSE")

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,#used
            }[self.model_mean_type]
            # print('model_mean_type',self.model_mean_type)
            assert model_output.shape == target.shape == x_start.shape
            # print(target.size())
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                if structural_loss:
                    def process_xstart(x):
                        return x.clamp(-1, 1)
                    pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
                    )
                    
                    # utils.save_image(pred_xstart[0:1,:,:,:], '/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/Imgs/step_%d.png' % (t[0]), nrow=1, normalize=True)
                    # utils.save_image(x_start[0:1,:,:,:], '/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/Imgs/gt_step_%d.png' % (t[0]), nrow=1, normalize=True)
                    # print("laksjdflakjweriwerpweiopk")
                    # print(pred_xstart.size())
                    # print(x_start.min())
                    # print(x_start.max())
                    # print(pred_xstart.min())
                    # print(pred_xstart.max())

                    if CL_loss:
                        #triplet loss
                        # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
                        # net = VGG16().cuda()
                        # struct_loss = triplet_loss(net(pred_xstart)[4], net(x_start)[4], net(I_52)[4])*0.1

                        # utils.save_image(pred_xstart[0:1,:,:,:], '/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/Imgs/step_%d.png' % (t[0]), nrow=1, normalize=True)
                        # utils.save_image(x_start[0:1,:,:,:], '/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/Imgs/gt_step_%d.png' % (t[0]), nrow=1, normalize=True)
                        # utils.save_image(I_52[0:1,:,:,:], '/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/Imgs/I52_step_%d.png' % (t[0]), nrow=1, normalize=True)
                        #contrastive learning loss
                        def calc_euclidean(x1, x2):
                            return (x1 - x2).pow(2).sum(1)
                        net = VGG16().cuda()
                        distance_positive = calc_euclidean(net(pred_xstart)[4], net(x_start)[4])
                        distance_negative = calc_euclidean(net(pred_xstart)[4], net(I_52)[4])
                        struct_loss = mean_flat((distance_positive / (distance_positive + distance_negative)))*0.01
                    elif use_LPIPS_loss:
                        # print("LPIPS loss")
                        scale = (1 - (t/1000))*0.01
                        # scale = 0.01
                        # pred_xstart = input_IC*mask + (1-mask)*pred_xstart
                        # utils.save_image(pred_xstart[0:1,:,:,:], '/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/Imgs/step_%d.png' % (t[0]), nrow=1, normalize=True)
                        # utils.save_image(mask[0:1,:,:,:], '/media/EXT0/daole/sd_scripts/GenerativeDiffusionPrior/scripts/Imgs/mask_step_%d.png' % (t[0]), nrow=1, normalize=True)
                        struct_loss = LPIPS_cal(pred_xstart, x_start)*scale
                    else:
                        #L2 
                        
                        # struct_loss = mean_flat((pred_xstart - x_start) ** 2)*scale
                        struct_loss = mean_flat((pred_xstart - x_start) ** 2)
                        if text_loss:
                            if isinstance(pred_xstart, torch.Tensor):
                                pred_xstart_ = pred_xstart.detach().cpu().numpy()  # Move to CPU and convert to NumPy

                            # If pred_xstart is a batch of images (B, C, H, W), convert each to PIL format
                            # if isinstance(pred_xstart, np.ndarray):
                            #     pred_xstart_ = [Image.fromarray((img * 255).astype("uint8").transpose(1, 2, 0)) for img in pred_xstart]
                            # Convert batch of (B, 3, 256, 256) images to a list of PIL images
                            batch_size = pred_xstart_.shape[0]
                            pil_images = []

                            for i in range(batch_size):
                                img = np.transpose(pred_xstart_[i], (1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
                                img = Image.fromarray((img * 255).astype("uint8"))  # Convert to PIL
                                pil_images.append(img)

                            # Process batch of images with CLIPProcessor
                            inputs = processor(images=pil_images, return_tensors="pt")
                            # inputs = processor(images=pred_xstart_, return_tensors="pt")
                            # Ensure inputs are on the correct device (match img_encoder's device)
                            device = next(clip_model.parameters()).device
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                            

                            with torch.no_grad():
                                emb_img = clip_model.get_image_features(**inputs)  # (B, 768)
                            emb_text = emb_text[:, 0, :]
                            emb_text = emb_text.to(device)

                            # with torch.no_grad():
                            #     img_outputs = img_encoder(**inputs)  # Get full output
                            #     emb_img = img_outputs.last_hidden_state  # (B, num_patches, 1024)
                            # emb_img = emb_img[:, 0, :]  # (B, 1024) -> Take only the first token

                            # # Project to 768 dimensions using CLIP's projection layer
                            # emb_img = img_encoder.visual_projection(emb_img)  # (B, 768)

                            # # Now emb_img has shape (B, 768), but emb_text is (B, 77, 768)
                            # # Reduce emb_text to match (B, 768) by taking the CLS token
                            # emb_text = emb_text[:, 0, :]  # (B, 77, 768) → (B, 768)

                            # # Ensure emb_img has the same shape as emb_text: (B, 77, 768)
                            # if emb_img.shape[1] != 77:
                            #     emb_img = torch.nn.functional.interpolate(emb_img.permute(0, 2, 1), size=77).permute(0, 2, 1)

                            txt_loss = mean_flat((emb_text - emb_img) ** 2)
                            struct_loss = struct_loss + 0.01* txt_loss
                        #using mask: L2(x0_hat*(1-mask)+I^D*mask, gt)
                        # scale = 1 - (t/1000)
                        # scale = _extract_into_tensor(self.alphas_cumprod, t, x_start.shape).mean()
                        # pred_xstart = input_IC*mask + (1-mask)*pred_xstart
                        # struct_loss = mean_flat((pred_xstart - x_start) ** 2)*scale
                        #L1
                        # scale = 1 - (t/1000)
                        
                        # struct_loss = abs(mean_flat((pred_xstart - x_start)))
                        # print(terms["mse"])
                        # print(terms["vb"])
                        # print(struct_loss)
                        
                    terms["loss"] = terms["mse"] + terms["vb"]+ struct_loss
                    
                else:
                    
                    terms["loss"] = terms["mse"] + terms["vb"]#used
                    
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms
    
    def validation_cal(self, model, input_IC, x_start, t, zs, mask, lpips_metric, model_kwargs=None, noise=None, emb_text = None, text_loss = False, both_text = False):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        def process_xstart(x):
            return x.clamp(-1, 1)
        
        inv_mask = 1 - mask
        with th.no_grad():
            if model_kwargs is None:
                model_kwargs = {}
            if noise is None:
                noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start, t, noise=noise)
            
            terms = {}
            # model_output = model(input_IC, x_t, self._scale_timesteps(t), zs)
            if both_text or (emb_text is not None and text_loss == False):
                model_output = model(input_IC, x_t, self._scale_timesteps(t), emb_text = emb_text)
            else:
                model_output = model(input_IC, x_t, self._scale_timesteps(t))
            # model_output = model(input_IC, x_t, self._scale_timesteps(t))
            model_output, model_var_values = th.split(model_output, input_IC.shape[1], dim=1)
            pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
                )
            # pred_xstart = process_xstart(model_output)
            lpips_score = lpips_metric(pred_xstart*inv_mask+input_IC*mask, x_start*inv_mask+input_IC*mask)

        return lpips_score

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
