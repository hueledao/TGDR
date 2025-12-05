import warnings
from typing import Tuple, Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from Network.ade20k import ModelBuilder

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

class ResNetPL(nn.Module):
    def __init__(self, weight=1,
                 weights_path=None, arch_encoder='resnet50dilated', segmentation=False):
        super().__init__()
        print('*'*10, weights_path)
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        result = torch.stack([F.mse_loss(cur_pred, cur_target)
                              for cur_pred, cur_target
                              in zip(pred_feats, target_feats)]).sum() * self.weight
        return result

def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty        

class BaseAdversarialLoss:
    def pre_generator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           generator: nn.Module, discriminator: nn.Module):
        """
        Prepare for generator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        """
        Prepare for discriminator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def generator_loss(self, discr_fake_pred: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate generator loss
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total generator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate discriminator loss and call .backward() on it
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total discriminator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def interpolate_mask(self, mask, shape):
        assert mask is not None
        assert self.allow_scale_mask or shape == mask.shape[-2:]
        if shape != mask.shape[-2:] and self.allow_scale_mask:
            if self.mask_scale_mode == 'maxpool':
                mask = F.adaptive_max_pool2d(mask, shape)
            else:
                mask = F.interpolate(mask, size=shape, mode=self.mask_scale_mode)
        return mask


def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty


class NonSaturatingWithR1(BaseAdversarialLoss):
    def __init__(self, gp_coef=5, weight=1, mask_as_fake_target=False, allow_scale_mask=False,
                 mask_scale_mode='nearest', extra_mask_weight_for_gen=0,
                 use_unmasked_for_gen=True, use_unmasked_for_discr=True):
        self.gp_coef = gp_coef
        self.weight = weight
        # use for discr => use for gen;
        # otherwise we teach only the discr to pay attention to very small difference
        assert use_unmasked_for_gen or (not use_unmasked_for_discr)
        # mask as target => use unmasked for discr:
        # if we don't care about unmasked regions at all
        # then it doesn't matter if the value of mask_as_fake_target is true or false
        assert use_unmasked_for_discr or (not mask_as_fake_target)
        self.use_unmasked_for_gen = use_unmasked_for_gen
        self.use_unmasked_for_discr = use_unmasked_for_discr
        self.mask_as_fake_target = mask_as_fake_target
        self.allow_scale_mask = allow_scale_mask
        self.mask_scale_mode = mask_scale_mode
        self.extra_mask_weight_for_gen = extra_mask_weight_for_gen

    def generator_loss(self, discr_fake_pred: torch.Tensor, mask=None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_loss = F.softplus(-discr_fake_pred)
        if (self.mask_as_fake_target and self.extra_mask_weight_for_gen > 0) or \
                not self.use_unmasked_for_gen:  # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            if not self.use_unmasked_for_gen:
                fake_loss = fake_loss * mask
            else:
                pixel_weights = 1 + mask * self.extra_mask_weight_for_gen
                fake_loss = fake_loss * pixel_weights

        return fake_loss.mean() * self.weight, dict()

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        real_batch.requires_grad = True

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor,
                           mask=None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, real_batch) * self.gp_coef
        fake_loss = F.softplus(discr_fake_pred)

        if not self.use_unmasked_for_discr or self.mask_as_fake_target:
            # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            # use_unmasked_for_discr=False only makes sense for fakes;
            # for reals there is no difference beetween two regions
            fake_loss = fake_loss * mask
            if self.mask_as_fake_target:
                fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)

        sum_discr_loss = real_loss + grad_penalty + fake_loss
        metrics = dict(discr_real_out=discr_real_pred.mean(),
                       discr_fake_out=discr_fake_pred.mean(),
                       discr_real_gp=grad_penalty)
        return sum_discr_loss.mean(), metrics

    def discriminator_real_loss(self, real_batch: torch.Tensor, discr_real_pred: torch.Tensor):

        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, real_batch) * self.gp_coef

        sum_discr_loss = real_loss + grad_penalty
        return sum_discr_loss.mean(), real_loss, grad_penalty

    def discriminator_fake_loss(self, discr_fake_pred: torch.Tensor, mask=None) -> Tuple[torch.Tensor]:

        fake_loss = F.softplus(discr_fake_pred)

        if not self.use_unmasked_for_discr or self.mask_as_fake_target:
            # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            # use_unmasked_for_discr=False only makes sense for fakes;
            # for reals there is no difference beetween two regions
            fake_loss = fake_loss * mask
            if self.mask_as_fake_target:
                fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)

        sum_discr_loss = fake_loss
        return sum_discr_loss.mean()


def feature_matching_loss(fake_features: List[torch.Tensor], target_features: List[torch.Tensor], mask=None):
    if mask is None:
        res = torch.stack([F.mse_loss(fake_feat, target_feat)
                           for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
    else:
        res = 0
        norm = 0
        for fake_feat, target_feat in zip(fake_features, target_features):
            cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
            error_weights = 1 - cur_mask
            cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
            res = res + cur_val
            norm += 1
        res = res / norm
    return res

class VGG19(nn.Module):
    def __init__(self, resize_input=False):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features

        self.resize_input = resize_input
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        prefix = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        posfix = [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        names = list(zip(prefix, posfix))
        self.relus = []
        for pre, pos in names:
            self.relus.append('relu{}_{}'.format(pre, pos))
            self.__setattr__('relu{}_{}'.format(
                pre, pos), torch.nn.Sequential())

        nums = [[0, 1], [2, 3], [4, 5, 6], [7, 8],
                [9, 10, 11], [12, 13], [14, 15], [16, 17],
                [18, 19, 20], [21, 22], [23, 24], [25, 26],
                [27, 28, 29], [30, 31], [32, 33], [34, 35]]

        for i, layer in enumerate(self.relus):
            for num in nums[i]:
                self.__getattr__(layer).add_module(str(num), features[num])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # resize and normalize input for pretrained vgg19
        x = (x + 1.0) / 2.0
        x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))
        if self.resize_input:
            x = F.interpolate(
                x, size=(256, 256), mode='bilinear', align_corners=True)
        features = []
        for layer in self.relus:
            x = self.__getattr__(layer)(x)
            features.append(x)
        out = {key: value for (key, value) in list(zip(self.relus, features))}
        return out
class Style(nn.Module):
    def __init__(self):
        super(Style, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f'relu{pre}_{pos}']), self.compute_gram(y_vgg[f'relu{pre}_{pos}']))
        return style_loss    

class Perceptual(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss        