"""
Helpers to train with 16-bit precision.
"""

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import logger
import torch.utils.tensorboard as tfboard
import os
INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        
        # new_params = []
        # for name, param in param_group:
        #     if "unet" not in name:
        #         new_params.append(param.detach().float())

        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
            # _flatten_dense_tensors(
            #     new_params
            # ).view(shape)
        )
        master_param.requires_grad = True  
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)

    # for (n, p) in named_model_params:
    #     print("N:",n)
    #     print("P-size",p.size())
    #     print(p.ndim)
    #     # print("P:",p)

    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        #original
        # for i, (name, _value) in enumerate(model.named_parameters()):
        #     assert name in state_dict
        #     state_dict[name] = master_params[i]

        #LE modified
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:

        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters() if "unet" not in name
        ]

        param_groups_and_shapes = get_param_groups_and_shapes(
            named_model_params
        )
    
        master_params = make_master_params(param_groups_and_shapes)


    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())



        # # LE MODIFIED
        # tf_board_logs = 'TensorBoard/'
        # # time_before = datetime.datetime.now()
        # self.tfb_train_writer = tfboard.SummaryWriter(log_dir=os.path.join(tf_board_logs, 'Time_Aware_Encoder_GD'))
        # parameters = []
        # parameters.extend(self.model.time_aware_enc.parameters())
        # self.master_params = parameters

        # parameters = list(self.model.time_aware_enc.parameters()) +\
        #       list(self.model.spade_256.parameters()) +\
        #         list(self.model.conv_256.parameters()) +\
        #         list(self.model.spade_128.parameters()) +\
        #         list(self.model.conv_128.parameters()) +\
        #         list(self.model.spade_64.parameters()) +\
        #         list(self.model.conv_64.parameters()) +\
        #         list(self.model.spade_32.parameters()) +\
        #         list(self.model.conv_32.parameters()) +\
        #         list(self.model.spade_256_dec.parameters()) +\
        #         list(self.model.conv_256_dec.parameters()) +\
        #         list(self.model.spade_128_dec.parameters()) +\
        #         list(self.model.conv_128_dec.parameters()) +\
        #         list(self.model.spade_64_dec.parameters()) +\
        #         list(self.model.conv_64_dec.parameters()) +\
        #         list(self.model.spade_32_dec.parameters()) +\
        #         list(self.model.conv_32_dec.parameters())
        
        # self.model_params = parameters

        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            # # params = {"name":[], "param":[]}
            params = []
            names = []
            for name, param in self.model.named_parameters():
                if "unet" not in name:
                    # print(name)
                    # params["name"].append(name)
                    # params["param"].append(param)
                    # params.append([name, param])
                    params.append(param)
                    names.append(name)

            parameter = zip(names, params)

            self.param_groups_and_shapes = get_param_groups_and_shapes(
                parameter
            )
            # self.param_groups_and_shapes = get_param_groups_and_shapes(
            #     self.model.named_parameters()
            # )
        
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()
        else:
            # # params = {"name":[], "param":[]}
            params = []
            names = []
            for name, param in self.model.named_parameters():
                if "unet" not in name:
                    # print(name)
                    # params["name"].append(name)
                    # params["param"].append(param)
                    # params.append([name, param])
                    params.append(param)
                    names.append(name)

            parameter = zip(names, params)

            self.param_groups_and_shapes = get_param_groups_and_shapes(
                parameter
            )
            # self.param_groups_and_shapes = get_param_groups_and_shapes(
            #     self.model.named_parameters()
            # )
        
            self.master_params = make_master_params(self.param_groups_and_shapes)

    def get_tensor_value(self, x):
        # check if x is a GPU tensor
        # print("x", x)
        try:
            if x.is_cuda:
                return x.cpu()
            else:
                return x
        except:
            return x
        
    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            loss_scaled = loss * loss_scale
            # print('loss.requires_grad', (loss * loss_scale).requires_grad)
            # print('loss.grad_fn', (loss * loss_scale).grad_fn)
            
            loss_scaled.backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer, step):
        if self.use_fp16:
            return self._optimize_fp16(opt, step)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer, step):
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale)) #original
        # for p in self.master_params:
        #     p.grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()

        # for name, param in self.model.named_parameters():
        #                                 # without bias
        #                                 if "bias" not in name:
        #                                     # print("BIASSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
        #                                     # writer_val.add_histogram("param/" + name, param.clone().cpu().data.numpy(), epoch)
        #                                     self.tfb_train_writer.add_histogram("param/" + name, self.get_tensor_value(param).data.numpy(), step)
        #                                     # icip_net grads
        #                                     if param.grad is not None:
        #                                         # writer_val.add_histogram("grad/" + name, param.grad.cpu(), epoch)
        #                                         self.tfb_train_writer.add_histogram("grad/" + name, self.get_tensor_value(param.grad), step)

        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
