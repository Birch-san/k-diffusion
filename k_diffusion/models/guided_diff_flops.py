from torch.nn import Module, Linear, Conv1d, Conv2d
from torch import FloatTensor
from typing import Tuple
from guided_diffusion.unet import UNetModel, QKVAttention, QKVAttentionLegacy

from . import flops
from .flops import hook_linear_flops, hook_conv1d_flops, hook_conv2d_flops


def hook_attn_flops(attn: QKVAttention, args: Tuple[FloatTensor, ...], _):
    qkv, *_ = args
    B, qkv_out_channels, N = qkv.shape
    per_proj_out_channels: int = qkv_out_channels // 3
    head_dim: int = per_proj_out_channels // attn.n_heads
    proj_shape = B, attn.n_heads, N, head_dim
    flops.op(flops.op_attention, proj_shape, proj_shape, proj_shape)

def instrument_module(module: Module):
    if isinstance(module, Linear):
        module.register_forward_hook(hook_linear_flops)
    # TODO: check correctness
    if isinstance(module, Conv1d):
        module.register_forward_hook(hook_conv1d_flops)
    # TODO: check correctness (even more so)
    if isinstance(module, Conv2d):
        module.register_forward_hook(hook_conv2d_flops)
    elif isinstance(module, QKVAttention) or isinstance(module, QKVAttentionLegacy):
        module.register_forward_hook(hook_attn_flops)

def instrument_gdiff_flops(model: UNetModel):
    model.apply(instrument_module)