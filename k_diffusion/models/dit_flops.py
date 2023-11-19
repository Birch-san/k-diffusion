from torch.nn import Module, Linear
from timm.models.vision_transformer import Attention
from torch import FloatTensor
from typing import Tuple

from . import flops
from ..external import DiTDenoiser
from .flops import hook_linear_flops


def hook_attn_flops(attn: Attention, args: Tuple[FloatTensor, ...], _):
    x, *_ = args
    B, N, _ = x.shape
    qkv_out_channels: int = attn.qkv.out_features
    per_proj_out_channels: int = qkv_out_channels // 3
    head_dim: int = per_proj_out_channels // attn.num_heads
    proj_shape = B, attn.num_heads, N, head_dim
    flops.op(flops.op_attention, proj_shape, proj_shape, proj_shape)

def instrument_module(module: Module):
    if isinstance(module, Linear):
        module.register_forward_hook(hook_linear_flops)
    elif isinstance(module, Attention):
        module.register_forward_hook(hook_attn_flops)

def instrument_dit_flops(model: DiTDenoiser):
    model.apply(instrument_module)