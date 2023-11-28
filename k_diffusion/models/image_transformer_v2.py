"""k-diffusion transformer diffusion models, version 2."""

from dataclasses import dataclass
from functools import lru_cache, reduce
import math
from typing import Union, Optional, Sequence, Literal, Tuple

from einops import rearrange
import torch
from torch import nn, FloatTensor, BoolTensor
import torch._dynamo
from torch.nn import functional as F
import numpy as np

from . import flags, flops
from .. import layers
from .axial_rope import make_axial_pos


try:
    import natten
except ImportError:
    natten = None

try:
    import flash_attn
except ImportError:
    flash_attn = None


if flags.get_use_compile():
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True


# Helpers

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def checkpoint(function, *args, **kwargs):
    if flags.get_checkpointing():
        kwargs.setdefault("use_reentrant", True)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)


def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)


# Param tags

def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param


def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module


def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


# Kernels

@flags.compile_wrap
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


@flags.compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


@flags.compile_wrap
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


@flags.compile_wrap
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


# Layers

class Linear(nn.Linear):
    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return super().forward(x)


class LinearGELU(nn.Linear):
    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return F.gelu(super().forward(x))


class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        flops.op(flops.op_linear, x.shape, self.weight.shape)
        return linear_geglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6, new_dims=2):
        super().__init__()
        self.eps = eps
        self.new_dims = new_dims
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        cond = self.linear(cond)
        cond = cond[:, *(None,)*self.new_dims, :] + 1
        return rms_norm(x, cond, self.eps)


# Rotary position embeddings

@flags.compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


@flags.compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


# [licensed code]
# Normal additive position embeddings
# From https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# license: CC BY-NC 4.0
# https://raw.githubusercontent.com/facebookresearch/mae/main/LICENSE
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
# [/licensed code]


# Shifted window attention

def window(window_size, x):
    *b, h, w, c = x.shape
    x = torch.reshape(
        x,
        (*b, h // window_size, window_size, w // window_size, window_size, c),
    )
    x = torch.permute(
        x,
        (*range(len(b)), -5, -3, -4, -2, -1),
    )
    return x


def unwindow(x):
    *b, h, w, wh, ww, c = x.shape
    x = torch.permute(x, (*range(len(b)), -5, -3, -4, -2, -1))
    x = torch.reshape(x, (*b, h * wh, w * ww, c))
    return x


def shifted_window(window_size, window_shift, x):
    x = torch.roll(x, shifts=(window_shift, window_shift), dims=(-2, -3))
    windows = window(window_size, x)
    return windows


def shifted_unwindow(window_shift, x):
    x = unwindow(x)
    x = torch.roll(x, shifts=(-window_shift, -window_shift), dims=(-2, -3))
    return x


@lru_cache
def make_shifted_window_masks(n_h_w, n_w_w, w_h, w_w, shift, device=None):
    ph_coords = torch.arange(n_h_w, device=device)
    pw_coords = torch.arange(n_w_w, device=device)
    h_coords = torch.arange(w_h, device=device)
    w_coords = torch.arange(w_w, device=device)
    patch_h, patch_w, q_h, q_w, k_h, k_w = torch.meshgrid(
        ph_coords,
        pw_coords,
        h_coords,
        w_coords,
        h_coords,
        w_coords,
        indexing="ij",
    )
    is_top_patch = patch_h == 0
    is_left_patch = patch_w == 0
    q_above_shift = q_h < shift
    k_above_shift = k_h < shift
    q_left_of_shift = q_w < shift
    k_left_of_shift = k_w < shift
    m_corner = (
        is_left_patch
        & is_top_patch
        & (q_left_of_shift == k_left_of_shift)
        & (q_above_shift == k_above_shift)
    )
    m_left = is_left_patch & ~is_top_patch & (q_left_of_shift == k_left_of_shift)
    m_top = ~is_left_patch & is_top_patch & (q_above_shift == k_above_shift)
    m_rest = ~is_left_patch & ~is_top_patch
    m = m_corner | m_left | m_top | m_rest
    return m


def apply_window_attention(window_size, window_shift, q, k, v, scale=None):
    # prep windows and masks
    q_windows = shifted_window(window_size, window_shift, q)
    k_windows = shifted_window(window_size, window_shift, k)
    v_windows = shifted_window(window_size, window_shift, v)
    b, heads, h, w, wh, ww, d_head = q_windows.shape
    mask = make_shifted_window_masks(h, w, wh, ww, window_shift, device=q.device)
    q_seqs = torch.reshape(q_windows, (b, heads, h, w, wh * ww, d_head))
    k_seqs = torch.reshape(k_windows, (b, heads, h, w, wh * ww, d_head))
    v_seqs = torch.reshape(v_windows, (b, heads, h, w, wh * ww, d_head))
    mask = torch.reshape(mask, (h, w, wh * ww, wh * ww))

    # do the attention here
    flops.op(flops.op_attention, q_seqs.shape, k_seqs.shape, v_seqs.shape)
    qkv = F.scaled_dot_product_attention(q_seqs, k_seqs, v_seqs, mask, scale=scale)

    # unwindow
    qkv = torch.reshape(qkv, (b, heads, h, w, wh, ww, d_head))
    return shifted_unwindow(window_shift, qkv)


# Transformer layers


def use_flash_2(x):
    if not flags.get_use_flash_attention_2():
        return False
    if flash_attn is None:
        return False
    if x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0, use_rope=True):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        if self.use_rope:
            pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
            theta = self.pos_emb(pos)
        if use_flash_2(qkv):
            qkv = rearrange(qkv, "n h w (t nh e) -> n (h w) t nh e", t=3, e=self.d_head)
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            if self.use_rope:
                theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
                qkv = apply_rotary_emb_(qkv, theta)
            flops_shape = qkv.shape[-5], qkv.shape[-2], qkv.shape[-4], qkv.shape[-1]
            flops.op(flops.op_attention, flops_shape, flops_shape, flops_shape)
            x = flash_attn.flash_attn_qkvpacked_func(qkv, softmax_scale=1.0)
            x = rearrange(x, "n (h w) nh e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        else:
            q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            if self.use_rope:
                theta = theta.movedim(-2, -3)
                q = apply_rotary_emb_(q, theta)
                k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_attention, q.shape, k.shape, v.shape)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0, use_rope=True):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
        if self.use_rope:
            theta = self.pos_emb(pos).movedim(-2, -4)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
        qk = natten.functional.natten2dqk(q, k, self.kernel_size, 1)
        a = torch.softmax(qk, dim=-1)
        x = natten.functional.natten2dav(a, v, self.kernel_size, 1)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class ShiftedWindowSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, window_size, window_shift, dropout=0.0, use_rope=True):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.window_size = window_size
        self.window_shift = window_shift
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head}, window_size={self.window_size}, window_shift={self.window_shift}"

    def forward(self, x, pos, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
        if self.use_rope:
            theta = self.pos_emb(pos).movedim(-2, -4)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
        x = apply_window_attention(self.window_size, self.window_shift, q, k, v, scale=1.0)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class CrossAttentionBlock(nn.Module):
    qk_scale: Optional[nn.Parameter]
    def __init__(self, d_model: int, d_cross: int, d_head: int, cond_features: int, scale_qk: bool, dropout=0.):
        super().__init__()
        self.d_head = d_head
        self.dropout = dropout
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.q_proj = apply_wd(Linear(d_model, d_model, bias=False))
        self.crossattn_norm = AdaRMSNorm(d_cross, cond_features, new_dims=1)
        self.kv_proj = apply_wd(Linear(d_cross, d_model * 2, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.qk_scale = nn.Parameter(torch.full([self.n_heads], 10.0)) if scale_qk else None
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x: FloatTensor, cond: FloatTensor, crossattn_cond: FloatTensor, crossattn_mask: Optional[BoolTensor] = None):
        skip = x
        x = self.norm(x, cond)
        q = self.q_proj(x)
        crossattn_cond = self.crossattn_norm(crossattn_cond, cond)
        kv = self.kv_proj(crossattn_cond)
        q = rearrange(q, "n h w (nh e) -> n nh (h w) e", e=self.d_head)
        k, v = rearrange(kv, "n l (t nh e) -> t n nh l e", t=2, e=self.d_head)
        if self.qk_scale is not None:
            q, k = scale_for_cosine_sim(q, k, self.qk_scale[:, None, None], 1e-6)
        # broadcast masked keys over every head and every query
        crossattn_mask = rearrange(crossattn_mask, "n l -> n 1 1 l")
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=crossattn_mask)
        x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=skip.shape[-3], w=skip.shape[-2])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0, up_proj_type=LinearGEGLU, up_bias=False):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        # TODO swap here
        self.up_proj = apply_wd(up_proj_type(d_model, d_ff, bias=up_bias))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, cross_attn: Optional[CrossAttentionBlock] = None, dropout=0.0, up_proj_type=LinearGEGLU, use_rope=True, ffn_up_bias=False):
        super().__init__()
        self.self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout, use_rope=use_rope)
        self.cross_attn = cross_attn
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout, up_proj_type=up_proj_type, up_bias=ffn_up_bias)

    def forward(self, x, pos, cond, crossattn_cond: Optional[FloatTensor] = None, crossattn_mask: Optional[BoolTensor] = None):
        x = checkpoint(self.self_attn, x, pos, cond)
        if self.cross_attn is not None:
            x = checkpoint(self.cross_attn, x, cond, crossattn_cond, crossattn_mask)
        x = checkpoint(self.ff, x, cond)
        return x


class NeighborhoodTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, kernel_size, cross_attn: Optional[CrossAttentionBlock] = None, dropout=0.0, up_proj_type=LinearGEGLU, use_rope=True, ffn_up_bias=False):
        super().__init__()
        self.self_attn = NeighborhoodSelfAttentionBlock(d_model, d_head, cond_features, kernel_size, dropout=dropout, use_rope=use_rope)
        self.cross_attn = cross_attn
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout, up_proj_type=up_proj_type, up_bias=ffn_up_bias)

    def forward(self, x, pos, cond, crossattn_cond: Optional[FloatTensor] = None, crossattn_mask: Optional[BoolTensor] = None):
        x = checkpoint(self.self_attn, x, pos, cond)
        if self.cross_attn is not None:
            x = checkpoint(self.cross_attn, x, cond, crossattn_cond, crossattn_mask)
        x = checkpoint(self.ff, x, cond)
        return x


class ShiftedWindowTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, window_size, index, cross_attn: Optional[CrossAttentionBlock] = None, dropout=0.0, up_proj_type=LinearGEGLU, use_rope=True, ffn_up_bias=False):
        super().__init__()
        window_shift = window_size // 2 if index % 2 == 1 else 0
        self.self_attn = ShiftedWindowSelfAttentionBlock(d_model, d_head, cond_features, window_size, window_shift, dropout=dropout, use_rope=use_rope)
        self.cross_attn = cross_attn
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout, up_proj_type=up_proj_type, up_bias=ffn_up_bias)

    def forward(self, x, pos, cond, crossattn_cond: Optional[FloatTensor] = None, crossattn_mask: Optional[BoolTensor] = None):
        x = checkpoint(self.self_attn, x, pos, cond)
        if self.cross_attn is not None:
            x = checkpoint(self.cross_attn, x, cond, crossattn_cond, crossattn_mask)
        x = checkpoint(self.ff, x, cond)
        return x


class NoAttentionTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0, up_proj_type=LinearGEGLU, ffn_up_bias=False):
        super().__init__()
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout, up_proj_type=up_proj_type, up_bias=ffn_up_bias)

    def forward(self, x, pos, cond, crossattn_cond: Optional[FloatTensor] = None, crossattn_mask: Optional[BoolTensor] = None):
        x = checkpoint(self.ff, x, cond)
        return x


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


# Mapping network

class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0, up_proj_type=LinearGEGLU, up_bias=False):
        super().__init__()
        self.norm = RMSNorm(d_model)
        # TODO swap here
        self.up_proj = apply_wd(up_proj_type(d_model, d_ff, bias=up_bias))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0, up_proj_type=LinearGEGLU, ffn_up_bias=False):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout, up_proj_type=up_proj_type, up_bias=ffn_up_bias) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


# Token merging and splitting

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)


class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2), skip_type: Literal['learned_lerp', 'add', 'concat'] = 'learned_lerp'):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.skip_type = skip_type
        if skip_type == 'learned_lerp':
            self.fac = nn.Parameter(torch.ones(1) * 0.5)
        elif skip_type == 'concat':
            self.skip_proj = apply_wd(Linear(out_features * 2, out_features, bias=False))

    def forward(self, x, skip):
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        if self.skip_type == 'learned_lerp':
            return torch.lerp(skip, x, self.fac.to(x.dtype))
        if self.skip_type == 'concat':
            return self.skip_proj(torch.cat([x, skip], dim=-1))
        assert self.skip_type == 'add', f"Unexpected skip_type, '{self.skip_type}'"
        return x + skip


# Configuration

@dataclass
class GlobalAttentionSpec:
    d_head: int


@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class ShiftedWindowAttentionSpec:
    d_head: int
    window_size: int


@dataclass
class NoAttentionSpec:
    pass

@dataclass
class CrossAttentionSpec:
    d_head: int
    d_cross: int
    scale_qk: bool
    dropout: float

@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[GlobalAttentionSpec, NeighborhoodAttentionSpec, ShiftedWindowAttentionSpec, NoAttentionSpec]
    cross_attn: Optional[CrossAttentionSpec]
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float
    ffn_up_bias: bool


# Model class

class ImageTransformerDenoiserModelV2(nn.Module):
    def __init__(self, levels: Sequence[LevelSpec], mapping: MappingSpec, in_channels, out_channels, patch_size, num_classes=0, mapping_cond_dim=0, up_proj_act: Literal["GELU", "GEGLU"] = "GEGLU", pos_emb_type: Literal["ROPE", "additive"] = "ROPE", input_size: Optional[Union[int, Tuple[int, int]]] = None, ffn_up_bias=False, backbone_skip_type: Literal['learned_lerp', 'add', 'concat'] = 'learned_lerp'):
        super().__init__()
        self.num_classes = num_classes

        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.aug_emb = layers.FourierFeatures(9, mapping.width)
        self.aug_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.class_emb = nn.Embedding(num_classes, mapping.width) if num_classes else None
        self.mapping_cond_in_proj = Linear(mapping_cond_dim, mapping.width, bias=False) if mapping_cond_dim else None
        self.up_proj_act = up_proj_act
        try:
            up_proj_type = { "GELU": LinearGELU, "GEGLU": LinearGEGLU }[self.up_proj_act]
        except KeyError:
            raise ValueError(f"Unknown activation '{self.up_proj_act}'.")

        self.pos_emb_type = pos_emb_type
        assert self.pos_emb_type in ["ROPE", "additive"]
        if self.pos_emb_type == "additive":
            assert not input_size is None, "Input size has to be provided to compute the correct positional embedding for additive PE."
            if isinstance(input_size, int):
                input_size = (input_size, input_size)
            assert len(input_size) == 2 and input_size[0] == input_size[1], "Additive PE only supports square images right now." # TODO: remove this limitation inherited from DiT
            p_s = [patch_size, patch_size] if isinstance(patch_size, int) else patch_size
            num_patches = (input_size[0] // p_s[0]) * (input_size[1] // p_s[0])
            # Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
            self.pos_emb = nn.Parameter(torch.zeros(1, input_size[0] // p_s[0], input_size[1] // p_s[1], levels[0].width), requires_grad=False)
            pos_embed = get_2d_sincos_pos_embed(self.pos_emb.shape[-1], int(num_patches**0.5))
            self.pos_emb.data.copy_(torch.from_numpy(pos_embed).float().reshape(*self.pos_emb.shape))

        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout, up_proj_type=up_proj_type, ffn_up_bias=mapping.ffn_up_bias), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            cross_attn: Optional[CrossAttentionBlock] = None if spec.cross_attn is None else CrossAttentionBlock(
                d_model=spec.width,
                d_cross=spec.cross_attn.d_cross,
                d_head=spec.cross_attn.d_head,
                cond_features=mapping.width,
                scale_qk=spec.cross_attn.scale_qk,
                dropout=spec.cross_attn.dropout,
            )
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, cross_attn=cross_attn, dropout=spec.dropout, up_proj_type=up_proj_type, use_rope=(self.pos_emb_type == "ROPE"), ffn_up_bias=ffn_up_bias)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, cross_attn=cross_attn, dropout=spec.dropout, up_proj_type=up_proj_type, use_rope=(self.pos_emb_type == "ROPE"), ffn_up_bias=ffn_up_bias)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.window_size, i, cross_attn=cross_attn, dropout=spec.dropout, up_proj_type=up_proj_type, use_rope=(self.pos_emb_type == "ROPE"), ffn_up_bias=ffn_up_bias)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(spec.width, spec.d_ff, mapping.width, dropout=spec.dropout, up_proj_type=up_proj_type, ffn_up_bias=ffn_up_bias)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width, skip_type=backbone_skip_type) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None, crossattn_cond: Optional[FloatTensor] = None, crossattn_mask: Optional[BoolTensor] = None) -> FloatTensor:
        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        # TODO: pixel aspect ratio for nonsquare patches
        if self.pos_emb_type == 'ROPE':
            pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)
        elif self.pos_emb_type == 'additive':
            x = x + self.pos_emb.to(x.dtype)
            pos = None
        else:
            raise ValueError(f"Unknown pos emb type '{self.pos_emb_type}'")

        # Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")
        if mapping_cond is None and self.mapping_cond_in_proj is not None:
            raise ValueError("mapping_cond must be specified if mapping_cond_dim > 0")

        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0
        mapping_emb = self.mapping_cond_in_proj(mapping_cond) if self.mapping_cond_in_proj is not None else 0
        cond = self.mapping(time_emb + aug_emb + class_emb + mapping_emb)

        # Hourglass transformer
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond, crossattn_cond, crossattn_mask)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            if not pos is None:
                pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond, crossattn_cond, crossattn_mask)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond, crossattn_cond, crossattn_mask)

        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)

        return x
