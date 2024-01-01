from functools import lru_cache, reduce
import math

from dctorch import functional as df
from einops import rearrange, repeat
import torch
from torch import nn, FloatTensor
from torch.nn import functional as F
from typing import Literal, Callable, Union, Dict, Any
from functools import partial

from . import sampling, utils


# Helper functions


def dct(x):
    if x.ndim == 3:
        return df.dct(x)
    if x.ndim == 4:
        return df.dct2(x)
    if x.ndim == 5:
        return df.dct3(x)
    raise ValueError(f'Unsupported dimensionality {x.ndim}')


@lru_cache
def freq_weight_1d(n, scales=0, dtype=None, device=None):
    ramp = torch.linspace(0.5 / n, 0.5, n, dtype=dtype, device=device)
    weights = -torch.log2(ramp)
    if scales >= 1:
        weights = torch.clamp_max(weights, scales)
    return weights


@lru_cache
def freq_weight_nd(shape, scales=0, dtype=None, device=None):
    indexers = [[slice(None) if i == j else None for j in range(len(shape))] for i in range(len(shape))]
    weights = [freq_weight_1d(n, scales, dtype, device)[ix] for n, ix in zip(shape, indexers)]
    return reduce(torch.minimum, weights)


# Karras et al. preconditioned denoiser

def weighting_karras_x0(
    sigma_data: float,
    sigma: FloatTensor,
) -> FloatTensor:
    """
    Karras loss weighting, λ(σ). Equal to 1/c_out**2
    If you have an x0 loss weighting, you can divide it by this to derive its corresponding EDM loss weighting.
    Conversely, if you have an EDM loss weighting: you can multiply it by this to derive its corresponding x0 loss weighting.
    """
    return (sigma**2 + sigma_data**2)/(sigma*sigma_data)**2

def weighting_snr_x0(sigma_data: float, sigma: FloatTensor, snr_adjust_for_sigma_data: bool) -> FloatTensor:
    """
    SNR (signal-to-noise ratio) loss weighting.
    snr_adjust_for_sigma_data
        Whether to consider the variance you expect to see in the signal.
        Most papers overlook this (and assume sigma_data=1, which can be true if you've standardized your data, e.g. latents).
        But RGB photography tends to be sigma_data=0.5.
        False:
        Defines SNR as 1/sigma**2.
        True:
        Defines SNR as sigma_data**2/sigma**2.
    """
    signal_std: float = sigma_data**2 if snr_adjust_for_sigma_data else 1.
    snr: FloatTensor = signal_std/sigma**2
    return snr

def weighting_min_snr_x0(
    sigma_data: float,
    sigma: FloatTensor,
    gamma: float,
    snr_adjust_for_sigma_data: bool,
    gamma_adjust_for_sigma_data: bool,
) -> FloatTensor:
    """
    Implements min-SNR weighting.
    https://arxiv.org/abs/2303.09556

    To reproduce Min-SNR paper, choose:
        gamma=5
        snr_adjust_for_sigma_data=False
        gamma_adjust_for_sigma_data=False
    
    But we recommend against this; the Min-SNR paper did not take into account the variance in signal (sigma_data).

    Instead we recommend:
        gamma=sigma_data**-2
        snr_adjust_for_sigma_data=True
        gamma_adjust_for_sigma_data=True
    
    Because their recommended value of gamma, 5, is suspiciously close to sigma_data**-2 given the common sigma_data value of 0.5.

    snr_adjust_for_sigma_data
        see weighting_snr_x0 docs
    
    gamma_adjust_for_sigma_data
        This is a bit of a guess. Setting both this and snr_adjust_for_sigma_data to True, has nice properties such as:
        - when gamma=sigma_data**-2, as we suspect might be optimal, it becomes min(snr, 1), which looks significant.
        - is just a constant scale factor sigma_data**2 away from the official Min-SNR formulation.
        this could explain how they were able to get good results on sigma_data!=1 datasets without considering sigma_data.
        i.e. the theory is that Adam grad scaling eliminated any constant scale factors, making their formulation equivalent to this variance-aware version.

        False:
        min(snr, gamma)
        True:
        min(snr, gamma*sigma_data**2)
    """
    snr: FloatTensor = weighting_snr_x0(
        sigma_data=sigma_data,
        sigma=sigma,
        snr_adjust_for_sigma_data=snr_adjust_for_sigma_data,
    )
    gamma_scale_factor: float = sigma_data**2 if gamma_adjust_for_sigma_data else 1.
    return snr.clamp_max(gamma * gamma_scale_factor)

def weighting_soft_min_snr_x0(
    sigma_data: float,
    sigma: FloatTensor,
    gamma: float,
    snr_adjust_for_sigma_data: bool,
    gamma_adjust_for_sigma_data: bool,
) -> FloatTensor:
    """
    It's like min-SNR, except instead of an abrupt clamp, it smoothly transitions from curved to flat, symmetrical around sigma_data.

    To fit a soft version of what's in the Min-SNR paper, choose:
        gamma=sigma_data**-2
        snr_adjust_for_sigma_data=False
        gamma_adjust_for_sigma_data=False
    
    But we recommend against this; fitting to the Min-SNR paper's curve verbatim, means not taking into account the variance in signal (sigma_data).

    Instead we recommend:
        gamma=sigma_data**-2
        snr_adjust_for_sigma_data=True
        gamma_adjust_for_sigma_data=True

    snr_adjust_for_sigma_data
        see weighting_snr_x0 docs
        implementated by trying to fit a soft version of our variance-aware Min-SNR with the same setting enabled
    gamma_adjust_for_sigma_data
        see weighting_min_snr_x0 docs
        implementated by trying to fit a soft version of our variance-aware Min-SNR with the same setting enabled
    """
    if snr_adjust_for_sigma_data:
        recip_gamma_scale_factor: float = sigma_data**2 if gamma_adjust_for_sigma_data else 1.
        recip_gamma_scaled: float = recip_gamma_scale_factor/gamma
        return (sigma_data**4 * sigma**2) / ((sigma**2 + sigma_data**2) * (sigma**2 + recip_gamma_scaled))
    # the original formulation was not parameterized on gamma, it was parameterized on sigma_data
    # it provided a soft version of what was described in the Min-SNR paper, and matched as sigma_data was adjusted,
    # provided Min-SNR was given gamma == sigma_data**-2.
    # however it is worth noting that the Min-SNR paper was not variance-aware, so matching its response to changes in sigma_data
    # is probably not a recipe for success.
    assert gamma == sigma_data**-2
    return 1/(sigma**2 + 1/gamma)

def loss_weighting_x0(
    loss_weighting: Literal['karras', 'snr', 'min-snr', 'soft-min-snr'],
    loss_weighting_params: Dict[str, Any],
    sigma_data: float,
    sigma: FloatTensor,
) -> FloatTensor:
    """
    Computes the x0-space loss weighting c_weight, used like so:
    c_weight = loss_weighting_x0(…)
    loss = c_weight * mse(denoiseds, reals)
    """
    if loss_weighting == 'karras':
        return weighting_karras_x0(sigma_data, sigma)
    if loss_weighting == 'snr':
        return weighting_snr_x0(
            sigma_data=sigma_data,
            sigma=sigma,
            snr_adjust_for_sigma_data=loss_weighting_params['snr_adjust_for_sigma_data'],
        )
    if loss_weighting == 'min-snr':
        return weighting_min_snr_x0(
            sigma_data=sigma_data,
            sigma=sigma,
            gamma=loss_weighting_params['gamma'],
            snr_adjust_for_sigma_data=loss_weighting_params['snr_adjust_for_sigma_data'],
            gamma_adjust_for_sigma_data=loss_weighting_params['gamma_adjust_for_sigma_data'],
        )
    if loss_weighting == 'soft-min-snr':
        return weighting_soft_min_snr_x0(
            sigma_data=sigma_data,
            sigma=sigma,
            gamma=loss_weighting_params['gamma'],
            snr_adjust_for_sigma_data=loss_weighting_params['snr_adjust_for_sigma_data'],
            gamma_adjust_for_sigma_data=loss_weighting_params['gamma_adjust_for_sigma_data'],
        )
    raise ValueError(f"x0 loss weighting not implemented for '{loss_weighting}'")

class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1., weighting: Union[Literal['karras', 'soft-min-snr', 'snr', 'min-snr'], Callable[[], FloatTensor]]='karras', scales=1, weighting_params: Dict[str, Any] = {}):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.scales = scales
        if callable(weighting):
            self.weighting = weighting
        elif weighting == 'karras':
            self.weighting = torch.ones_like
        elif weighting == 'soft-min-snr':
            if 'gamma' not in weighting_params:
                raise ValueError(f'soft-min-snr weighting lacked a gamma property. usually config.py would provide a default. gamma=sigma_data**-2 is recommended.')
            if 'snr_adjust_for_sigma_data' not in weighting_params:
                raise ValueError(f'soft-min-snr weighting lacked a snr_adjust_for_sigma_data property. usually config.py would provide a default. you might consider snr_adjust_for_sigma_data=True to gain variance-awareness, or snr_adjust_for_sigma_data=False to produce a soft version of official Min-SNR weighting')
            if 'gamma_adjust_for_sigma_data' not in weighting_params:
                raise ValueError(f'soft-min-snr weighting lacked a gamma_adjust_for_sigma_data property. usually config.py would provide a default. you might consider gamma_adjust_for_sigma_data=True for our guess at how to achieve variance-awareness, or gamma_adjust_for_sigma_data=False to produce a soft version of official Min-SNR weighting')
            if weighting_params['gamma_adjust_for_sigma_data']:
                assert weighting_params['snr_adjust_for_sigma_data'], 'gamma_adjust_for_sigma_data can only be enabled in tandem with snr_adjust_for_sigma_data, as it is a further variance-adjustment'
            self.weighting = partial(
                self._weighting_soft_min_snr,
                gamma=weighting_params['gamma'],
                snr_adjust_for_sigma_data=weighting_params['snr_adjust_for_sigma_data'],
                gamma_adjust_for_sigma_data=weighting_params['gamma_adjust_for_sigma_data'],
            )
        elif weighting == 'min-snr':
            if 'gamma' not in weighting_params:
                raise ValueError(f'min-snr weighting lacked a gamma property. usually config.py would provide a default. you might consider gamma=sigma_data**-2, or gamma=5 to follow the Min-SNR paper')
            if 'snr_adjust_for_sigma_data' not in weighting_params:
                raise ValueError(f'min-snr weighting lacked a snr_adjust_for_sigma_data property. usually config.py would provide a default. you might consider snr_adjust_for_sigma_data=True to gain variance-awareness, or snr_adjust_for_sigma_data=False to follow the Min-SNR paper')
            if 'gamma_adjust_for_sigma_data' not in weighting_params:
                raise ValueError(f'min-snr weighting lacked a gamma_adjust_for_sigma_data property. usually config.py would provide a default. you might consider gamma_adjust_for_sigma_data=True for our guess at how to achieve variance-awareness, or gamma_adjust_for_sigma_data=False to follow the Min-SNR paper')
            self.weighting = partial(
                self._weighting_min_snr,
                gamma=weighting_params['gamma'],
                snr_adjust_for_sigma_data=weighting_params['snr_adjust_for_sigma_data'],
                gamma_adjust_for_sigma_data=weighting_params['gamma_adjust_for_sigma_data'],
            )
        elif weighting == 'snr':
            if 'snr_adjust_for_sigma_data' not in weighting_params:
                raise ValueError(f'snr weighting lacked a snr_adjust_for_sigma_data property. usually config.py would provide a default. you might consider snr_adjust_for_sigma_data=True to gain variance-awareness, or snr_adjust_for_sigma_data=False to follow the Min-SNR paper')
            self.weighting = partial(
                self._weighting_snr,
                snr_adjust_for_sigma_data=weighting_params['snr_adjust_for_sigma_data'],
            )
        else:
            raise ValueError(f'Unknown weighting type {weighting}')

    def _weighting_soft_min_snr(self, sigma: FloatTensor, gamma: float, snr_adjust_for_sigma_data: bool, gamma_adjust_for_sigma_data: bool) -> FloatTensor:
        soft_min_snr_x0: FloatTensor = weighting_soft_min_snr_x0(
            sigma_data=self.sigma_data,
            sigma=sigma,
            gamma=gamma,
            snr_adjust_for_sigma_data=snr_adjust_for_sigma_data,
            gamma_adjust_for_sigma_data=gamma_adjust_for_sigma_data,
        )
        return soft_min_snr_x0 / weighting_karras_x0(self.sigma_data, sigma)

    def _weighting_min_snr(self, sigma: FloatTensor, gamma: float, snr_adjust_for_sigma_data: bool, gamma_adjust_for_sigma_data: bool) -> FloatTensor:
        min_snr_x0: FloatTensor = weighting_min_snr_x0(
            sigma_data=self.sigma_data,
            sigma=sigma,
            gamma=gamma,
            snr_adjust_for_sigma_data=snr_adjust_for_sigma_data,
            gamma_adjust_for_sigma_data=gamma_adjust_for_sigma_data,
        )
        return min_snr_x0 / weighting_karras_x0(self.sigma_data, sigma)

    def _weighting_snr(self, sigma: FloatTensor, snr_adjust_for_sigma_data: bool):
        snr_x0: FloatTensor = weighting_snr_x0(
            sigma_data=self.sigma_data,
            sigma=sigma,
            snr_adjust_for_sigma_data=snr_adjust_for_sigma_data,
        )
        return snr_x0 / weighting_karras_x0(self.sigma_data, sigma)

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        if self.scales == 1:
            return ((model_output - target) ** 2).flatten(1).mean(1) * c_weight
        sq_error = dct(model_output - target) ** 2
        f_weight = freq_weight_nd(sq_error.shape[2:], self.scales, dtype=sq_error.dtype, device=sq_error.device)
        return (sq_error * f_weight).flatten(1).mean(1) * c_weight

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, sigma, **kwargs) * c_out + input * c_skip


class DenoiserWithVariance(Denoiser):
    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output, logvar = self.inner_model(noised_input * c_in, sigma, return_variance=True, **kwargs)
        logvar = utils.append_dims(logvar, model_output.ndim)
        target = (input - c_skip * noised_input) / c_out
        losses = ((model_output - target) ** 2 / logvar.exp() + logvar) / 2
        return losses.flatten(1).mean(1)


class SimpleLossDenoiser(Denoiser):
    """L_simple with the Karras et al. preconditioner."""

    def loss(self, input, noise, sigma, **kwargs):
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        denoised = self(noised_input, sigma, **kwargs)
        eps = sampling.to_d(noised_input, sigma, denoised)
        return (eps - noise).pow(2).flatten(1).mean(1)


# Residual blocks

class ResidualBlock(nn.Module):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


# Noise level (and other) conditioning

class ConditionedModule(nn.Module):
    pass


class UnconditionedModule(ConditionedModule):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, cond=None):
        return self.module(input)


class ConditionedSequential(nn.Sequential, ConditionedModule):
    def forward(self, input, cond):
        for module in self:
            if isinstance(module, ConditionedModule):
                input = module(input, cond)
            else:
                input = module(input)
        return input


class ConditionedResidualBlock(ConditionedModule):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = ConditionedSequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input, cond):
        skip = self.skip(input, cond) if isinstance(self.skip, ConditionedModule) else self.skip(input)
        return self.main(input, cond) + skip


class AdaGN(ConditionedModule):
    def __init__(self, feats_in, c_out, num_groups, eps=1e-5, cond_key='cond'):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.cond_key = cond_key
        self.mapper = nn.Linear(feats_in, c_out * 2)
        nn.init.zeros_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)

    def forward(self, input, cond):
        weight, bias = self.mapper(cond[self.cond_key]).chunk(2, dim=-1)
        input = F.group_norm(input, self.num_groups, eps=self.eps)
        return torch.addcmul(utils.append_dims(bias, input.ndim), input, utils.append_dims(weight, input.ndim) + 1)


# Attention


class SelfAttention2d(ConditionedModule):
    def __init__(self, c_in, n_head, norm, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        self.norm_in = norm(c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input, cond):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm_in(input, cond))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class CrossAttention2d(ConditionedModule):
    def __init__(self, c_dec, c_enc, n_head, norm_dec, dropout_rate=0.,
                 cond_key='cross', cond_key_padding='cross_padding'):
        super().__init__()
        assert c_dec % n_head == 0
        self.cond_key = cond_key
        self.cond_key_padding = cond_key_padding
        self.norm_enc = nn.LayerNorm(c_enc)
        self.norm_dec = norm_dec(c_dec)
        self.n_head = n_head
        self.q_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.kv_proj = nn.Linear(c_enc, c_dec * 2)
        self.out_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input, cond):
        n, c, h, w = input.shape
        q = self.q_proj(self.norm_dec(input, cond))
        q = q.view([n, self.n_head, c // self.n_head, h * w]).transpose(2, 3)
        kv = self.kv_proj(self.norm_enc(cond[self.cond_key]))
        kv = kv.view([n, -1, self.n_head * 2, c // self.n_head]).transpose(1, 2)
        k, v = kv.chunk(2, dim=1)
        attn_mask = (cond[self.cond_key_padding][:, None, None, :]) * -10000
        y = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


# Downsampling/upsampling

_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic': 
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
        0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3': 
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
        -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
        0.44638532400131226, 0.13550527393817902, -0.066637322306633,
        -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}
_kernels['bilinear'] = _kernels['linear']
_kernels['bicubic'] = _kernels['cubic']


class Downsample2d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv2d(x, weight, stride=2)


class Upsample2d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv_transpose2d(x, weight, stride=2, padding=self.pad * 2 + 1)


# Embeddings

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


# U-Nets

class UNet(ConditionedModule):
    def __init__(self, d_blocks, u_blocks, skip_stages=0):
        super().__init__()
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(u_blocks)
        self.skip_stages = skip_stages

    def forward(self, input, cond):
        skips = []
        for block in self.d_blocks[self.skip_stages:]:
            input = block(input, cond)
            skips.append(input)
        for i, (block, skip) in enumerate(zip(self.u_blocks, reversed(skips))):
            input = block(input, cond, skip if i > 0 else None)
        return input
