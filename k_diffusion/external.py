import math

import torch
from torch import nn, Tensor
from typing import Protocol, Generic, TypeVar, TYPE_CHECKING

from . import sampling, utils


class Model(Protocol):
    def __call__(self, *args, **kwargs) -> Tensor: ...

class DenoiserModel(Model):
    def __call__(self, noised_input: Tensor, t: Tensor, **kwargs) -> Tensor: ...

class CompVisModel(Model):
    alphas_cumprod: Tensor
    def apply_model(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor: ...

# the 'default' arg of TypeVar isn't valid at runtime, but amazingly seems to be utilised
# at compile-time by some type-checkers
# https://github.com/python/mypy/issues/4236#issuecomment-344660299
if TYPE_CHECKING:
    TModel = TypeVar('TModel', bound=Model, default=Model)
    TDenoiserModel = TypeVar('TDenoiserModel', bound=DenoiserModel, default=DenoiserModel)
    TCompVisModel = TypeVar('TCompVisModel', bound=CompVisModel, default=CompVisModel)
else:
    TModel = TypeVar('TModel', bound=Model)
    TDenoiserModel = TypeVar('TDenoiserModel', bound=DenoiserModel)
    TCompVisModel = TypeVar('TCompVisModel', bound=CompVisModel)

class BaseModelWrapper(nn.Module, Generic[TModel]):
    inner_model: TModel

    """The base wrapper class for the k-diffusion model wrapper idiom. Model
    wrappers should subclass this class and customize the behavior of the
    wrapped model by implementing or overriding methods."""
    def __init__(self, model: TModel):
        super().__init__()
        self.inner_model = model

    def __dir__(self):
        return list(set(super().__dir__() + dir(self.inner_model)))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner_model, name)

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


class VDenoiser(BaseModelWrapper[TDenoiserModel]):
    """A v-diffusion-pytorch model wrapper for k-diffusion."""

    def __init__(self, model: TDenoiserModel):
        super().__init__(model)
        self.sigma_data = 1.

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigma):
        return sigma.atan() / math.pi * 2

    def t_to_sigma(self, t):
        return (t * math.pi / 2).tan()

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class DiscreteSchedule(BaseModelWrapper[TModel]):
    """A mapping between continuous noise levels (sigmas) and a list of discrete noise
    levels."""

    def __init__(self, sigmas, quantize, model: TModel):
        super().__init__(model)
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('log_sigmas', sigmas.log())
        self.quantize = quantize

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self, n=None):
        if n is None:
            return sampling.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigma.shape)
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t):
        t = t.float()
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t-low_idx if t.device.type == 'mps' else t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()


class DiscreteEpsDDPMDenoiser(DiscreteSchedule[TModel]):
    """A wrapper for discrete schedule DDPM models that output eps (the predicted
    noise)."""

    def __init__(self, model: TModel, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize, model)
        self.sigma_data = 1.

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    def get_eps(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def loss(self, input, noise, sigma, **kwargs):
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        return (eps - noise).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        return input + eps * c_out


class OpenAIDenoiser(DiscreteEpsDDPMDenoiser[TModel]):
    """A wrapper for OpenAI diffusion models."""

    def __init__(self, model: TModel, diffusion, quantize=False, has_learned_sigmas=True, device='cpu'):
        alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)
        super().__init__(model, alphas_cumprod, quantize=quantize)
        self.has_learned_sigmas = has_learned_sigmas

    def get_eps(self, *args, **kwargs):
        model_output = self.inner_model(*args, **kwargs)
        if self.has_learned_sigmas:
            return model_output.chunk(2, dim=1)[0]
        return model_output


class CompVisDenoiser(DiscreteEpsDDPMDenoiser[TCompVisModel]):
    """A wrapper for CompVis diffusion models."""

    def __init__(self, model: TCompVisModel, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)


class DiscreteVDDPMDenoiser(DiscreteSchedule[TModel]):
    """A wrapper for discrete schedule DDPM models that output v."""

    def __init__(self, model: TModel, alphas_cumprod, quantize):
        super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize, model)
        self.sigma_data = 1.

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def get_v(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def loss(self, input, noise, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.get_v(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        return self.get_v(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip


class CompVisVDenoiser(DiscreteVDDPMDenoiser[TCompVisModel]):
    """A wrapper for CompVis diffusion models that output v."""

    def __init__(self, model: TCompVisModel, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_v(self, x, t, cond, **kwargs):
        return self.inner_model.apply_model(x, t, cond)
