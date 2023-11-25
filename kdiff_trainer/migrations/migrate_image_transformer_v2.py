from typing import Dict, Callable, List
import torch
from torch import Tensor, zeros, zeros_like, full, tensor
from torch.optim import Optimizer, AdamW
from torch.nn import Module, Parameter
from functools import partial

from k_diffusion.models import ImageTransformerDenoiserModelV2
from k_diffusion.models.image_transformer_v2 import FeedForwardBlock, MappingFeedForwardBlock

ModuleFn = Callable[[Module], None]

def ffn_on_load_state_dict_pre(module: FeedForwardBlock, state_dict: Dict[str, Tensor], prefix: str, local_metadata: Dict, strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]) -> None:
  assert f'{prefix}up_proj.weight' in state_dict
  bias_key = f'{prefix}up_proj.bias'
  assert bias_key not in state_dict
  assert module.up_proj.bias is not None
  state_dict[bias_key] = zeros(module.up_proj.bias.shape)

def mapping_ffn_on_load_state_dict_pre(module: MappingFeedForwardBlock, state_dict: Dict[str, Tensor], prefix: str, local_metadata: Dict, strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]) -> None:
  assert f'{prefix}up_proj.weight' in state_dict
  bias_key = f'{prefix}up_proj.bias'
  assert bias_key not in state_dict
  assert module.up_proj.bias is not None
  state_dict[bias_key] = zeros(module.up_proj.bias.shape)

def _add_ffn_up_bias(module: Module) -> None:
  if isinstance(module, FeedForwardBlock):
    module._register_load_state_dict_pre_hook(ffn_on_load_state_dict_pre, with_module=True)

def _add_mapping_ffn_up_bias(module: Module) -> None:
  if isinstance(module, MappingFeedForwardBlock):
    module._register_load_state_dict_pre_hook(mapping_ffn_on_load_state_dict_pre, with_module=True)

def _discover_ffn_up_bias_params(params: List[Parameter], module: Module) -> None:
  if isinstance(module, FeedForwardBlock):
    params.append(module.up_proj.bias)

def _discover_mapping_ffn_up_bias_params(params: List[Parameter], module: Module) -> None:
  if isinstance(module, MappingFeedForwardBlock):
    params.append(module.up_proj.bias)

def _module_fn(fns: List[ModuleFn], module: Module) -> None:
  for fn in fns:
    fn(module)

def register_load_hooks_vit_v2(model: ImageTransformerDenoiserModelV2, orig_model_config: Dict, new_model_config: Dict) -> None:
  fns: List[ModuleFn] = []
  if new_model_config['ffn_up_bias'] and not orig_model_config['ffn_up_bias']:
    fns.append(_add_ffn_up_bias)
  if new_model_config['mapping_ffn_up_bias'] and not orig_model_config['mapping_ffn_up_bias']:
    fns.append(_add_mapping_ffn_up_bias)
  if fns:
    model.apply(partial(_module_fn, fns))

def forge_opt_state_vit_v2(model: Module, optim: Optimizer, step: int, new_opt_state_dict: Dict, orig_model_config: Dict, new_model_config: Dict) -> None:
  if new_model_config['ffn_up_bias'] and not orig_model_config['ffn_up_bias']:
    assert isinstance(optim, AdamW)
    opt_state_next_key: int = len(new_opt_state_dict['state'].keys())
    ffn_up_biases: List[Parameter] = []
    model.apply(partial(_discover_ffn_up_bias_params, ffn_up_biases))
    param_group_ix = 1
    group = optim.param_groups[param_group_ix]
    for bias in ffn_up_biases:
      new_opt_state_dict['state'][opt_state_next_key] = {
        # Deliberately host `step` on CPU if both capturable and fused are off.
        # This is because kernel launches are costly on CUDA.
        'step': full((), step, dtype=torch.float, device=bias.device)
                    if group["capturable"] or group["fused"]
                    else tensor(step, dtype=torch.float),
        'exp_avg': zeros_like(bias, memory_format=torch.preserve_format),
        'exp_avg_sq': zeros_like(bias, memory_format=torch.preserve_format),
      }
      # splice this param into the right position in the params list
      index_in_group: int = next((ix for ix, x in enumerate(group['params']) if x is bias))
      # TODO: looks like this isn't enough. maybe we need to rewrite the indices in every param group to be monotonically increasing?
      new_opt_state_dict['param_groups'][param_group_ix]['params'].insert(index_in_group, opt_state_next_key)
      opt_state_next_key += 1
  if new_model_config['mapping_ffn_up_bias'] and not orig_model_config['mapping_ffn_up_bias']:
    assert isinstance(optim, AdamW)
    opt_state_next_key: int = len(new_opt_state_dict['state'].keys())
    mapping_ffn_up_biases: List[Parameter] = []
    model.apply(partial(_discover_mapping_ffn_up_bias_params, mapping_ffn_up_biases))
    param_group_ix = 3
    group = optim.param_groups[param_group_ix]
    for bias in mapping_ffn_up_biases:
      new_opt_state_dict['state'][opt_state_next_key] = {
        # Deliberately host `step` on CPU if both capturable and fused are off.
        # This is because kernel launches are costly on CUDA.
        'step': full((), step, dtype=torch.float, device=bias.device)
                    if group["capturable"] or group["fused"]
                    else tensor(step, dtype=torch.float),
        'exp_avg': zeros_like(bias, memory_format=torch.preserve_format),
        'exp_avg_sq': zeros_like(bias, memory_format=torch.preserve_format),
      }
      index_in_group: int = next((ix for ix, x in enumerate(group['params']) if x is bias))
      new_opt_state_dict['param_groups'][param_group_ix]['params'].insert(opt_state_next_key, index_in_group)
      opt_state_next_key += 1