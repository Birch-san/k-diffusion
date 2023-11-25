from typing import Dict, Callable, List
from torch import Tensor, zeros
from torch.nn import Module
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

def should_discard_optim_state_vit_v2(model: ImageTransformerDenoiserModelV2, orig_model_config: Dict, new_model_config: Dict) -> bool:
  if new_model_config['ffn_up_bias'] and not orig_model_config['ffn_up_bias']:
    return True
  if new_model_config['mapping_ffn_up_bias'] and not orig_model_config['mapping_ffn_up_bias']:
    return True
  return False