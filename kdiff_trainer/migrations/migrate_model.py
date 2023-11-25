from typing import Dict
from torch.nn import Module
from torch.optim import Optimizer
from k_diffusion.models import ImageTransformerDenoiserModelV2
from .migrate_image_transformer_v2 import register_load_hooks_vit_v2, should_discard_optim_state_vit_v2

def register_load_hooks(model: Module, orig_model_config: Dict, new_model_config: Dict) -> None:
  if isinstance(model, ImageTransformerDenoiserModelV2):
    register_load_hooks_vit_v2(model, orig_model_config, new_model_config)

def should_discard_optim_state(model: Module, orig_model_config: Dict, new_model_config: Dict) -> bool:
  if isinstance(model, ImageTransformerDenoiserModelV2):
    return should_discard_optim_state_vit_v2(model, orig_model_config, new_model_config)
  return False