import torch
import k_diffusion as K
from typing import Dict, Any, List
from dataclasses import dataclass
from k_diffusion.models import ImageTransformerDenoiserModelV2
# from k_diffusion.layers import Denoiser
from kdiff_trainer.flops.hook_flops import instrument_ffn_flops
from k_diffusion.models.flops import FlopCounter
from torch.cuda.amp.autocast_mode import autocast
from math import log

@dataclass
class ScaleSpec:
    max_res_area: int
    min_res_area: int

def compute_levels_total(
    max_res_area: int,
    min_res_area = 16**2,
    ingress_patch_area = 4**2,
    merge_patch_area = 2**2,
) -> int:
    return int(log(max_res_area/ingress_patch_area, merge_patch_area) - log(min_res_area, merge_patch_area) + 1)

def compute_levels_global(
    max_res_area_global: int,
    min_res_area = 16**2,
    merge_patch_area = 2**2,
) -> int:
    return int(log(max_res_area_global, merge_patch_area) - log(min_res_area, merge_patch_area) + 1)

def compute_levels_local(
    max_res_area: int,
    max_res_area_global: int,
    ingress_patch_area = 4**2,
    merge_patch_area = 2**2,
) -> int:
    return int(log(max_res_area/ingress_patch_area, merge_patch_area) - log(max_res_area_global, merge_patch_area))

def compute_ffn_flops(
    max_res_area: int,
    model_dim: int,
    min_res_area = 16**2,
    depth_outer = 2,
    depth_inner = 2,
    batch_size = 1,
    ffn_multiple = 3,
    ingress_patch_area = 4**2,
    merge_patch_area = 2**2,
) -> int:
    levels_total: int = compute_levels_total(
        max_res_area = max_res_area,
        min_res_area = min_res_area,
        ingress_patch_area = ingress_patch_area,
        merge_patch_area = merge_patch_area,
    )
    # instead of range(1, levels_total+1) (as per the paper draft),
    # setting this to range(1, levels_total) seems to give us a consistent
    # ratio of 4/3 between theory and practice, which seems like a step in the right direction
    level_token_quotients: List[int] = [merge_patch_area ** l for l in range(1, levels_total)]
    level_token_quotient_sum: int = sum(level_token_quotients)
    return 4 * ffn_multiple * batch_size * model_dim**2 * min_res_area * (2 * depth_outer * level_token_quotient_sum + depth_inner)

def main():
    device=torch.device('cuda')

    max_res_side = 512
    ingress_patch_side = 4
    ingress_patch_area = ingress_patch_side**2
    merge_patch_area = 2**2
    max_res_area = max_res_side**2
    min_res_area = 16**2
    max_res_area_global = min_res_area
    depth_outer = 2
    depth_inner = 2
    model_base_dim = 128
    ffn_multiple = 3
    head_dim = 64
    kernel_size = 7
    kernel_area = kernel_size**2
    batch_size = 1

    local_levels: int = compute_levels_local(
        max_res_area = max_res_area,
        max_res_area_global = max_res_area_global,
        ingress_patch_area = ingress_patch_area,
        merge_patch_area = merge_patch_area,
    )
    global_levels: int = compute_levels_global(
        max_res_area_global = max_res_area_global,
        min_res_area = min_res_area,
        merge_patch_area = merge_patch_area,
    )
    total_levels: int = compute_levels_total(
        max_res_area = max_res_area,
        min_res_area = min_res_area,
        ingress_patch_area = ingress_patch_area,
        merge_patch_area = merge_patch_area,
    )
    assert total_levels == global_levels + local_levels
    ffn_flops_predicted: int = compute_ffn_flops(
        max_res_area = max_res_area,
        model_dim = model_base_dim,
        min_res_area = min_res_area,
        depth_outer = depth_outer,
        depth_inner = depth_inner,
        batch_size = batch_size,
        ffn_multiple = ffn_multiple,
        ingress_patch_area = ingress_patch_area,
        merge_patch_area = merge_patch_area,
    )
    print(f'[theorised] ffn GFLOPs: {ffn_flops_predicted / 1_000_000_000:,.3f}')

    depths: List[int] = [
        *[depth_outer]*(total_levels-1),
        depth_inner,
    ]

    # our FLOP formulae assume all levels have same width
    widths: List[int] = [model_base_dim]*total_levels
    d_ffs: List[int] = [width * ffn_multiple for width in widths]

    self_attns: List[Dict[str, Any]] = [
        *[{"type": "neighborhood", "d_head": head_dim, "kernel_size": kernel_size}]*local_levels,
        *[{"type": "global", "d_head": head_dim}]*global_levels,
    ]

    model_config = {
        "type": "image_transformer_v2",
        "input_channels": 3,
        "input_size": [max_res_side, max_res_side],
        "patch_size": [ingress_patch_side, ingress_patch_side],
        "depths": depths,
        "widths": widths,
        "d_ffs": d_ffs,
        "self_attns": self_attns,
    }
    config_nominal = {
        "model": model_config,
        "dataset": {
            "num_classes": 1000
        },
    }
    config: Dict[str, Any] = K.config.load_config(config_nominal)

    inner_model: ImageTransformerDenoiserModelV2 = K.config.make_model(config).eval().to(device)
    # model: Denoiser = K.config.make_denoiser_wrapper(config)(inner_model)
    ffn_counter = FlopCounter()
    instrument_ffn_flops(ffn_counter, inner_model)
    with torch.inference_mode(), autocast(dtype=torch.bfloat16):
        x = torch.zeros([batch_size, model_config['input_channels'], *model_config['input_size']], device=device)
        sigma = torch.ones([batch_size], device=device)
        extra_args = {}
        if getattr(inner_model, "num_classes", 0):
            extra_args['class_cond'] = torch.zeros([batch_size], dtype=torch.long, device=device)
        inner_model.forward(x, sigma, **extra_args)

    print(f' [measured] ffn GFLOPs: {ffn_counter.flops / 1_000_000_000:,.3f}')
    pass

if __name__ == '__main__':
    main()