import torch
from torch import FloatTensor, BoolTensor, LongTensor, distributed as dist
from accelerate import Accelerator
from typing import Optional, List, Literal

from .masked_cond import MaskedCond
from .crossattn_extra_args import CrossAttnExtraArgs
from .crossattn_cfg_args import CrossAttnCFGArgs

def get_precomputed_cond_cfg_args(
    accelerator: Accelerator,
    masked_conds: MaskedCond,
    uncond_class_ix: int,
    use_allzeros_uncond: bool,
    n_per_proc: int,
    distribute: bool,
    include_uncond: bool,
    rng: Optional[torch.Generator] = None,
) -> CrossAttnCFGArgs:
    num_conds_excl_uncond = masked_conds.cond.shape[0] - 1
    if not include_uncond:
        assert uncond_class_ix == num_conds_excl_uncond, "Expected uncond to be situated at the final element of the pre-computed conditions. We rely on this when picking which class numbers to generate, to enable us to omit uncond from the possible choices. If you don't mind generating uncond images in demos, enable --demo-classcond-include-uncond."
    captions_shape: List[int] = [accelerator.num_processes, n_per_proc] if distribute else [n_per_proc]
    maybe_uncond: Literal[0, 1] = 1 if include_uncond else 0
    caption_ix: LongTensor = torch.randint(0, num_conds_excl_uncond + maybe_uncond, captions_shape, generator=rng).to(accelerator.device)
    if distribute:
        dist.broadcast(caption_ix, 0)
        my_captions: LongTensor = caption_ix[accelerator.process_index]
    else:
        my_captions: LongTensor = caption_ix
    cond: FloatTensor = masked_conds.cond.index_select(0, my_captions)
    cond_mask: BoolTensor = masked_conds.mask.index_select(0, my_captions)
    if use_allzeros_uncond:
        cond[my_captions == uncond_class_ix] = 0
        cond_mask[my_captions == uncond_class_ix] = 1
    sampling_extra_args = CrossAttnExtraArgs(
        crossattn_cond=cond,
        crossattn_mask=cond_mask,
    )
    return CrossAttnCFGArgs(
        sampling_extra_args=sampling_extra_args,
        caption_ix=caption_ix,
    )