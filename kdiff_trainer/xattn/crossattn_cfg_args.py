from torch import LongTensor
from typing import NamedTuple

from .crossattn_extra_args import CrossAttnExtraArgs

class CrossAttnCFGArgs(NamedTuple):
  sampling_extra_args: CrossAttnExtraArgs
  caption_ix: LongTensor