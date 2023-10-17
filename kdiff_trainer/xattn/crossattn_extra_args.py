from typing import TypedDict
from torch import FloatTensor, BoolTensor

class CrossAttnExtraArgs(TypedDict):
  crossattn_cond: FloatTensor
  crossattn_mask: BoolTensor