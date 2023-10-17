from torch import FloatTensor, BoolTensor
from typing import NamedTuple

class MaskedCond(NamedTuple):
  cond: FloatTensor
  mask: BoolTensor