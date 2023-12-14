import torch
from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.nn.functional import interpolate
from typing import Tuple
from .resizey_feature_extractor import Resize

class BicubicResize(Module, Resize):
  size: Tuple[int, int]
  def __init__(self, size: Tuple[int, int] = (299, 299)) -> None:
    super().__init__()
    self.size = size
  
  def forward(self, x: Tensor) -> FloatTensor:
    x = interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
    if x.shape[1] == 1:
      x = torch.cat([x] * 3, dim=1)
    return x