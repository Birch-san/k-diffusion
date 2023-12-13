from torch import FloatTensor, Tensor
from torch.nn import Module, Sequential
from typing import Protocol

class Resize(Protocol):
  def __call__(self, x: Tensor) -> Tensor: ...
  def forward(self, x: Tensor) -> Tensor: ...

class ResizeyFeatureExtractor(Module):
  layers: Sequential
  def __init__(self, extractor: Module, resize: Resize) -> None:
    super().__init__()
    self.layers = Sequential(
      resize,
      extractor,
    )
  
  def forward(self, x: Tensor) -> FloatTensor:
    return self.layers(x)