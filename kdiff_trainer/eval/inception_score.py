import torch
from torch import Tensor, FloatTensor
from torch.nn import Module

class InceptionSoftmax(Module):
  inception: Module
  use_fp16: bool
  def __init__(self, inception_torchscript: Module, use_fp16=False) -> None:
    super().__init__()
    self.inception = inception_torchscript
    self.use_fp16 = use_fp16

  def forward(self, x: Tensor) -> FloatTensor:
    _, _, h, w = x.shape
    assert h == 299 and w == 299
    if self.use_fp16:
      x = x.to(torch.float16)
    # TODO: check whether this gives us pool3:0 tensors
    features: FloatTensor = self.inception(x).float()
    probabilities: FloatTensor = features.softmax(-1)
    return probabilities