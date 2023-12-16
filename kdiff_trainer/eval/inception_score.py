import torch
from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.nn.functional import linear

class InceptionSoftmax(Module):
  inception: Module
  use_fp16: bool
  output_bias: bool
  def __init__(self, inception_torchscript: Module, use_fp16=False, output_bias=True) -> None:
    super().__init__()
    self.inception = inception_torchscript
    self.use_fp16 = use_fp16
    self.output_bias = output_bias

  def forward(self, x: Tensor) -> FloatTensor:
    _, _, h, w = x.shape
    assert h == 299 and w == 299
    if self.use_fp16:
      x = x.to(torch.float16)
    # TODO: check whether this gives us pool3:0 tensors
    features: FloatTensor = self.inception.layers(x)
    features = features.flatten(start_dim=1).float()
    if self.output_bias:
      logits: FloatTensor = self.inception.model.base.output.forward(features)
    else:
      logits: FloatTensor = linear(features, self.inception.model.base.output.weight)
    probabilities: FloatTensor = logits.softmax(-1)
    return probabilities