import torch
from torch import Tensor, FloatTensor
from torch.nn import Module, Sequential

class SFID(Module):
  layers: Sequential
  use_fp16: bool
  def __init__(self, inception_torchscript: Module, use_fp16=False) -> None:
    super().__init__()
    self.use_fp16 = use_fp16
    layers: Module = inception_torchscript.layers
    self.layers = Sequential(
      layers.conv,
      layers.conv_1,
      layers.conv_2,
      layers.pool0,
      layers.conv_3,
      layers.conv_4,
      layers.pool1,
      layers.mixed,
      layers.mixed_1,
      layers.mixed_2,
      layers.mixed_3,
      layers.mixed_4,
      layers.mixed_5,
      layers.mixed_6.conv,
    )

  def forward(self, x: Tensor) -> FloatTensor:
    _, _, h, w = x.shape
    assert h == 299 and w == 299
    if self.use_fp16:
      x = x.to(torch.float16)
    features: FloatTensor = self.layers.forward(x)
    # we have a 17x17 feature map. taking the first 7 channels (7*17*17=2023)
    # gives us a comparable size to the 2048 pool_3 feature vector.
    features = features[:,:7,:,:].flatten(start_dim=1)
    return features.float()