import torch
from cleanfid.inception_torchscript import InceptionV3W
from torch import Tensor, FloatTensor
from torch.nn import Module, Sequential
from torch.nn.functional import affine_grid, grid_sample, interpolate
from typing import Tuple

class SFID(Module):
  do_resize: bool
  layers: Sequential
  def __init__(self, inception: InceptionV3W, do_resize=False) -> None:
    super().__init__()
    self.do_resize = do_resize
    layers: Module = inception.base.layers
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

  def _resize(
    self,
    img: Tensor,
    use_fp16: bool=False,
  ) -> Tensor:
    batch_size, channels, height, width = img.shape
    assert torch.eq(channels, 3)
    x: FloatTensor = img.to(torch.float16 if use_fp16 else torch.float32)
    theta = torch.eye(2, 3, device=x.device)
    _3 = torch.select(torch.select(theta, 0, 0), 0, 2)
    _4 = torch.select(torch.select(theta, 0, 0), 0, 0)
    _5 = torch.div(_4, width)
    _6 = torch.select(torch.select(theta, 0, 0), 0, 0)
    _3.add_(torch.sub(_5, torch.div(_6, 299)))
    _8 = torch.select(torch.select(theta, 0, 1), 0, 2)
    _9 = torch.select(torch.select(theta, 0, 1), 0, 1)
    _10 = torch.div(_9, height)
    _11 = torch.select(torch.select(theta, 0, 1), 0, 1)
    _8.add_(torch.sub(_10, torch.div(_11, 299)))
    _13 = torch.unsqueeze(theta.to(x.dtype), 0)
    theta0 = _13.repeat([batch_size, 1, 1])
    grid = affine_grid(theta0, [batch_size, channels, 299, 299], False)
    x0 = grid_sample(x, grid, "bilinear", "border", False)
    x1 = x0.sub_(128)
    x2 = x1.div_(128)
    return x2

  def forward(self, x: Tensor) -> FloatTensor:
    if self.do_resize:
      x = self._resize(x)
    else:
      assert (x.shape[2] == 299) and (x.shape[3] == 299)
      # apply normalization
      x1 = x - 128
      x2 = x1 / 128
    features: FloatTensor = self.layers.forward(x2)
    # we have a 17x17 feature map. taking the first 7 channels (7*17*17=2023)
    # gives us a comparable size to the 2048 pool_3 feature vector.
    features = features[:,:7,:,:].flatten(start_dim=1)
    return features.float()

class ResizeyFeatureExtractor(Module):#
  extractor: Module
  size: Tuple[int, int]
  def __init__(self, extractor: Module, size: Tuple[int, int] = (299, 299)) -> None:
    super().__init__()
    self.extractor = extractor
    self.size = size
  
  def forward(self, x: Tensor) -> FloatTensor:
    x = interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
    if x.shape[1] == 1:
      x = torch.cat([x] * 3, dim=1)
    x = (x * 127.5 + 127.5).clamp(0, 255)
    return self.extractor.forward(x)