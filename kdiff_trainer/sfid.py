import torch
from cleanfid.inception_torchscript import InceptionV3W
from torch import Tensor, FloatTensor
from torch.nn import Module, Sequential
from torch.nn.functional import affine_grid, grid_sample, interpolate#, linear, softmax
from typing import Tuple

class Contiguous(Module):
  def forward(self, x: Tensor) -> Tensor:
    return x.contiguous()

class TruncatedMixed6(Module):
  layers: Sequential
  def __init__(self, mixed6: Module) -> None:
    super().__init__()
    contig = Contiguous()
    self.layers = Sequential(
      mixed6.conv,
      contig,
      # for some reason whatever comes out of contig, has the wrong number of channels for tower
      mixed6.tower,
      contig,
      mixed6.tower_1,
      contig,
      mixed6.tower_2,
      contig,
      # but we skip the final operation, cat(1)
    )
  def forward(self, x: Tensor) -> FloatTensor:
    y: FloatTensor = self.layers.forward(x)
    return y

class SFID(Module):
  resize_inside: bool
  layers: Sequential
  def __init__(self, inception: InceptionV3W, resize_inside=False) -> None:
    super().__init__()
    self.resize_inside = resize_inside
    layers: Module = inception.base.layers
    mixed6 = TruncatedMixed6(layers.mixed_6)
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
      mixed6,
    )

  def _resizey_forward(
    self,
    img: Tensor,
    # return_features: bool=False,
    use_fp16: bool=False,
    # no_output_bias: bool=False
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
    x3: FloatTensor = self.layers.forward(x2)
    return x3.to(torch.float32)
    # _14 = torch.reshape(self.layers.forward(x2), [-1, 2048])
    # features = torch.to(_14, 6)
    # if return_features:
    #   _15 = features
    # else:
    #   if no_output_bias:
    #     output = self.output
    #     weight = output.weight
    #     logits0 = linear(features, weight, None, )
    #     logits = logits0
    #   else:
    #     output0 = self.output
    #     logits = (output0).forward(features, )
    #   _16 = softmax(logits, 1, 3, None, )
    #   _15 = _16
    # return _15

  def forward(self, x: Tensor) -> FloatTensor:
    # bs = x.shape[0]
    if self.resize_inside:
      features = self._resizey_forward(x, return_features=True)#.view((bs, 2048))
    else:
      # make sure it is resized already
      assert (x.shape[2] == 299) and (x.shape[3] == 299)
      # apply normalization
      x1 = x - 128
      x2 = x1 / 128
      features = self.layers.forward(x2)#.view((bs, 2048))
    features_7chan: FloatTensor = features[:,:7,:,:]
    return features_7chan

class ResizeyFeatureExtractor(Module):
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