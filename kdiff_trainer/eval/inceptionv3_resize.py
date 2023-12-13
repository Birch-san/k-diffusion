import torch
from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.nn.functional import affine_grid, grid_sample
from typing import Protocol

class Resize(Protocol):
  def __call__(self, x: Tensor) -> Tensor: ...
  def forward(self, x: Tensor) -> Tensor: ...

class InceptionV3Resize(Module, Resize):
  def forward(self, img: Tensor) -> FloatTensor:
    batch_size, channels, height, width = img.shape
    assert channels == 3
    theta = torch.eye(2, 3, device=img.device)
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
    _13 = torch.unsqueeze(theta.to(img.dtype), 0)
    theta0 = _13.repeat([batch_size, 1, 1])
    grid = affine_grid(theta0, [batch_size, channels, 299, 299], False)
    resized = grid_sample(img, grid, "bilinear", "border", False)
    return resized