import torch
from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.nn.functional import affine_grid, grid_sample
from .resizey_feature_extractor import Resize

class InceptionV3Resize(Module, Resize):
  def forward(self, img: Tensor) -> FloatTensor:
    batch_size, channels, height, width = img.shape
    assert channels == 3
    theta = torch.eye(2, 3, device=img.device)
    theta[0,2] += 1/width - 1/299
    theta[1,2] += 1/height - 1/299
    _13 = torch.unsqueeze(theta.to(img.dtype), 0)
    theta0 = _13.repeat([batch_size, 1, 1])
    grid = affine_grid(theta0, [batch_size, channels, 299, 299], False)
    resized = grid_sample(img, grid, "bilinear", "border", False)
    return resized