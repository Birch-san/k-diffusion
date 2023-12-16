import torch
from torch import Tensor, FloatTensor
from torch.nn import Module, Linear


_imagenet_classes=1000

class InceptionLogits(Module):
  inception_layers: Module
  use_fp16: bool
  output: Linear
  def __init__(self, inception_torchscript: Module, use_fp16=False, output_bias=True) -> None:
    super().__init__()
    self.inception_layers = inception_torchscript.layers
    self.use_fp16 = use_fp16
    self.output = Linear(
      in_features=inception_torchscript.output.weight.shape[-1],
      out_features=_imagenet_classes,
      bias=output_bias,
      device=inception_torchscript.output.weight.device,
      dtype=inception_torchscript.output.weight.dtype,
    )
    # the CleanFID inception model outputs 1008 classes. only 1000 are required for imagenet;
    # the rest are for backwards-compatibility with an older, Google-internal system. we remove those.
    # https://github.com/tensorflow/tensorflow/issues/4128
    self.output.weight.data.copy_(inception_torchscript.output.weight[:_imagenet_classes,:])
    if output_bias:
      self.output.bias.data.copy_(inception_torchscript.output.bias[:_imagenet_classes])

  def forward(self, x: Tensor) -> FloatTensor:
    _, _, h, w = x.shape
    assert h == 299 and w == 299
    if self.use_fp16:
      x = x.to(torch.float16)
    # TODO: check whether this gives us pool3:0 tensors
    features: FloatTensor = self.inception_layers(x)
    features = features.flatten(start_dim=1).float()
    logits: FloatTensor = self.output.forward(features)
    return logits