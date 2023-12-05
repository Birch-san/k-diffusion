from torch import Tensor
import torch
import torch.distributed as dist

class AllGather(torch.autograd.Function):
  """all_gather with support for backprop"""
  @staticmethod
  def forward(ctx, in_tensor: Tensor) -> Tensor:
    if dist.get_world_size() == 1:
      return in_tensor
    in_tensor = torch.atleast_1d(in_tensor)
    in_tensor = in_tensor.contiguous()
    out_tensor: Tensor = torch.empty((in_tensor.size(0)*dist.get_world_size(), *in_tensor.shape[1:]), dtype=in_tensor.dtype, layout=in_tensor.layout, device=in_tensor.device)
    dist.all_gather_into_tensor(out_tensor, in_tensor)
    return out_tensor

  @staticmethod
  def backward(ctx, in_grad: Tensor) -> Tensor:
    if dist.get_world_size() == 1:
      return in_grad
    out_grad: Tensor = torch.empty((in_grad.size(0)//dist.get_world_size(), *in_grad.shape[1:]), dtype=in_grad.dtype, layout=in_grad.layout, device=in_grad.device)
    dist.reduce_scatter_tensor(out_grad, in_grad)
    return out_grad

all_gather = AllGather.apply