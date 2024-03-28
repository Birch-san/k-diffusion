import torch
from torch.utils.flop_counter import (
  get_shape,
  transpose_shape,
  mm_flop,
  bmm_flop,
  conv_flop_count,
)
from typing import List

aten = torch.ops.aten

def mm_mac(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
  """Count MACs for matmul."""
  # torch counts multiplies *and* adds
  flos: int = mm_flop(a_shape, b_shape, *args, **kwargs)
  # OpenAI guided diffusion paper reported multiply-accumulates, not conventional FLOs
  macs: int = flos // 2
  return macs

def addmm_mac(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
  """Count MACs for addmm."""
  return mm_mac(a_shape, b_shape)

def bmm_mac(a_shape, b_shape, out_shape=None, **kwargs) -> int:
  """Count MACs for the bmm operation."""
  flos: int = bmm_flop(a_shape, b_shape, **kwargs)
  macs: int = flos // 2
  return macs

def baddbmm_mac(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
  """Count MACs for the baddbmm operation."""
  # Inputs should be a list of length 3.
  # Inputs contains the shapes of three tensors.
  return bmm_mac(a_shape, b_shape)

def conv_mac_count(
  x_shape: List[int],
  w_shape: List[int],
  out_shape: List[int],
  transposed: bool = False,
) -> int:
  flos: int = conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)
  macs: int = flos // 2
  return macs

def conv_mac(x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> int:
  """Count flops for convolution."""
  return conv_mac_count(x_shape, w_shape, out_shape, transposed=transposed)

def conv_backward_mac(
  grad_out_shape,
  x_shape,
  w_shape,
  _bias,
  _stride,
  _padding,
  _dilation,
  transposed,
  _output_padding,
  _groups,
  output_mask,
  out_shape,
) -> int:
  mac_count = 0

  if output_mask[0]:
    grad_input_shape = get_shape(out_shape[0])
    mac_count += conv_mac_count(grad_out_shape, w_shape, grad_input_shape, not transposed)
  if output_mask[1]:
    grad_weight_shape = get_shape(out_shape[1])
    mac_count += conv_mac_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, transposed)

  return mac_count

def sdpa_mac_count(query_shape, key_shape, value_shape):
  """
  Count MACs for self-attention.

  NB: We can assume that value_shape == key_shape
  """
  b, h, s_q, d_q = query_shape
  _b2, _h2, s_k, _d2 = key_shape
  _b3, _h3, _s3, d_v = value_shape
  assert b == _b2 == _b3 and h == _h2 == _h3 and d_q == _d2 and s_k == _s3 and d_q == _d2
  total_macs = 0
  # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
  total_macs += bmm_mac((b * h, s_q, d_q), (b * h, d_q, s_k))
  # scores: [b, h, s_q, s_k] @ v: [b, h, s_k, d_v] -> out: [b, h, s_q, d_v]
  total_macs += bmm_mac((b * h, s_q, s_k), (b * h, s_k, d_v))
  return total_macs

def sdpa_mac(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
  """Count MACs for self-attention."""
  # NB: We aren't accounting for causal attention here
  return sdpa_mac_count(query_shape, key_shape, value_shape)

def sdpa_backward_mac_count(grad_out_shape, query_shape, key_shape, value_shape):
  b, h, s_q, d_q = query_shape
  _b2, _h2, s_k, _d2 = key_shape
  _b3, _h3, _s3, d_v = value_shape
  _b4, _h4, _s4, _d4 = grad_out_shape
  assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and d_q == _d2
  assert d_v == _d4 and s_k == _s3 and s_q == _s4
  total_macs = 0
  # Step 1: We recompute the scores matrix.
  # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
  total_macs += bmm_mac((b * h, s_q, d_q), (b * h, d_q, s_k))

  # Step 2: We propagate the gradients through the score @ v operation.
  # gradOut: [b, h, s_q, d_v] @ v: [b, h, d_v, s_k] -> gradScores: [b, h, s_q, s_k]
  total_macs += bmm_mac((b * h, s_q, d_v), (b * h, d_v, s_k))
  # scores: [b, h, s_k, s_q] @ gradOut: [b, h, s_q, d_v] -> gradV: [b, h, s_k, d_v]
  total_macs += bmm_mac((b * h, s_k, s_q), (b * h, s_q, d_v))

  # Step 3: We propagate th gradients through the k @ v operation
  # gradScores: [b, h, s_q, s_k] @ k: [b, h, s_k, d_q] -> gradQ: [b, h, s_q, d_q]
  total_macs += bmm_mac((b * h, s_q, s_k), (b * h, s_k, d_q))
  # q: [b, h, d_q, s_q] @ gradScores: [b, h, s_q, s_k] -> gradK: [b, h, d_q, s_k]
  total_macs += bmm_mac((b * h, d_q, s_q), (b * h, s_q, s_k))
  return total_macs

def sdpa_backward_mac(grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count MACs for self-attention backward."""
    return sdpa_backward_mac_count(grad_out_shape, query_shape, key_shape, value_shape)

custom_mapping = {
  aten.mm: mm_mac,
  aten.addmm: addmm_mac,
  aten.bmm: bmm_mac,
  aten.baddbmm: baddbmm_mac,
  aten.convolution: conv_mac,
  aten._convolution: conv_mac,
  aten.convolution_backward: conv_backward_mac,
  aten._scaled_dot_product_efficient_attention: sdpa_mac,
  aten._scaled_dot_product_flash_attention: sdpa_mac,
  aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_mac,
  aten._scaled_dot_product_flash_attention_backward: sdpa_backward_mac,
}