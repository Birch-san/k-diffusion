from contextlib import contextmanager
import math
import threading
from torch.nn import Linear, Conv1d, Conv2d
from torch import FloatTensor
from typing import Tuple
from .flops_to_macs import conv_mac_count


state = threading.local()
state.flop_counter = None


@contextmanager
def flop_counter(enable=True):
    try:
        old_flop_counter = state.flop_counter
        state.flop_counter = FlopCounter() if enable else None
        yield state.flop_counter
    finally:
        state.flop_counter = old_flop_counter


class FlopCounter:
    def __init__(self):
        self.ops = []

    def op(self, op, *args, **kwargs):
        self.ops.append((op, args, kwargs))

    @property
    def flops(self):
        flops = 0
        for op, args, kwargs in self.ops:
            flops += op(*args, **kwargs)
        return flops


def op(op, *args, **kwargs):
    if getattr(state, "flop_counter", None):
        state.flop_counter.op(op, *args, **kwargs)


def op_linear(x, weight):
    return math.prod(x) * weight[0]


def op_attention(q, k, v):
    *b, s_q, d_q = q
    *b, s_k, d_k = k
    *b, s_v, d_v = v
    return math.prod(b) * s_q * s_k * (d_q + d_v)


def op_natten(q, k, v, kernel_size):
    *q_rest, d_q = q
    *_, d_v = v
    return math.prod(q_rest) * (d_q + d_v) * kernel_size**2

def hook_linear_flops(module: Linear, args: Tuple[FloatTensor, ...], _):
    x, *_ = args
    op(op_linear, x.shape, module.weight.shape)

def hook_conv1d_flops(module: Conv1d, args: Tuple[FloatTensor, ...], out: FloatTensor):
    x, *_ = args
    op(conv_mac_count, x.shape, module.weight.shape, out.shape)

def hook_conv2d_flops(module: Conv2d, args: Tuple[FloatTensor, ...], out: FloatTensor):
    x, *_ = args
    op(conv_mac_count, x.shape, module.weight.shape, out.shape)