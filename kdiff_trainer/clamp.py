from torch import FloatTensor
from torch.nn import Module
from dataclasses import dataclass

# convenient for consumers such as Sequential, which only use forward()
@dataclass()
class Clamp_(Module):
    min: float
    max: float
    def forward(self, x: FloatTensor) -> FloatTensor:
        x.clamp_(self.min, self.max)
        return x