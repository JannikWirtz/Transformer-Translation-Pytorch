import torch
import torch.nn as nn
from LayerNorm import LayerNorm


class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        # using the more modern pre-norm here; instead of original post-norm
        return x + self.dropout(sublayer(self.norm(x)))
