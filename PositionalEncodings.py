import torch
import torch.nn as nn
import math


class PositionalEncodings(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        pos = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # use log for numerical stability
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * denominator)  # all even columns (sin)
        pe[:, 1::2] = torch.cos(pos * denominator)  # all odd columns (cos)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer(
            "pe", pe
        )  # add pe to the model but not to the parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)  # non trainable
        return self.dropout(x)
