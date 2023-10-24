import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert self.d_model % self.h == 0, "d_model must be divisible by h"
        self.d_k = self.d_model // self.h
        self.w_q = nn.Linear(self.d_model, self.d_model)  # query weight matrix
        self.w_k = nn.Linear(self.d_model, self.d_model)  # key weight matrix
        self.w_v = nn.Linear(self.d_model, self.d_model)  # value weight matrix
        # output weight matrix
        self.w_o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        d_k: int,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ) -> torch.Tensor:
        d_k = q.shape[-1]
        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        attention_scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ v), attention_scores

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # q, k, v: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # split heads along embedding dimension
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], -1, self.h, self.d_k).transpose(1, 2)

        # scaled dot product attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # Combine heads
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # output weight matrix
        return self.w_o(x)
