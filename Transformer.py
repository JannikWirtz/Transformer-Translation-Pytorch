import torch
import torch.nn as nn
import math
import MultiHeadAttention
import FeedForward
import ResidualConnection
import LayerNorm
import InputEmbeddings
import PositionalEncodings


class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout_rate: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate)] * 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
        just repeats the EncoderBlock n-times
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout_rate: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate)] * 3)

    def forward(self, x, encoder_output, enc_mask, dec_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, dec_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention(x, encoder_output, encoder_output, enc_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """
        just repeats the DecoderBlock n-times
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, enc_mask, dec_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, enc_mask, dec_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    





