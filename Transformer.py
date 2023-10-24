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
    def __init__(
        self,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout_rate)] * 2
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, mask)
        )
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
    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout_rate)] * 3
        )

    def forward(self, x, encoder_output, enc_mask, dec_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, dec_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.self_attention(x, encoder_output, encoder_output, enc_mask),
        )
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
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        enc_embed: InputEmbeddings,
        dec_embed: InputEmbeddings,
        enc_pe: PositionalEncodings,
        dec_pe: PositionalEncodings,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_embed = enc_embed
        self.dec_embed = dec_embed
        self.enc_pe = enc_pe
        self.dec_pe = dec_pe
        self.projection_layer = projection_layer

    def encode(self, src, enc_mask):
        src = self.enc_embed(src)  # input embedding
        src = self.enc_pe(src)  # add positional encoding
        return self.encoder(src, enc_mask)

    def decode(self, target, encoder_output, enc_mask, dec_mask):
        target = self.dec_embed(target)
        target = self.dec_pe(target)
        return self.decoder(target, encoder_output, enc_mask, dec_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
):
    """
    Builds and initializes all components of the transformer.

        Parameters:
            src_vocab_size: source vocabulary size
            tgt_vocab_size: target vocabulary size
            src_seq_len: source sequence length
            tgt_seq_len: target sequence length
            d_model: embedding dimension
            N: number of encoder and decoder blocks
            h: number of heads
            dropout: dropout rate
            d_ff: feed forward dimension

        Returns:
            transformer: transformer model
    """

    # create embeddings
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)

    # create positional encodings
    src_pe = PositionalEncodings(d_model, src_seq_len, dropout)
    tgt_pe = PositionalEncodings(d_model, tgt_seq_len, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)  # MHA
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # create encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)  # M-MHA
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)  # MHA
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention, decoder_cross_attention, feed_forward, dropout
        )
        decoder_blocks.append(decoder_block)

    # create decoder
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create transformer
    transformer = Transformer(
        encoder, decoder, src_embedding, tgt_embedding, src_pe, tgt_pe, projection_layer
    )

    # init with Xavier
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
