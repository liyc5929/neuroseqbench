"""
According to: Ashish Vaswani \emph{et al.}, Attention is All you Need, 2017.
"""

import math
import torch
import torch.nn as nn

from .module import MergeDimension, SplitDimension


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x = self.self_attn(x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )[0]
        return self.dropout1(x)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 nhead,
                 num_hidden_layers=1,
                 dropout=0,
                 use_pool=False):
        super(TransformerNet, self).__init__()
        self.use_pool = use_pool
        if self.use_pool:
            self.flatten = nn.Flatten()
            self.max_pool = nn.MaxPool2d(4, 4)
        self.encoder = nn.Linear(input_size, hidden_size)

        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
                                                dropout=dropout,
                                                )
        encoder_norm = nn.LayerNorm(hidden_size)
        custom_encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers, encoder_norm)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4, num_encoder_layers=num_hidden_layers, dropout=dropout, custom_encoder=custom_encoder).encoder

        self.linear = nn.Linear(hidden_size, output_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=0.)

    def forward(self, x, **kwargs):
        if self.use_pool:
            time_step = x.size(0)
            x = MergeDimension()(x)
            x = self.max_pool(x)
            x = self.flatten(x)
            x = SplitDimension(time_step)(x)
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # input should have dimension (N, C, L)
        return self.linear(x)


###


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(
        words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse,
    )
    return X


class LMTransformer(nn.Module):

    def __init__(self,
        nhead,
        nlayers,
        emb_dim,
        hidden_dim,
        vocab_size,
        dropout_embedding,
        dropout_forward,
        spiking_neuron=None,
        spiking=False
    ):
        super(LMTransformer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout_forward,)
        encoder_norm = nn.LayerNorm(emb_dim)
        custom_encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim, num_encoder_layers=nlayers, dropout=dropout_forward, custom_encoder=custom_encoder).encoder
        self.decoder = nn.Linear(emb_dim, vocab_size)

        self.emb_dropout = dropout_embedding
        self.init_weights()
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=dropout_embedding)
        self.src_mask = None

        self.decoder.weight = self.embeddings.weight

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz, sz)))

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, has_mask=True):
        if has_mask:
            device = input.device
            if self.src_mask is None or self.src_mask.size(0) != len(input):
                mask = self._generate_square_subsequent_mask(len(input)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        emb = embedded_dropout(self.embeddings, input, dropout=0 if self.training else 0)
        emb = self.pos_encoder(emb)
        y = self.transformer(emb, mask=self.src_mask)
        y = self.decoder(y)
        return y
