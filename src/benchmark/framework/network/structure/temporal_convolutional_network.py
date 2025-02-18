"""
    According to: Shaojie Bai \emph{et al.}, An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, 2018.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from . import MergeDimension, SplitDimension


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, spiking_neuron=None):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.ReLU() if spiking_neuron is None else spiking_neuron()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.ReLU() if spiking_neuron is None else spiking_neuron()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.spiking_neuron = spiking_neuron

        self.relu = nn.ReLU() if spiking_neuron is None else nn.Identity()
        self.init_weights()
        if spiking_neuron is not None and self.downsample is not None:
            self.downsample_neuron = spiking_neuron()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x): 
        if self.spiking_neuron is None: # input shape [B, N, T]
            out = self.conv1(x)
            out = self.chomp1(out)
            out = self.act1(out)
            out = self.dropout1(out)
            out = self.conv2(out)
            out = self.chomp2(out)
            out = self.act2(out)
            out = self.dropout2(out)
            res = x if self.downsample is None else self.downsample(x)
        else: # input shape [T_in, B, N, T]
            inner_time_step = x.size(0) # fetch `T_in`
            out = MergeDimension()(x)
            out = self.conv1(out)
            out = self.chomp1(out)
            out = SplitDimension(inner_time_step)(out)
            out = self.act1(out)

            out = MergeDimension()(out)
            out = self.dropout1(out)
            out = self.conv2(out)
            out = self.chomp2(out)
            out = SplitDimension(inner_time_step)(out)
            out = self.act2(out)

            out = MergeDimension()(out)
            out = self.dropout2(out)
            out = SplitDimension(inner_time_step)(out)

            if self.downsample:
                x = MergeDimension()(x)
                res = self.downsample(x)
                res = SplitDimension(inner_time_step)(res)
                res = self.downsample_neuron(res)
            else:
                res = x
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, spiking_neuron=None):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, 
                padding=(kernel_size-1)*dilation_size, dropout=dropout, spiking_neuron=spiking_neuron,
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """
        input size: [B, N, T] (non-spiking) | [T_in, B, N, T] (spiking)
    """
    def __init__(self, 
        input_size, 
        output_size, 
        num_channels, 
        kernel_size, 
        dropout = 0.0, 
        spiking_neuron = None, 
        output_last_step = False,
    ):
        super(TCN, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, spiking_neuron=spiking_neuron)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        self.spiking = spiking_neuron is not None
        self.output_last_step = output_last_step
        self.time_window = 1

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, **kwargs):
        x = x.permute(1, 2, 0).contiguous()  # [T, B, N] -> [B, N, T]
        if self.spiking:
            x = x.unsqueeze(0).repeat(self.time_window, 1, 1, 1) # [B, N, T] -> [T_in, B, N, T]
            y1 = self.tcn(x).mean(0)
            if self.output_last_step:
                output = self.linear(y1[:, :, -1]).unsqueeze(0)
            else:
                y1 = y1.permute(2, 0, 1).contiguous()
                output = self.linear(y1)
        else:
            y1 = self.tcn(x)
            if self.output_last_step:
                output = self.linear(y1[:, :, -1]).unsqueeze(0)
            else:
                y1 = y1.permute(2, 0, 1).contiguous()
                output = self.linear(y1)
        return output


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
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
        embed.scale_grad_by_freq, embed.sparse
    )
    return X


class LMTCN(nn.Module):
    def __init__(self,
        emb_dim,
        vocab_size,
        dropout_words,
        dropout_embedding,
        dropout_forward,
        dropout,
        num_channels,
        kernel_size,
        spiking_neuron = None,
    ):
        super(LMTCN, self).__init__()

        # Language model specifics
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        # Set dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_forward = dropout_forward
        self.dropout = dropout

        # Set input and output layers
        self.locked_dropout = LockedDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)
        self.return_state = True

        self.init_weights(initrange=0.1)

        # Tie weights of embedding and decoder
        self.decoder.weight = self.embeddings.weight

        self.tcn = TemporalConvNet(emb_dim, num_channels, kernel_size, dropout=dropout_forward, spiking_neuron=spiking_neuron)
        assert num_channels[-1] == emb_dim

        self.spiking = spiking_neuron is not None
        self.time_window = 1

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return [
            (weight.new_zeros(1, batch_size, self.emb_dim), weight.new_zeros(1, batch_size, self.emb_dim)) 
            for l in range(1)
        ]

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs, state):
        # Embedding forward
        embedded = embedded_dropout(self.embeddings, inputs, dropout=self.dropout_words if self.training else 0)
        embedded = self.locked_dropout(embedded, dropout=self.dropout_embedding)

        # RNN forward
        hiddens = embedded
        # TCN here
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        hiddens = hiddens.permute(1, 2, 0).contiguous()  # [T, B, N] - >[B, N, T]
        if self.spiking:
            hiddens = hiddens.unsqueeze(0).repeat(self.time_window, 1, 1, 1)  # [B, N, T] -> [T_in, B, N, T]
            hiddens = self.tcn(hiddens).mean(0)
        else:
            hiddens = self.tcn(hiddens)
        hiddens = hiddens.permute(2, 0, 1).contiguous()
        # Decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)
        decoded = self.decoder(hiddens)
        return decoded, state
