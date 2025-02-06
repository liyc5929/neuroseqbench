"""
According to: 
    Xiang Hao \emph{et al.} Towards Ultra-Low-Power Neuromorphic Speech Enhancement with Spiking-FullSubNet, 2024.
    Sepp Hochreiter \emph{et al.}, Long Short-Term Memory, 1997.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class StackedLSTM(nn.Module):

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]], **kwargs) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = []
        output = input
        # enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state, **kwargs)
            output_states += [out_state]
            i += 1
        return output, output_states


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor], **kwargs) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, **kwargs)
            outputs += [out]
        return torch.stack(outputs), state


class GSUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state, threshold=0.5, surrogate_function=None):
        hx, cx = state

        gates_ih = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gates_hh = (torch.mm(hx, self.weight_hh.t()) + self.bias_hh)

        updategate_ih, cellgate_ih = gates_ih.chunk(2, 1)
        updategate_hh, cellgate_hh = gates_hh.chunk(2, 1)

        updategate = torch.sigmoid(updategate_ih + updategate_hh)

        cell_gate = cellgate_hh + cellgate_ih

        cy = cx * updategate + (1 - updategate) * cell_gate
        hy = surrogate_function(cy - threshold)
        #cy = cy - cy * hy

        return hy, (hy, cy)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx, cx = state

        gates = (
                torch.mm(input, self.weight_ih.t())
                + self.bias_ih
                + torch.mm(hx, (self.weight_hh).t())
                + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


def script_lstm(
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=False,
        GSU=False,
        spiking_neuron=None
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    assert not batch_first

    stack_type = StackedLSTM
    layer_type = LSTMLayer
    dirs = 1
    if GSU:
        cell_type = GSUCell

    else:
        cell_type = LSTMCell

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[cell_type, input_size, hidden_size],
        other_layer_args=[cell_type, hidden_size * dirs, hidden_size],
    )


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type, num_hidden_layers=1, spiking_neuron=None,):
        super(LSTMNet, self).__init__()

        self.rnn_type = rnn_type
        self.time_window = 2
        self.nlayers = num_hidden_layers
        if isinstance(hidden_size, int):
            self.hidden_size = [hidden_size] * num_hidden_layers
        else:
            assert len(hidden_size) == num_hidden_layers
            self.hidden_size = hidden_size

        if rnn_type == 'lstm':
            self.rnns = [nn.LSTM(input_size if l == 0 else self.hidden_size[l-1],
                                 self.hidden_size[l],
                                 num_layers=1, batch_first=False)
                         for l in range(num_hidden_layers)]
        elif rnn_type == 'gsu':
            self.spiking_neuron = spiking_neuron()
            self.rnns = [script_lstm(input_size if l == 0 else self.hidden_size[l-1],
                                     self.hidden_size[l],
                                     num_layers=1, batch_first=False, GSU=True)
                         for l in range(num_hidden_layers)]

        else:
            raise NotImplementedError
        self.rnns = nn.ModuleList(self.rnns)
        self.classifier = nn.Linear(self.hidden_size[-1], output_size)

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':

            return [(weight.new_zeros(1, batch_size,
                                      self.hidden_size[l]).to(device),
                     weight.new_zeros(1, batch_size,
                                      self.hidden_size[l]).to(device))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'gsu':
            return [[(weight.new_zeros(batch_size,
                                       self.hidden_size[l]).to(device),
                      weight.new_zeros(batch_size,
                                       self.hidden_size[l]).to(device))]
                    for l in range(self.nlayers)]

        else:
            raise NotImplementedError(
                f"Model '{self.rnn_type}' not implemented.")

    def forward(self, inputs, **kwargs):
        hiddens = inputs
        state = self.init_hidden(batch_size=hiddens.size(1), device=hiddens.device)

        for l, rnn in enumerate(self.rnns):
            if self.rnn_type == 'gsu':
                hiddens, final_states = rnn(hiddens, state[l], threshold=self.spiking_neuron.neuron_thresh,
                                            surrogate_function=self.spiking_neuron.surro_func)
            elif self.rnn_type == 'lstm':
                hiddens, final_states = rnn(hiddens, state[l])
            else:
                raise NotImplementedError

        return self.classifier(hiddens)


##


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
        embed.scale_grad_by_freq, embed.sparse,
    )
    return X


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


class LMLSTM(nn.Module):
    def __init__(self,
        rnn_type,
        nlayers,
        emb_dim,
        hidden_dim,
        vocab_size,
        dropout_words,
        dropout_embedding,
        dropout_forward,
        dropout,
        spiking_neuron=None,
    ):
        super(LMLSTM, self).__init__()

        # language model specifics
        self.nlayers = nlayers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # dropout initializations
        self.dropout_words = dropout_words
        self.dropout_embedding = dropout_embedding
        self.dropout_forward = dropout_forward
        self.dropout = dropout

        # input and output layers
        self.locked_dropout = LockedDropout()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.decoder = nn.Linear(emb_dim, vocab_size)

        self.init_weights(initrange=0.1)

        # Tie weights of embedding and decoder
        self.decoder.weight = self.embeddings.weight

        # RNN model definition
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnns = [nn.LSTM(emb_dim if l == 0 else hidden_dim,
                                     emb_dim if l == nlayers - 1 else hidden_dim,
                                     num_layers=1, batch_first=False)
                         for l in range(nlayers)]

        elif rnn_type == 'gsu':
            self.spiking_neuron = spiking_neuron()
            self.rnns = [script_lstm(emb_dim if l == 0 else hidden_dim,
                                     emb_dim if l == nlayers - 1 else hidden_dim,
                                     num_layers=1, batch_first=False, GSU=True)
                         for l in range(nlayers)
                         ]
        else:
            raise NotImplementedError(f"Model '{rnn_type}' not implemented.")

        self.rnns = nn.ModuleList(self.rnns)

    def init_weights(self, initrange=0.1):
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        self.decoder.bias.data.fill_(0)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return [(weight.new_zeros(1, batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                     weight.new_zeros(1, batch_size,
                                      self.emb_dim if l == self.nlayers - 1 else self.hidden_dim))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'gsu':
            return [[(weight.new_zeros(batch_size,
                                       self.emb_dim if l == self.nlayers - 1 else self.hidden_dim),
                      weight.new_zeros(batch_size,
                                       self.emb_dim if l == self.nlayers - 1 else self.hidden_dim))]
                    for l in range(self.nlayers)]
        else:
            raise NotImplementedError(
                f"Model '{self.rnn_type}' not implemented.")

    def forward(self, inputs, state):

        # embedding forward
        embedded = embedded_dropout(self.embeddings, inputs,
                                    dropout=self.dropout_words if self.training else 0)

        embedded = self.locked_dropout(
            embedded, dropout=self.dropout_embedding)

        # rnn forward
        new_states = []
        hiddens = embedded
        for l, rnn in enumerate(self.rnns):
            if self.rnn_type == 'gsu':
                hiddens, final_states = rnn(hiddens, state[l], threshold=self.spiking_neuron.neuron_thresh, surrogate_function=self.spiking_neuron.surro_func)
            elif self.rnn_type == 'lstm':
                hiddens, final_states = rnn(hiddens, state[l])
            else:
                raise NotImplementedError

            new_states.append(final_states)

            if l != self.nlayers - 1:
                hiddens = self.locked_dropout(
                    hiddens, dropout=self.dropout_forward)

        # decoder forward
        hiddens = self.locked_dropout(hiddens, self.dropout)

        decoded = self.decoder(hiddens)
        return decoded, new_states
