import numbers
import warnings
from collections import namedtuple
from typing import List, Tuple
from functools import partial
import math
import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from src.benchmark.framework.network.trainer import TriangleSurroGrad


def script_lstm(
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=False,
        GRU=False,
        spiking=False,
        spiking_neuron=None
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    assert not batch_first

    stack_type = StackedLSTM
    layer_type = LSTMLayer
    dirs = 1
    if GRU:
        if spiking:
            cell_type = SpikingGRUCell
        else:
            cell_type = GRUCell
    else:
        if spiking:
            cell_type = SpikingLSTMCell3

        else:
            cell_type = LSTMCell

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[cell_type, input_size, hidden_size],
        other_layer_args=[cell_type, hidden_size * dirs, hidden_size],
    )


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    # @jit.script_method
    def forward(self, input, state, **kwargs):
        # type: # (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        # outputs = torch.jit.annotate(List[Tensor], [])
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, **kwargs)
            outputs += [out]
        return torch.stack(outputs), state


class StackedLSTM(nn.Module):
    # __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    # @jit.script_method
    def forward(self, input, states, **kwargs):
        # type: # (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        # output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output_states = []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state, **kwargs)
            output_states += [out_state]
            i += 1
        return output, output_states


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
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


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx = state

        gates_ih = (
                torch.mm(input, self.weight_ih.t())
                + self.bias_ih
        )
        gates_hh = (torch.mm(hx, self.weight_hh.t())
                    + self.bias_hh)

        resetgate_ih, updategate_ih, cellgate_ih = gates_ih.chunk(3, 1)
        resetgate_hh, updategate_hh, cellgate_hh = gates_hh.chunk(3, 1)

        resetgate = torch.sigmoid(resetgate_ih + resetgate_hh)
        updategate = torch.sigmoid(updategate_ih + updategate_hh)

        cell_gate = torch.tanh(cellgate_hh * resetgate + cellgate_ih)

        hy = hx * updategate + (1 - updategate) * cell_gate

        return hy, hy


class SpikingGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(2 * hidden_size))
        self.bias_hh = Parameter(torch.randn(2 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state, threshold=0.5, surrogate_function=TriangleSurroGrad.apply):
        hx, cx = state

        gates_ih = (
                torch.mm(input, self.weight_ih.t())
                + self.bias_ih
        )
        gates_hh = (torch.mm(hx, self.weight_hh.t())
                    + self.bias_hh)

        updategate_ih, cellgate_ih = gates_ih.chunk(2, 1)
        updategate_hh, cellgate_hh = gates_hh.chunk(2, 1)

        updategate = torch.sigmoid(updategate_ih + updategate_hh)

        cell_gate = cellgate_hh + cellgate_ih

        cy = cx * updategate + (1 - updategate) * cell_gate
        hy = surrogate_function(cy - threshold)
        #cy = cy - cy * hy

        return hy, (hy, cy)


class SpikingLSTMCell3(nn.Module):
    """
    `Long Short-Term Memory Spiking Networks and Their Applications <https://arxiv.org/abs/2007.04779>`
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state, threshold=0.5, surrogate_function=TriangleSurroGrad.apply):
        hx, cx = state

        gates = (
                torch.mm(input, self.weight_ih.t())
                + self.bias_ih
                + torch.mm(hx, (self.weight_hh).t())
                + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = surrogate_function(ingate)
        forgetgate = surrogate_function(forgetgate)
        cellgate = surrogate_function(cellgate)
        outgate = surrogate_function(outgate)

        cy = (forgetgate * cx) + ingate * cellgate
        with torch.no_grad():
            torch.clamp_max_(cy, 1.)
        hy = outgate * cy

        return hy, (hy, cy)
class SpikingLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, spiking_neuron=None):
        super().__init__()
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        # self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        # self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.lstmcell = torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)
        # self.reset_parameters()
        self.spiking_neuron = spiking_neuron()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
    #     for weight in self.parameters():
    #         torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx, cx = state
        # print(f"hx: {hx.size()}, cx: {cx.size()}")
        time_window = input.size(0)
        input = input.flatten(0, 1)
        hx = hx.flatten(0, 1)


        # gates = (
        #         torch.mm(input, self.weight_ih.t())
        #         + self.bias_ih
        #         + torch.mm(hx, (self.weight_hh).t())
        #         + self.bias_hh
        # )
        # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        #
        # ingate = torch.sigmoid(ingate)
        # forgetgate = torch.sigmoid(forgetgate)
        # cellgate = torch.tanh(cellgate)
        # outgate = torch.sigmoid(outgate)
        #
        # cy = (forgetgate * cx) + (ingate * cellgate)
        # # print(f"cy: {cy.size()}")
        # hy = outgate * torch.tanh(cy)
        hy, cy = self.lstmcell(input, (hx, cx))
        hy = hy.reshape(time_window, hy.shape[0] // time_window, *hy.shape[1:])
        hy = self.spiking_neuron(hy)
        # print(f"hy: {hy.sum()}")
        # print(f"hy: {hy.size()}")
        return hy, (hy, cy)


class SpikingLSTMCell2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state, threshold=0.5, surrogate_function=TriangleSurroGrad.apply):
        hx, cx = state
        gates = (
                torch.mm(input, self.weight_ih.t())
                + self.bias_ih
                + torch.mm(hx, self.weight_hh.t())
                + self.bias_hh
        )
        ingate, forgetgate, cellgate = gates.chunk(3, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)

        # cellgate = cellgate
        # outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = surrogate_function(cy - threshold)
        cy = cy - cy * hy

        return hy, (hy, cy)


class SpikingLSTMCell2(nn.Module):
    def __init__(self, input_size, hidden_size, shared_weights=False, bn=False):
        super(SpikingLSTMCell2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.shared_weights = shared_weights
        self.use_bn = bn
        if self.shared_weights:
            self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        else:
            self.weight_ih = Parameter(torch.empty(2 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(2 * hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()

        # self.scale_factor = Parameter(torch.ones(hidden_size))
        if self.use_bn:
            self.batchnorm = nn.BatchNorm1d(hidden_size)
            # self.bn2 = nn.BatchNorm1d(hidden_size)
            # self.bn3 = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    # def reset_parameters(self, param, hidden_size):
    #     stdv = 1.0 / math.sqrt(hidden_size)
    #     torch.nn.init.uniform_(param, -stdv, stdv)

    # @jit.script_method
    def forward(self, input, state):
        # type: # (Tensor,
        #
        #
        # Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        if self.shared_weights:
            weight_ih = self.weight_ih.repeat((2, 1))
            weight_hh = self.weight_hh.repeat((2, 1))
        else:
            weight_ih = self.weight_ih
            weight_hh = self.weight_hh
        gates = (
                torch.mm(input, weight_ih.t())
                + self.bias_ih
                + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )
        forgetgate, cellgate = gates.chunk(2, 1)

        # ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        # ingate = Triangle.apply(self.bn2(ingate))
        # forgetgate = Triangle.apply(self.bn3(forgetgate))
        # cellgate = torch.tanh(cellgate)
        # outgate = torch.sigmoid(outgate)

        cy = forgetgate * cx + (1 - forgetgate) * cellgate
        if self.use_bn:
            cy = self.batchnorm(cy)
        hy = Triangle.apply(cy)
        # hy = GradedSpike.apply(self.bn(cy), self.scale) * self.scale
        # hy = torch.tanh(cy)
        return hy, (hy, cy)


LSTMState = namedtuple("LSTMState", ["hx", "cx"])


def script_stacked_lnlstm(seq_len, batch, input_size, hidden_size, num_layers):
    print(
        f"seq_len: {seq_len}, batch: {batch}, input_size: {input_size}, hidden_size: {hidden_size}, num_layers: {num_layers}"
    )
    inp = torch.randn(seq_len, batch, input_size)
    states = [
        LSTMState(torch.zeros(batch, hidden_size), torch.zeros(batch, hidden_size))
        for _ in range(num_layers)
    ]
    rnn = script_lstm(input_size, hidden_size, num_layers)

    # just a smoke test
    out, out_state = rnn(inp, states)
