import math
from collections import namedtuple
import logging
import torch
import torch.nn as nn
from torch.nn import Parameter


MemoryState = namedtuple("MemoryState", ["hx", "cx"])


def efficient_spiking_neuron(
    input_size,
    hidden_size,
    num_layers,
    shared_weights=False,
    bn=False,
    batch_first=False,
    neuron_name='GSN',
    decay_factor=0.5,
    threshold=1.0
):
    """
    Instantiate efficient spiking networks where each spiking neuron uses the gating mechanism to control the decay of membrane potential.
    :param input_size:
    :param hidden_size:
    :param num_layers:
    :param shared_weights: whether weights of the gate are shared with the ones of the cell or not.
    :param bn: whether batchnorm is used or not.
    :param batch_first: Not used.
    :return:
    """
    #     # The following are not implemented.
    assert not batch_first
    assert neuron_name in ['GSN', 'LIF', 'PLIF', 'ALIF']
    # assert shared_weights
    # assert bn
    if neuron_name == 'GSN':
        return StackedGSU(
            num_layers,
            GSULayer,
            first_layer_args=[GSUCell, input_size, hidden_size, shared_weights, bn],
            other_layer_args=[GSUCell, hidden_size, hidden_size, shared_weights, bn],
        )
    elif neuron_name == 'LIF':
        return StackedGSU(
            num_layers,
            GSULayer,
            first_layer_args=[LIFCell, input_size, hidden_size, bn, decay_factor, threshold],
            other_layer_args=[LIFCell, hidden_size, hidden_size, bn, decay_factor, threshold],
        )
    elif neuron_name == 'PLIF':
        return StackedGSU(
            num_layers,
            GSULayer,
            first_layer_args=[PLIFCell, input_size, hidden_size, bn, threshold],
            other_layer_args=[PLIFCell, hidden_size, hidden_size, bn, threshold],
        )
    elif neuron_name == 'ALIF':
        return StackedGSU(
            num_layers,
            GSULayer,
            first_layer_args=[ALIFCell, input_size, hidden_size, bn, threshold],
            other_layer_args=[ALIFCell, hidden_size, hidden_size, bn, threshold],
        )


class StackedGSU(nn.Module):
    # __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedGSU, self).__init__()
        self.layers = init_stacked_gsu(num_layers, layer, first_layer_args, other_layer_args)

    def forward(self, input, states):
        output_states = []
        all_layer_forgetgates = []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        all_layer_output = []
        for rnn_layer in self.layers:
            state = states[i]
            # output, out_state, forgetgates = rnn_layer(output, state)
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            all_layer_output += [output]
            # all_layer_forgetgates += [forgetgates]
            i += 1
        return output, output_states, all_layer_output#, all_layer_forgetgates


def init_stacked_gsu(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class GSULayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(GSULayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []
        # mem_states = []
        forgetgates = []
        # cellgates = []
        for i in range(len(inputs)):
            # out, state, forgetgate, cellgate = self.cell(inputs[i], state)
            out, state = self.cell(inputs[i], state)
            outputs += [out]
            # mem_states += [state[1]]
            # forgetgates += [forgetgate]
            # cellgates += [cellgate]
        return torch.stack(outputs), state#, torch.stack(forgetgates)


class Triangle(torch.autograd.Function):
    """Spike firing activation function"""

    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class GSUCell(nn.Module):
    def __init__(self, input_size, hidden_size, shared_weights=False, bn=False):
        super(GSUCell, self).__init__()
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
        # if self.use_bn:
        #     self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
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
        forgetgate = torch.sigmoid(forgetgate)
        cy = forgetgate * cx + (1 - forgetgate) * cellgate
        # if self.use_bn:
        #     cy = self.batchnorm(cy)
        hy = Triangle.apply(cy)  # replace the Tanh activation function with step function to ensure binary outputs.

        return hy, (hy, cy), forgetgate, cellgate



class LIFCell(nn.Module):
    def __init__(self, input_size, hidden_size, bn=False, decay_factor=0.5, threshold=1.0):
        super(LIFCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bn = bn
        self.decay_factor = decay_factor
        self.threshold = threshold
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        # self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.zeros(hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()
        # self.scale_factor = Parameter(torch.ones(hidden_size))
        # if self.use_bn:
        #     self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx, cx = state

        weight_ih = self.weight_ih
        # weight_hh = self.weight_hh
        input_current = (
            torch.mm(input, weight_ih.t())
            + self.bias_ih
            # + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )
        cy = 0.8 * cx + input_current
        # cy = input_current
        # if self.use_bn:
        #     hy = Triangle.apply(self.batchnorm(cy - self.threshold))
        #     # cy = self.batchnorm(cy)
        # else:
        hy = Triangle.apply(cy - self.threshold)  # replace the Tanh activation function with step function to ensure binary outputs.
        cy = cy - hy.detach() * self.threshold
        return hy, (hy, cy)

class PLIFCell(nn.Module):
    def __init__(self, input_size, hidden_size, bn=False, threshold=1.0):
        super(PLIFCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bn = bn
        self.threshold = threshold
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.zeros(hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()
        self.decay_factor = Parameter(torch.zeros(hidden_size))
        # self.decay_factor = Parameter(torch.zeros(1))
        # self.decay_factor = Parameter(torch.ones(hidden_size)*(-1.3))
        # logging.info(f"decay_factor: {self.decay_factor}")
        # self.scale_factor = Parameter(torch.ones(hidden_size))
        # if self.use_bn:
        #     self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx, cx = state

        weight_ih = self.weight_ih
        weight_hh = self.weight_hh
        input_current = (
            torch.mm(input, weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )
        decay_factor = self.decay_factor.sigmoid()
        cy = decay_factor * cx + (1 - decay_factor) * input_current
        # if self.use_bn:
        #     hy = Triangle.apply(self.batchnorm(cy - self.threshold))
        #     # cy = self.batchnorm(cy)
        # else:
        hy = Triangle.apply(cy - self.threshold)  # replace the Tanh activation function with step function to ensure binary outputs.
        cy = cy - hy * self.threshold
        return hy, (hy, cy)


class ALIFCell(nn.Module):
    def __init__(self, input_size, hidden_size, bn=False, threshold=1.0):
        super(ALIFCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bn = bn
        self.threshold = threshold
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.zeros(hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()
        self.tau_m = Parameter(torch.zeros(hidden_size))
        # self.tau_m = Parameter(torch.ones(hidden_size) * (-1.3))
        # self.tau_adp = Parameter(torch.zeros(hidden_size))
        self.tau_adp = Parameter(torch.ones(hidden_size) * (-1.3))
        # tauM = 20
        # tauAdp_inital = 100
        # tauM_inital_std = 5
        # tauAdp_inital_std = 5
        # nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
        # nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
        # logging.info(f"tau_m: {self.tau_m}")
        # logging.info(f"tau_adp: {self.tau_adp}")
        # logging.info(f"decay_factor: {self.decay_factor}")
        # self.scale_factor = Parameter(torch.ones(hidden_size))
        # if self.use_bn:
        #     self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        if len(state) == 2:
            hx, cx = state
            cb = torch.ones_like(hx, device=hx.device)
        elif len(state) == 3:
            hx, cx, cb = state
        weight_ih = self.weight_ih
        weight_hh = self.weight_hh
        input_current = (
            torch.mm(input, weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )

        tau_m = self.tau_m.sigmoid()
        tau_adp = self.tau_adp.sigmoid()
        cy = tau_m * cx + (1 - tau_m) * input_current
        hy = Triangle.apply(cy - self.threshold)  # replace the Tanh activation function with step function to ensure binary outputs.

        cb = tau_adp * cb + (1 - tau_adp) * hy
        # CB = 0.1 + 1.84 * cb
        # if self.use_bn:
        #     hy = Triangle.apply(self.batchnorm(cy - self.threshold))
        #     # cy = self.batchnorm(cy)
        # else:
        cy = cy - hy * cb
        return hy, (hy, cy, cb)


if __name__ == "__main__":
    input_size = 256
    hidden_size = 320
    num_layers = 2
    shared_weights = True
    bn = True
    batch_size = 128
    T = 100
    x = torch.rand((batch_size, input_size, T))  # [B, F, T]
    sequence_model = efficient_spiking_neuron(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        shared_weights=shared_weights,
        bn=bn,
    )

    states = [
        MemoryState(
            torch.zeros(batch_size, hidden_size, device=x.device),
            torch.zeros(batch_size, hidden_size, device=x.device),
        )
        for _ in range(num_layers)
    ]
    x = x.permute(2, 0, 1).contiguous()  # [B, F, T] => [T, B, F]
    o, _ = sequence_model(x, states)  # [T, B, F] => [T, B, F]
