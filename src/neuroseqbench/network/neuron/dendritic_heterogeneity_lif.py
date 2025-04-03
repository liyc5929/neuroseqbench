import torch
import torch.nn as nn

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from .lif import LIFAct


class DHSNN(BaseNeuron):
    """
        Hanle Zheng \emph{et al.}, Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics, 2024.
    """

    def __init__(self,
        rest: float = 0.0,
        decay: float = 0.2,
        threshold: float = 0.3,
        input_features: int = 1,
        neuron_num: int = 1,
        time_step: int = None,
        surro_grad: SG = None,
        exec_mode: str = "serial",
        recurrent: bool = False,
        branch: int = 4,
        tau_m: list = [0, 4],
        tau_n: list = [0, 4],
        zero_init: bool = True,
    ):
        super(DHSNN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.input_features = input_features
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.zero_init = zero_init

        if self.recurrent:
            self.pad = ((input_features + neuron_num) // branch * branch + branch - (input_features + neuron_num)) % branch
            self.dense = nn.Linear(input_features + neuron_num + self.pad, neuron_num * branch)

        else:
            self.pad = ((input_features) // branch * branch + branch - (input_features)) % branch
            self.dense = nn.Linear(input_features + self.pad, neuron_num * branch)
        self.return_mem = False

        self.tau_m = nn.Parameter(torch.Tensor(self.neuron_num))
        self.tau_n = nn.Parameter(torch.Tensor(self.neuron_num, branch))
        self.branch = branch # the number of dendritic branch
        self.create_mask()

        nn.init.uniform_(self.tau_m, tau_m[0], tau_m[1])
        nn.init.uniform_(self.tau_n, tau_n[0], tau_n[1])

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}, "
            f"branch={self.branch}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            d_input = state[2]
            return_state = True
        else:
            v = torch.ones(tx.size(1), self.neuron_num, device=tx.device) * self.rest
            y = torch.zeros(tx.size(1), self.neuron_num, device=tx.device)
            if self.zero_init:
                d_input = torch.zeros(tx.size(1), self.neuron_num, self.branch, device=tx.device)
            else:
                d_input = torch.rand(tx.size(1), self.neuron_num, self.branch, device=tx.device)
            return_state = False

        for x in tx:
            beta = torch.sigmoid(self.tau_n)
            padding = torch.zeros(x.size(0), self.pad).to(x.device)
            if self.recurrent:
                x = torch.cat((x.float(), y, padding), 1)
            else:
                x = torch.cat((x.float(), padding), 1)
            x = self.dense(x)

            # Update dendritic currents
            d_input = beta * d_input + (1 - beta) * x.reshape(-1, self.neuron_num, self.branch)

            l_input = d_input.sum(dim=2, keepdim=False)

            alpha = torch.sigmoid(self.tau_m)

            v = v * alpha + l_input - self.threshold * y
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y, d_input)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)

    def create_mask(self):
        if self.recurrent:
            input_size = self.input_features + self.neuron_num + self.pad
        else:
            input_size = self.input_features + self.pad
        self.mask = torch.zeros(self.neuron_num * self.branch, input_size).cuda()
        for i in range(self.neuron_num):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                self.mask[i * self.branch + j, seq[j * input_size // self.branch:(j + 1) * input_size // self.branch]] = 1

    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data * self.mask
