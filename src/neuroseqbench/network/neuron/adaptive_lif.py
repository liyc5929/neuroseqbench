import torch
import torch.nn as nn
import numpy as np

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from .lif import LIFAct_thresh, LIFAct


class ALIF(BaseNeuron):
    """
        Bojian Yin \emph{et al.}, Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks, 2021.
    """

    def __init__(self,
        rest: float = 0.0,
        decay: float = 0.2,
        threshold: float = 0.3,
        neuron_num: int = 1,
        time_step: int = None,
        surro_grad: SG = None,  
        exec_mode: str = "serial",
        recurrent: bool = False,
        beta : float = 1.8,
        tau_adp: list = [700, 25],
        tau_m: list = [20, 5],
        gain: float = 1.,
    ):
        super(ALIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
        self.return_mem = False
        self.tau_adp = nn.Parameter(torch.Tensor(self.neuron_num))
        self.tau_m = nn.Parameter(torch.Tensor(self.neuron_num))
        self.beta = beta
        self.dt = 1.
        self.input_gain = gain
        nn.init.normal_(self.tau_adp, tau_adp[0], tau_adp[1])
        nn.init.normal_(self.tau_m, tau_m[0], tau_m[1])

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
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            b = state[2]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            b = 0.01
            return_state = False

        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v, y, thresh, b = self.mem_update_adp(x, v, y, self.tau_adp, self.tau_m, b)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y, b)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)

    def mem_update_adp(self, inputs, mem, spike, tau_adp, tau_m, b):
        ro = torch.exp(-1. * self.dt / tau_adp).cuda()
        alpha = torch.exp(-1. * self.dt / tau_m).cuda()
        b = ro * b + (1 - ro) * spike
        B = self.threshold + self.beta * b
        mem = mem * alpha + inputs - B * spike * self.dt

        spike = LIFAct_thresh.apply(mem, self.rest, self.decay, B, self.time_step, self.surro_grad)
        return mem, spike, B, b


class adLIF(BaseNeuron):
    """
        Alexandre Bittar \emph{et al.}, A surrogate gradient spiking baseline for speech command recognition, 2022.
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
        a_lim: list = [-1.0, 1.0],
        b_lim: list = [0.0, 2.0],
        decay_lim: list = [5, 25, 30, 120],
        init_zero: bool = False,
    ):
        super(adLIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.input_features = input_features
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent

        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
            nn.init.orthogonal_(self.recurrent_weight.weight)
        self.return_mem = False

        self.alpha_lim = [np.exp(-1 / decay_lim[0]), np.exp(-1 / decay_lim[1])]
        self.beta_lim = [np.exp(-1 / decay_lim[2]), np.exp(-1 / decay_lim[3])]

        self.a_lim = a_lim
        self.b_lim = b_lim

        # Trainable parameters
        self.alpha = nn.Parameter(torch.Tensor(self.neuron_num))
        self.beta = nn.Parameter(torch.Tensor(self.neuron_num))
        self.a = nn.Parameter(torch.Tensor(self.neuron_num))
        self.b = nn.Parameter(torch.Tensor(self.neuron_num))

        self.norm = nn.BatchNorm1d(self.neuron_num, momentum=0.05)

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        if init_zero:
            nn.init.uniform_(self.a, 0., 0.)
            nn.init.uniform_(self.b, 0., 0.)
        else:
            nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
            nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

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
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            wt = state[2]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            wt = torch.zeros_like(tx[0])
            return_state = False

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        _tx = self.norm(tx.reshape(tx.shape[0] * tx.shape[1], tx.shape[2]))
        tx = _tx.reshape(tx.shape[0], tx.shape[1], tx.shape[2])

        for x in tx:
            if self.recurrent:
                # Set diagonal elements of recurrent matrix to zero
                r_weight = self.recurrent_weight.weight.clone().fill_diagonal_(0)
                x = x + torch.matmul(y, r_weight)
            # Compute potential (adLIF)
            wt = beta * wt + a * v + b * y
            v = alpha * (v - y) + (1 - alpha) * (x - wt)

            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)

        if return_state:
            return torch.stack(ty), (v, y, wt)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)
