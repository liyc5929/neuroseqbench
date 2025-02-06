import math
import torch
import torch.nn as nn

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from .lif import LIFAct_thresh


class GLIF(BaseNeuron):
    """
        Xingting Yao \emph{et al.}, GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks, 2022.
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
        gate: list = [0.8, 0.2, 0.8],
    ):
        super(GLIF, self).__init__(exec_mode=exec_mode)
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

        self.gate = gate
        self.param = [0.25, self.threshold, 0.5 / 8, 0.5]
        self.alpha, self.beta, self.gamma = [
            nn.Parameter(- math.log(1 / ((i - 0.5) * 0.5 + 0.5) - 1) * torch.ones(self.neuron_num, dtype=torch.float))
            for i in self.gate
        ]

        self.tau, self.Vth, self.leak = [
            nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.neuron_num, dtype=torch.float))
            for i in self.param[:-1]
        ]
        self.reVth = nn.Parameter(- math.log(1 / self.param[1] - 1) * torch.ones(self.neuron_num, dtype=torch.float))
        # t, c
        self.conduct = [
            nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.time_step, self.neuron_num), dtype=torch.float))
            for i in self.param[3:]
        ][0]

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
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            return_state = False

        step = 0
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v, y = self.extended_state_update(v, y, x, 
                tau=self.tau.sigmoid(),
                Vth=self.Vth.sigmoid(),
                leak=self.leak.sigmoid(),
                conduct=self.conduct[step].sigmoid(),
                reVth=self.reVth.sigmoid()
            )
            ty.append(y)
            step = step + 1
        if return_state:
            return torch.stack(ty), (v, y)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)

    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        # [v: T B C]
        al, be, ga = self.alpha.view(1, -1).sigmoid(), self.beta.view(1, -1).sigmoid(), self.gamma.view(1, -1).sigmoid()
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :]))
        u_t_n1 = ((1 - al * (1 - tau[None, :])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :]) \
               + I_t1 - (1 - ga) * reVth[None, :] * o_t_n1.clone()
        o_t_n1 = LIFAct_thresh.apply(u_t_n1, self.rest, self.decay, Vth[None, :], self.time_step, self.surro_grad)
        return u_t_n1, o_t_n1
