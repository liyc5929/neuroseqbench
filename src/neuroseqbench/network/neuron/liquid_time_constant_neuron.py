import torch
import torch.nn as nn

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from .lif import LIFAct_thresh,SpikeGeneration


class LTC(BaseNeuron):
    """
        Bojian Yin \emph{et al.}, Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time, 2023.
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
        b_j0: float = 0.2,
    ):
        super(LTC, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.return_mem = False
        self.beta = 0.2
        self.b_j0 = b_j0
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()
        self.layer1_tauM = nn.Linear(self.neuron_num * 2, self.neuron_num)
        self.layer1_tauAdp = nn.Linear(self.neuron_num * 2, self.neuron_num)
        nn.init.xavier_normal_(self.layer1_tauM.weight)
        nn.init.xavier_normal_(self.layer1_tauAdp.weight)
        nn.init.constant_(self.layer1_tauM.bias, 0)
        nn.init.constant_(self.layer1_tauAdp.bias, 0)
        self.act = SpikeGeneration()
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)

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
            mem = state[0]
            y = state[1]
            b = state[2]
            return_state = True
        else:
            mem = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            b = self.threshold * torch.ones_like(tx[0])
            return_state = False
        step = 0
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)

            alpha = self.act1(self.layer1_tauM(torch.cat((x, mem), dim=-1))) # to avoid gradient explosion
            ro = self.act2(self.layer1_tauAdp(torch.cat((x, b), dim=-1)))
            beta = self.beta

            b = ro * b + (1 - ro) * y
            B = self.threshold + beta * b

            d_mem = - mem + x
            mem = mem + d_mem * alpha

            #y = LIFAct_thresh.apply(mem, self.rest, self.decay, B, self.time_step, self.surro_grad)
            y = self.act(mem, self.rest, self.decay, B, self.time_step, self.surro_grad, thresh_require_grad=True)
            mem = (1 - y) * mem
            ty.append(y)
            step = step + 1
        if return_state:
            return torch.stack(ty), (mem, y, b)
        elif self.return_mem:
            return mem.unsqueeze(0)
        else:
            return torch.stack(ty)
