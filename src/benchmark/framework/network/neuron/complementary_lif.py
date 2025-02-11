import torch
import torch.nn as nn

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from .lif import LIFAct


class CLIF(BaseNeuron):
    """
        Yulong Huang \emph{et al.}, CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks, 2024.
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
    ):
        super(CLIF, self).__init__(exec_mode=exec_mode)
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
        self.gamma = 0.5

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
            u = state[0]
            y = state[1]
            m = state[2]
            return_state = True
        else:
            u = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            m = torch.zeros_like(tx[0])
            return_state = False

        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            u = self.gamma * u + x
            y = LIFAct.apply(u, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
            m = m * torch.sigmoid_((1. - self.gamma) * u) + y
            u = u - y * (self.threshold + torch.sigmoid_(m))
        if return_state:
            return torch.stack(ty), (u, y, m)
        elif self.return_mem:
            return u.unsqueeze(0)
        else:
            return torch.stack(ty)
