import torch
import torch.nn as nn

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from .lif import LIFAct


class TCLIF(BaseNeuron):
    """
        Shimin Zhang \emph{et al.}, TC-LIF: A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling, 2024.
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
        beta1: float = 0.,
        beta2: float = 0.,
        gamma: float = 0.5
    ):
        super(TCLIF, self).__init__(exec_mode=exec_mode)
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

        decay_factor = torch.full([1, 2], 0, dtype=torch.float)

        decay_factor[0][0] = beta1
        decay_factor[0][1] = beta2

        self.gamma = gamma
        self.decay_factor = torch.nn.Parameter(decay_factor)

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
            v1 = state[0]
            v2 = state[1]
            y = state[2]
            return_state = True
        else:
            v1 = torch.ones_like(tx[0]) * self.rest
            v2 = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            return_state = False
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v1 = v1 - torch.sigmoid(self.decay_factor[0][0]) * v2 + x
            v2 = v2 + torch.sigmoid(self.decay_factor[0][1]) * v1
            y = LIFAct.apply(v2, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
            v1 = v1 - y * self.gamma
            v2 = v2 - y * self.threshold
        if return_state:
            return torch.stack(ty), (v1, v2, y)
        elif self.return_mem:
            return v2.unsqueeze(0)
        else:
            return torch.stack(ty)
