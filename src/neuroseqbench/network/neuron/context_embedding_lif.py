import torch
import torch.nn as nn

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from .lif import LIFAct_thresh,SpikeGeneration


class CELIF(BaseNeuron):
    """
        Xinyi Chen \emph{et al.}, Unleashing the Potential of Spiking Neural Networks for Sequential Modeling with Contextual Embedding, 2023.
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
        beta: float = 0.02,
    ):
        super(CELIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.TE = None
        self.beta = beta
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
        self.return_mem = False
        self.act = SpikeGeneration()

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
            thresh = state[2]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            thresh = torch.ones_like(tx[0]) * self.threshold
            return_state = False
        step = 0
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            thresh = thresh + v * self.TE[:self.neuron_num,step] - (thresh - self.threshold) * self.beta
            v = v * self.decay * (1. - y) + x
            #y = LIFAct_thresh.apply(v, self.rest, self.decay, thresh, self.time_step, self.surro_grad)
            y = self.act(v, self.rest, self.decay, thresh, self.time_step, self.surro_grad,thresh_require_grad=True)
            ty.append(y)
            step = step + 1
        if return_state:
            return torch.stack(ty), (v, y, thresh)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)
