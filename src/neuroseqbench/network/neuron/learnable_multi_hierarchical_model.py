import torch
import torch.nn as nn

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from .lif import LIFAct


class LMH(BaseNeuron):
    """
        Zecheng Hao \emph{et al.}, A Progressive Training Framework for Spiking Neural Networks with Learnable Multi-hierarchical Model, 2024.
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
        a: list = [1, 1, 1, 1],
        b: list = [-0.5, -0.5, 0.5, 0.5],
    ):
        super(LMH, self).__init__(exec_mode=exec_mode)
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

        self.alpha_1 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.beta_1 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.beta_2 = nn.Parameter(torch.tensor([0.]), requires_grad=True)

        self.a = a
        self.b = b

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
            vd = state[0]
            vs = state[1]
            y = state[2]
            return_state = True
        else:
            vd = torch.ones_like(tx[0]) * 0. * self.threshold
            vs = torch.ones_like(tx[0]) * 0.5 * self.threshold
            y = torch.zeros_like(tx[0])
            return_state = False
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            vd = (self.a[0] * self.alpha_1.sigmoid() + self.b[0]) * vd + (self.a[1] * self.beta_1.sigmoid() + self.b[1]) * vs + x
            vs = (self.a[2] * self.alpha_2.sigmoid() + self.b[2]) * vs + (self.a[3] * self.beta_2.sigmoid() + self.b[3]) * vd
            y = LIFAct.apply(vs, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
            vs = vs - y.detach()
        if return_state:
            return torch.stack(ty), (vd, vs, y)
        elif self.return_mem:
            return vs.unsqueeze(0)
        else:
            return torch.stack(ty)
