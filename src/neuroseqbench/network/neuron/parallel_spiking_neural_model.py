import torch
import torch.nn as nn

from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron
from . import SpikeGeneration


class SPSN(BaseNeuron):
    """
        Wei Fang \emph{et al.}, Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies, 2023.
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
        k: int = 32
    ):
        super(SPSN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.return_mem = False
        self.act = SpikeGeneration()

        self.k = k
        self.backend = "conv"
        self.thresh = torch.tensor([self.threshold]).cuda()

        weight = torch.ones([self.k])
        for i in range(self.k - 2, -1, -1):
            weight[i] = weight[i + 1] / 2.0

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.as_tensor(-0.0))

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
        step_num = tx.size(0)
        if isinstance(state, tuple):
            return_state = True
        else:
            return_state = False

        x_seq = tx.flatten(1).t().unsqueeze(1)
        x_seq = nn.functional.pad(x_seq, pad=(self.k - 1, 0))
        v = nn.functional.conv1d(x_seq, self.weight.view(1, 1, -1), stride=1)

        v = v.squeeze(1).t().contiguous().view(step_num,-1,self.neuron_num) + self.bias * self.thresh

        ty = self.act(v, self.rest, self.decay, self.thresh, self.time_step, self.surro_grad, thresh_require_grad=True)
        if return_state:
            return ty, (state)
        elif self.return_mem:
            return v[-1,].unsqueeze(0)
        else:
            return ty
