import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from ...kernel import temporal_fusion_kernel
except BaseException as e:
    temporal_fusion_kernel = None
from ..trainer import SurrogateGradient as SG
from ..trainer.adaptive_smoothing_gradient_learning import EfficientNoisySpike, InvRectangle
from .base_neuron import BaseNeuron
from .membrane_update import MembraneUpdate


class FusedLIF(Function):
    @staticmethod
    def forward(ctx, tx, rest, decay, threshold, time_step, surro_grad: SG, use_tv: bool = False):
        ctx.rest = rest
        ctx.decay = decay
        ctx.threshold = threshold
        ctx.time_step = time_step
        ctx.surro_grad = surro_grad.func_name
        ctx.sg_kwargs = surro_grad.kwargs
        ty = torch.zeros_like(tx)
        if use_tv:
            v_tv = torch.zeros_like(tx)
        else:
            v_tv = torch.zeros_like(tx[0])
        temporal_fusion_kernel.fusedForwardLIF(tx, v_tv, ty, rest, decay, threshold, time_step, use_tv)
        ctx.tv = v_tv
        ctx.save_for_backward(ty)
        return ty

    @staticmethod
    def backward(ctx, grad_ty):
        (ty,) = ctx.saved_tensors
        tv = ctx.tv
        decay = ctx.decay
        threshold = ctx.threshold
        time_step = ctx.time_step
        surro_grad = ctx.surro_grad
        sg_kwargs = ctx.sg_kwargs
        grad_tx = torch.zeros_like(grad_ty)
        temporal_fusion_kernel.fusedBackwardLIF(grad_ty, grad_tx, ty, tv, decay, threshold, time_step, surro_grad, sg_kwargs)
        return grad_tx, None, None, None, None, None, None


class LIFAct(Function):
    @staticmethod
    def forward(ctx, v, rest, decay, threshold, time_step, surro_grad):
        ctx.save_for_backward(v)
        ctx.rest = rest
        ctx.decay = decay
        ctx.threshold = threshold
        ctx.time_step = time_step
        ctx.surro_grad = surro_grad
        return v.gt(threshold).float()

    @staticmethod
    def backward(ctx, grad_y):
        (v,) = ctx.saved_tensors
        grad_v = grad_y * ctx.surro_grad(
            v,
            rest=ctx.rest,
            decay=ctx.decay,
            threshold=ctx.threshold,
            time_step=ctx.time_step,
        )
        return grad_v, None, None, None, None, None


class LIFAct_thresh(Function):
    @staticmethod
    def forward(ctx, v, rest, decay, threshold, time_step, surro_grad):
        ctx.save_for_backward(v, threshold)
        ctx.rest = rest
        ctx.decay = decay
        ctx.time_step = time_step
        ctx.surro_grad = surro_grad
        return v.gt(threshold).float()

    @staticmethod
    def backward(ctx, grad_y):
        (v,threshold) = ctx.saved_tensors
        grad_v = grad_y * ctx.surro_grad(
            v,
            rest=ctx.rest,
            decay=ctx.decay,
            threshold=threshold,
            time_step=ctx.time_step,
        )
        return grad_v, None, None, -grad_v, None, None


class LIF(BaseNeuron):
    """
        Explicitly Iterative Leaky-Integrate-and-Fire Model.
        Yujie Wu \emph{et al.}, Direct Training for Spiking Neural Networks: Faster, Larger, Better, 2019.

        Hard Reset Case:
        $$ v_i^{(t)} = k_{\tau} \cdot v_i^{(t-1)} \cdot (1 - y_i^{(t-1)}) + V_\text{rest} \cdot y_i^{(t-1)} + x_i^{(t)} $$
        Soft Reset Case:
        $$ v_i^{(t)} = k_{\tau} \cdot v_i^{(t-1)} + (V_\text{rest} - V_\text{th}) \cdot y_i^{(t-1)} + x_i^{(t)} $$
        Final:
        $$ y_i^{(t)} = H(v_i^{(t)} - V_\text{th}) $$
    """

    def __init__(self,
        rest: float      = 0.0,
        decay: float     = 0.2,
        threshold: float = 0.3,
        time_step: int   = None,
        surro_grad: SG   = None,   
        reset_mode: str  = "hard", 
        prop_mode: str   = "STBP", 
        exec_mode: str   = "serial"
    ):
        super(LIF, self).__init__(exec_mode=exec_mode)
        self.rest       = rest
        self.decay      = decay
        self.threshold  = threshold
        self.time_step  = time_step
        self.surro_grad = surro_grad
        self.prop_mode  = prop_mode
        self.reset_mode = reset_mode
        self.mem_update = MembraneUpdate(prop_mode=self.prop_mode, reset_mode=self.reset_mode)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"time_step={self.time_step}, "
            f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"propogation_mode=\"{self.prop_mode}\", "
            f"reset_mode=\"{self.reset_mode}\", "
            f"execution_mode=\"{self.exec_mode}\""
            f")"
        )

    def _serial_process(self, tx, v=None):
        ty = []
        y = torch.zeros_like(tx[0])
        if v is None:
            v = torch.ones_like(tx[0]) * self.rest
            return_v = False
        else:
            return_v = True
        for x in tx:
            v = self.mem_update(x, v, y, self.rest, self.decay, self.threshold)
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
        if return_v:
            return torch.stack(ty), v
        else:
            return torch.stack(ty)

    def _temporal_fused_process(self, tx):
        if self.reset_mode != "hard": raise NotImplementedError
        if self.prop_mode  != "STBP": raise NotImplementedError
        return FusedLIF.apply(tx, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad, self.training)


class RLIF(BaseNeuron):
    """
        Recurrent spiking neural network.
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
        learning_rule: str = "stbp",
        truncated_t: int = 1000,
        bn=None,
        last_layer=False,
    ):
        super(RLIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.truncated_t = truncated_t
        self.learning_rule = learning_rule
        self.recurrent = recurrent
        self.bn = bn
        self.last_layer = last_layer
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
            f"recurrent={self.recurrent}\", "
            f"learning_rule=\"{self.learning_rule}\", "
            f"truncated_t=\"{self.truncated_t}\", "
            f"batchnorm=\"{self.bn}\", "
            f"last_layer=\"{self.last_layer}\", "
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

        if self.recurrent and self.learning_rule == "eprop":
            recurrent_trace = torch.zeros_like(tx[0])

        if self.bn is not None:
            tx = self.bn(tx)

        for t, x in enumerate(tx):
            if self.recurrent:
                if self.training and self.learning_rule == "eprop":
                    recurrent_trace = self.decay * recurrent_trace.detach() + y.detach()
                    recurrent_trace_output = self.recurrent_weight(recurrent_trace.detach())
                    x = x + self.recurrent_weight(y.detach()).detach() + recurrent_trace_output - recurrent_trace_output.detach()
                elif self.learning_rule in ["sltt"]:
                    x = x + self.recurrent_weight(y.detach())
                elif self.learning_rule in ["tbptt", "tstbp"]:
                    if t % self.truncated_t == 0:
                        x = x + self.recurrent_weight(y.detach())
                    else:
                        x = x + self.recurrent_weight(y)
                else:
                    x = x + self.recurrent_weight(y)
            if self.learning_rule == "stbp":
                v = self.decay * v + x
            elif self.learning_rule in ["sdbp", "ottt", "sltt", "eprop"]:
                v = self.decay * v.detach() + x
            elif self.learning_rule == "notd":
                v = x
            elif self.learning_rule in ["tbptt", "tstbp"]:
                if t % self.truncated_t == 0:
                    v = v.detach()
                    y = y.detach()
                v = self.decay * v + x
            else:
                raise NotImplementedError
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            if self.learning_rule in ["sltt", "eprop", "sdbp", "ottt"]:
                v = v - v * y.detach() + self.rest * y.detach()  # Hard reset
            elif self.learning_rule == "notd":
                v = v
            else:
                v = v - v * y + self.rest * y  # Hard reset
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        else:
            return torch.stack(ty)

    def _temporal_fused_process(self, tx):
        if not self.recurrent:
            return FusedLIF.apply(tx, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad, self.training)
        else: 
            pass


class Recurrent_LIF(BaseNeuron):
    """
        Recurrent spiking neural network.
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
        super(Recurrent_LIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.return_mem = False
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
            v = state[0]
            y = state[1]
            return_state = True
        else:
            v = torch.ones_like(tx[0]) * self.rest
            y = torch.zeros_like(tx[0])
            return_state = False
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v = self.decay * v * (1.0 - y) + self.rest * y + x
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)

    def _temporal_fused_process(self, tx):
        if not self.recurrent:
            return FusedLIF.apply(tx, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad, self.training)
        else:
            pass


class NonSpikingLIF(BaseNeuron):
    """
        $$ 
        v_i^{(t)} = k_{\tau} \cdot v_i^{(t-1)} \cdot (1 - y_i^{(t-1)}) + V_\text{rest} \cdot y_i^{(t-1)} + x_i^{(t)} \\
        y_i^{(t)} = v_i^{(t)}
        $$
    """

    def __init__(self,
        rest: float = 0.0,
        decay: float = 0.2,
        time_step: int = None,
        exec_mode: str = "serial",
    ):
        super(NonSpikingLIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.time_step = time_step

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"time_step={self.time_step}, "
            f"execution_mode=\"{self.exec_mode}\""
            f")"
        )

    def _serial_process(self, tx, v=None):
        ty = []
        y = torch.zeros_like(tx[0])
        v = torch.ones_like(tx[0]) * self.rest
        for x in tx:
            v = self.decay * v * (1.0 - y) + self.rest * y + x
            ty.append(v)
        return torch.stack(ty)


class ASGL_LIF(BaseNeuron):
    """
        $$
        v_i^{(t)} = k_{\tau} \cdot v_i^{(t-1)} \cdot (1 - y_i^{(t-1)}) + V_\text{rest} \cdot y_i^{(t-1)} + x_i^{(t)} \\
        y_i^{(t)} = v_i^{(t)}
        $$
    """

    def __init__(self,
        rest: float = 0.0,
        decay: float = 0.2,
        threshold: float = 0.3,
        neuron_num: int = 1,
        time_step: int = None,
        exec_mode: str = "serial",
        a: float = 1.0,
        recurrent: bool = False,
    ):
        super(ASGL_LIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surrogate = EfficientNoisySpike(inv_sg=InvRectangle(alpha=a))
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
        self.return_mem = False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            f"execution_mode=\"{self.exec_mode}\","
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
        self.surrogate.reset_mask()
        for x in tx:
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v = self.decay * v * (1.0 - y) + self.rest * y + x
            y = self.surrogate(v-self.threshold)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)
