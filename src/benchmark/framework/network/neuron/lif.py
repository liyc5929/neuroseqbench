import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from ...kernel import temporal_fusion_kernel
except BaseException as e:
    temporal_fusion_kernel = None
from ..trainer import SurrogateGradient as SG
from .base_neuron import BaseNeuron


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
        (v, threshold) = ctx.saved_tensors
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
        Leaky Integrate-and-Fire model.
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
        super(LIF, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
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


        for t, x in enumerate(tx):
            if self.recurrent:
                x = x + self.recurrent_weight(y)
            v = self.decay * v * (1.0 - y) + self.rest * y + x
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        else:
            return torch.stack(ty)

    def _temporal_fused_process(self, tx):
        if not self.recurrent:
            return FusedLIF.apply(tx, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad, self.training)
