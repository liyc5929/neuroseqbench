import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Function

try:
    from ...kernel import temporal_fusion_kernel
except BaseException as e:
    temporal_fusion_kernel = None
from ..trainer import SurrogateGradient as SG
from ..trainer.ASGL_surrogate import EfficientNoisySpike, InvSigmoid, InvRectangle
from ..trainer.surrogate import PMSN_surrogate
from .base_neuron import BaseNeuron
from torch.autograd import Variable
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

    def __init__(
        self,
        rest: float      = 0.0,
        decay: float     = 0.2,
        threshold: float = 0.3,
        neuron_num: int  = -1,
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

    def __init__(
            self,
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
            last_layer=False
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

        if self.recurrent and self.learning_rule == 'eprop':
            recurrent_trace = torch.zeros_like(tx[0])

        if self.bn is not None:
            tx = self.bn(tx)

        for t, x in enumerate(tx):
            if self.recurrent:
                if self.training and self.learning_rule == 'eprop':
                    recurrent_trace = self.decay * recurrent_trace.detach() + y.detach()
                    recurrent_trace_output = self.recurrent_weight(recurrent_trace.detach())
                    x = x + self.recurrent_weight(y.detach()).detach() + recurrent_trace_output - recurrent_trace_output.detach()
                elif self.learning_rule in ['sltt']:
                    x = x + self.recurrent_weight(y.detach())
                elif self.learning_rule in ['tbptt', 'tstbp']:
                    if t % self.truncated_t == 0:
                        x = x + self.recurrent_weight(y.detach())
                    else:
                        x = x + self.recurrent_weight(y)
                else:
                    x = x + self.recurrent_weight(y)
            if self.learning_rule == 'stbp':
                v = self.decay * v  + x
            elif self.learning_rule in ['sdbp', 'ottt', 'sltt', 'eprop']:
                v = self.decay * v.detach() + x
            elif self.learning_rule == 'notd':
                v = x
            elif self.learning_rule in ['tbptt', 'tstbp']:
                if t % self.truncated_t == 0:
                    v = v.detach()
                    y = y.detach()
                v = self.decay * v + x
            else:
                raise NotImplementedError
            # if not self.last_layer:
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            if self.learning_rule in ['sltt', 'eprop', 'sdbp', 'ottt']:
                v = v - v * y.detach() + self.rest * y.detach()  # Hard reset
                # print(f"t: {t} y: {y.sum()} y size: {y.size()}")
            elif self.learning_rule == 'notd':
                v = v
            else:
                v = v - v * y + self.rest * y  # Hard reset
            # else:
            #     y = v
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        else:
            return torch.stack(ty)

    def _temporal_fused_process(self, tx):
        if not self.recurrent:
            return FusedLIF.apply(tx, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad,
                                  self.training)
        else: 
            pass


class Recurrent_LIF(BaseNeuron):
    """
        Recurrent spiking neural network.
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None, 
            exec_mode: str = "serial",
            recurrent: bool = False
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
            #print(torch.stack(ty).mean())
        if return_state:
            return torch.stack(ty), (v, y)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)

    def _temporal_fused_process(self, tx):
        if not self.recurrent:
            return FusedLIF.apply(tx, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad,
                                  self.training)
        else:
            pass


class NonSpikingLIF(BaseNeuron):
    """
        $$ 
        v_i^{(t)} = k_{\tau} \cdot v_i^{(t-1)} \cdot (1 - y_i^{(t-1)}) + V_\text{rest} \cdot y_i^{(t-1)} + x_i^{(t)} \\
        y_i^{(t)} = v_i^{(t)}
        $$
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            time_step: int = None,
            exec_mode: str = "serial"
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

    # def _temporal_fused_process(self, tx):
    #     pass 


class ASGL_LIF(BaseNeuron):
    """
        $$
        v_i^{(t)} = k_{\tau} \cdot v_i^{(t-1)} \cdot (1 - y_i^{(t-1)}) + V_\text{rest} \cdot y_i^{(t-1)} + x_i^{(t)} \\
        y_i^{(t)} = v_i^{(t)}
        $$
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            exec_mode: str = "serial",
            a: float = 1.0,
            recurrent: bool = False
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


class PLIF(BaseNeuron):
    """
        Altered from Spikingjelly
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None, 
            exec_mode: str = "serial",
            recurrent: bool = False,
            init_w: float = 0
    ):
        super(PLIF, self).__init__(exec_mode=exec_mode)
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
        self.w = nn.Parameter(torch.as_tensor(init_w))
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
            v = self.w.sigmoid() * v * (1.0 - y) + self.rest * y + x
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)




class ALIF(BaseNeuron):
    """
        Altered from https://github.com/byin-cwi/Efficient-spiking-networks
    """

    def __init__(
            self,
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
        #print(tau_adp[0], tau_adp[1],tau_m[0], tau_m[1],self.beta)

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
        #mem = mem * alpha + (1 - alpha) * 1. * inputs - B * spike * self.dt # original seting
        #mem = mem * alpha + inputs - B * spike * self.dt # hard reset
        mem = mem * alpha + inputs - B * spike * self.dt  # remove the (1-alpha)

        spike = LIFAct_thresh.apply(mem, self.rest, self.decay, B, self.time_step, self.surro_grad)
        return mem, spike, B, b




class GLIF(BaseNeuron):
    """
        Altered from https://github.com/Ikarosy/Gated-LIF
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  
            exec_mode: str = "serial",
            recurrent: bool = False,
            gate: list = [0.8, 0.2, 0.8]
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
            for i in self.gate]

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.neuron_num, dtype=torch.float))
                                         for i in self.param[:-1]]
        self.reVth = nn.Parameter(- math.log(1 / self.param[1] - 1) * torch.ones(self.neuron_num, dtype=torch.float))
        # t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.time_step, self.neuron_num), dtype=torch.float))
                        for i in self.param[3:]][0]
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
            v, y = self.extended_state_update(v, y, x, tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
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
        u_t_n1 = ((1 - al * (1 - tau[None, :])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :]) + \
            I_t1 - (1 - ga) * reVth[None, :] * o_t_n1.clone()
        o_t_n1 = LIFAct_thresh.apply(u_t_n1, self.rest, self.decay, Vth[None, :], self.time_step, self.surro_grad)
        return u_t_n1, o_t_n1



class CLIF(BaseNeuron):
    """
        Altered from https://github.com/HuuYuLong/Complementary-LIF
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None, 
            exec_mode: str = "serial",
            recurrent: bool = False
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


class CELIF(BaseNeuron):
    """
        Implementation for https://arxiv.org/abs/2308.15150.
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  
            exec_mode: str = "serial",
            recurrent: bool = False,
            beta: float = 0.02
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
            y = LIFAct_thresh.apply(v, self.rest, self.decay, thresh, self.time_step, self.surro_grad)
            ty.append(y)
            step = step + 1
        if return_state:
            return torch.stack(ty), (v, y, thresh)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)


class TCLIF(BaseNeuron):
    """
        Altered from https://github.com/ZhangShimin1/TC-LIF
    """

    def __init__(
            self,
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



class PSN(BaseNeuron):
    """
        Altered from Spikingjelly
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None, 
            exec_mode: str = "serial",
            recurrent: bool = False
    ):
        super(PSN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.return_mem = False

        weight = torch.zeros([self.time_step, self.time_step])
        bias = torch.zeros([self.time_step, 1])
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.thresh = torch.tensor([self.threshold]).cuda()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, 0.)
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
        if tx.size(0) < self.time_step:
            padding = torch.zeros(self.time_step-tx.size(0), tx.size(1),tx.size(2)).to(tx.device)
            tx = torch.cat((tx, padding), 0)
        v = torch.addmm(self.bias, self.weight, tx.flatten(1))[:tx.size(0)][:step_num,]
        ty = LIFAct_thresh.apply(v, self.rest, self.decay, self.thresh, self.time_step, self.surro_grad)
        ty = ty.view(step_num, -1, self.neuron_num)

        if return_state:
            return ty, (state)
        elif self.return_mem:
            return v[-1,].unsqueeze(0)
        else:
            return ty



class SPSN(BaseNeuron):
    """
        Altered from Spikingjelly
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  
            exec_mode: str = "serial",
            recurrent: bool = False
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

        #self.register_memory('queue', [])
        self.k = 32
        self.backend = 'conv'
        self.thresh = torch.tensor([self.threshold]).cuda()


        weight = torch.ones([self.k])
        for i in range(self.k - 2, -1, -1):
            weight[i] = weight[i + 1] / 2.

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.as_tensor(-0.))
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

        ty = LIFAct_thresh.apply(v, self.rest, self.decay, self.thresh, self.time_step, self.surro_grad)

        if return_state:
            return ty, (state)
        elif self.return_mem:
            return v[-1,].unsqueeze(0)
        else:
            return ty



class LMH(BaseNeuron):
    """
        Altered from https://github.com/hzc1208/STBP_LMH
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None, 
            exec_mode: str = "serial",
            recurrent: bool = False,
            a: list = [1, 1, 1, 1],
            b: list = [-0.5, -0.5, 0.5, 0.5]
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



class sigmoid_beta(nn.Module):
    def __init__(self, alpha=1., is_train=False):
        super(sigmoid_beta, self).__init__()
        # initialize alpha
        if alpha == None:
            self.alpha = nn.Parameter(torch.tensor(1.))  # create a tensor out of alpha
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))  # create a tensor out of alpha

        self.alpha.requiresGrad = is_train  # set requiresGrad to true!

    def forward(self, x):
        if (self.alpha == 0.0):
            return x
        else:
            return torch.sigmoid(self.alpha * x)


class LTC(BaseNeuron):
    """
        Altered from https://github.com/byin-cwi/sFPTT/blob/main/fptt/fptt_mnist/snn_models_LIF4_save4.py
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  
            exec_mode: str = "serial",
            recurrent: bool = False,
            b_j0: float = 0.2
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
        # self.act1 = sigmoid_beta(is_train=True)
        # self.act2 = sigmoid_beta(is_train=True)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()
        self.layer1_tauM = nn.Linear(self.neuron_num * 2, self.neuron_num)
        self.layer1_tauAdp = nn.Linear(self.neuron_num * 2, self.neuron_num)
        nn.init.xavier_normal_(self.layer1_tauM.weight)
        nn.init.xavier_normal_(self.layer1_tauAdp.weight)
        nn.init.constant_(self.layer1_tauM.bias, 0)
        nn.init.constant_(self.layer1_tauAdp.bias, 0)

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
            #b = self.b_j0 * torch.ones_like(tx[0])
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

            y = LIFAct_thresh.apply(mem, self.rest, self.decay, B, self.time_step, self.surro_grad)
            mem = (1 - y) * mem
            ty.append(y)
            step = step + 1
        #print(self.neuron_num,torch.stack(ty)[0].mean().item(),torch.stack(ty)[-1].mean().item(), tx[0].mean().item(), tx[-1].mean().item(), mem.mean().item(), b.mean().item(), alpha.mean().item(),ro.mean().item())
        if return_state:
            return torch.stack(ty), (mem, y, b)
        elif self.return_mem:
            return mem.unsqueeze(0)
        else:
            return torch.stack(ty)



class DHSNN(BaseNeuron):
    """
        Altered from https://github.com/eva1801/DH-SNN
    """

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            input_features: int = 1,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,
            exec_mode: str = "serial",
            recurrent: bool = False,
            branch: int = 4,
            tau_m: list = [0, 4],
            tau_n: list = [0, 4],
            zero_init: bool = True
    ):
        super(DHSNN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.input_features = input_features
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.surro_grad = surro_grad
        self.recurrent = recurrent
        self.zero_init = zero_init

        if self.recurrent:
            self.pad = ((input_features + neuron_num) // branch * branch + branch - (input_features + neuron_num)) % branch
            self.dense = nn.Linear(input_features + neuron_num + self.pad, neuron_num * branch)

        else:
            self.pad = ((input_features) // branch * branch + branch - (input_features)) % branch
            self.dense = nn.Linear(input_features + self.pad, neuron_num * branch)
        self.return_mem = False
        mask_rate = 1 / branch

        self.tau_m = nn.Parameter(torch.Tensor(self.neuron_num))
        self.tau_n = nn.Parameter(torch.Tensor(self.neuron_num, branch))
        # the number of dendritic branch
        self.branch = branch
        self.create_mask()

        nn.init.uniform_(self.tau_m, tau_m[0], tau_m[1])
        nn.init.uniform_(self.tau_n, tau_n[0], tau_n[1])

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
            f"recurrent={self.recurrent}, "
            f"branch={self.branch}"
            f")"
        )

    def _serial_process(self, tx, state=None):

        ty = []
        if isinstance(state, tuple):
            v = state[0]
            y = state[1]
            d_input = state[2]
            return_state = True
        else:
            v = torch.ones(tx.size(1), self.neuron_num, device=tx.device) * self.rest
            y = torch.zeros(tx.size(1), self.neuron_num, device=tx.device)
            if self.zero_init:
                d_input = torch.zeros(tx.size(1), self.neuron_num, self.branch, device=tx.device)
            else:
                d_input = torch.rand(tx.size(1), self.neuron_num, self.branch, device=tx.device)
            return_state = False

        for x in tx:
            beta = torch.sigmoid(self.tau_n)
            padding = torch.zeros(x.size(0), self.pad).to(x.device)
            if self.recurrent:
                x = torch.cat((x.float(), y, padding), 1)
            else:
                x = torch.cat((x.float(), padding), 1)
            x = self.dense(x)

            # update dendritic currents

            d_input = beta * d_input + (1 - beta) * x.reshape(-1, self.neuron_num, self.branch)
            # summation of dendritic currents

            l_input = d_input.sum(dim=2, keepdim=False)  # thr original setting

            alpha = torch.sigmoid(self.tau_m)

            v = v * alpha + l_input - self.threshold * y
            #v = v * alpha + (1 - alpha) * l_input * self.R_m - self.threshold * y  # thr original setting
            y = LIFAct.apply(v, self.rest, self.decay, self.threshold, self.time_step, self.surro_grad)
            ty.append(y)
        if return_state:
            return torch.stack(ty), (v, y, d_input)
        elif self.return_mem:
            return v.unsqueeze(0)
        else:
            return torch.stack(ty)

    def create_mask(self):
        if self.recurrent:
            input_size = self.input_features + self.neuron_num + self.pad
        else:
            input_size = self.input_features + self.pad
        self.mask = torch.zeros(self.neuron_num * self.branch, input_size).cuda()
        for i in range(self.neuron_num):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                self.mask[i * self.branch + j, seq[j * input_size // self.branch:(j + 1) * input_size // self.branch]] = 1
    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data * self.mask

class adLIF(BaseNeuron):
    """
        Altered from https://github.com/idiap/sparch
    """

    def __init__(
            self,
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




class PMSN(BaseNeuron):

    def __init__(
            self,
            rest: float = 0.0,
            decay: float = 0.2,
            threshold: float = 0.3,
            neuron_num: int = 1,
            time_step: int = None,
            surro_grad: SG = None,  
            exec_mode: str = "serial",
            recurrent: bool = False,
    ):
        super(PMSN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.recurrent = recurrent
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
        self.return_mem = False

        self.kernel = PMSN_kernel(self.neuron_num, N=4)
        self.D = nn.Parameter(torch.randn(self.neuron_num))
        self.thresh = torch.tensor([self.threshold])
        self.bn = nn.BatchNorm1d(self.neuron_num)
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            #f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        if isinstance(state, tuple):
            tx[0,] = tx[0,] + state[0]
            return_state = True
        else:
            return_state = False
        step_num = tx.size(0) #(T,B,H)

        tx = self.bn(tx.view(-1, tx.size(-1))).view(step_num, -1, self.neuron_num)
        tx = tx.permute(1,2,0) #(B, H, T)
        # Compute SSM Kernel
        k = self.kernel(L=step_num,u=tx) # (H T)

        # # Convolution
        k_f = torch.fft.rfft(k, n=2*step_num) # (H T)
        u_f = torch.fft.rfft(tx, n=2*step_num) # (B H T)
        _y = torch.fft.irfft(u_f*k_f, n=2*step_num)[..., :step_num] # (B H T)
        y = _y + (tx * self.D.unsqueeze(-1))
        # proposed reset mechanism
        ty = PMSN_surrogate.apply(y.relu(), self.thresh.to(tx.device))
        ty = ty.permute(2,0,1)

        if return_state:
            return ty, (_y[...,-1], None)
        elif self.return_mem:
            return y[-1,].unsqueeze(0)
        else:
            return ty

class PMSN_kernel(nn.Module):
    def __init__(self, d_model, N=4, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H).uniform_(0, 1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)  # [H]

        self.log_dt = nn.Parameter(log_dt)
        diag_indices = torch.arange(N)
        sub_diag_indices = diag_indices[:-1] + 1
        super_diag_indices = diag_indices[1:] - 1

        S = torch.zeros(N,N)
        S[diag_indices, diag_indices] = -0.5
        S[diag_indices[:-1], sub_diag_indices] = 5. * ((torch.arange(N-1)+1))
        S[diag_indices[1:], super_diag_indices] = -5. * ((torch.arange(N-1)+1)) # 超对角线

        S_diag = torch.diagonal(S)
        A_real = (torch.mean(S_diag) * torch.ones_like(S_diag)).unsqueeze(0).repeat(H,1)

        A_imag, V = torch.linalg.eigh(S * -1j)  # [N; N,N]
        A_imag = A_imag.unsqueeze(0).repeat(H,1)

        self.mask = torch.zeros(N,N).cuda()
        self.mask[diag_indices, diag_indices] = 1
        self.mask[diag_indices[:-1], sub_diag_indices] = 1


        log_A_real = torch.log(-A_real)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        B = torch.ones(H, N)
        C= torch.zeros(H,N)
        C[:,-1]=1
        Vinv=V.conj().T  # [N,N]
        CV= torch.einsum('hm,mn->hn',C+0j,V) # [H,N]
        VinvB=torch.einsum('mn,hn->hm',Vinv,B+0j) #[H,N]

        self.VinvB_real = nn.Parameter(VinvB.real)
        self.VinvB_imag = nn.Parameter(VinvB.imag)
        self.CV_real = nn.Parameter(CV.real)
        self.CV_imag = nn.Parameter(CV.imag)

    def forward(self, L, u=None):
        # u [B,H,L]
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)
        B = self.VinvB_real + 1j * self.VinvB_imag  #(H,N)
        C = self.CV_real + self.CV_imag * 1j

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H,1)
        A_bar = torch.exp(A*dt.unsqueeze(-1))  #[H N]
        B_bar = (A_bar-1)*B/A
        # Vandermonde multiplication
        logK = (A*dt.unsqueeze(-1)).unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)   e-At
        K = torch.exp(logK)
        KB = torch.einsum('hnl,hn->hnl',K,B_bar)  # e-At*B  # (H N L)
        CKB = torch.einsum('hn, hnl -> hl', C, KB).real #(H L)
        return CKB
