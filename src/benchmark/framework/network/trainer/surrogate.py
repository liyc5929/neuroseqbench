import torch
import math


def _rectangular_function(v, threshold, a, b = 1., *_, **__):
    """
        Yujie Wu \emph{et al.}, Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks, 2018.
        $$ h(u) = \frac{1}{a} \cdot \mathrm{sign}(|u - V_\text{th}| < \frac{a}{2}) $$
        TODO: replace the '/a' by 'b' and change the citation
    """
    grad_v = (torch.abs(v - threshold) < (a / 2)) * b
    return grad_v


def _triangle_function(v, threshold, a, *_, **__):
    """
        Altered from the code of Temporal Efficient Training, ICLR 2022 (https://openreview.net/forum?id=_XNtisL32jv)
        max(0, 1 - |ui[t] - θ|)
    """

    grad_v = (1 / a) * (1 / a) * ((a - abs(v - threshold)).clamp(min=0))
    return grad_v

def _sigmoid_function(v, threshold, a, *_, **__):
    """
       Yujie Wu \emph{et al.}, Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks, 2018.
    """
    a = a / 4  # Default 1 / a = 4
    sgax = ((v - threshold) / a).sigmoid_()
    grad_v = (1. - sgax) * sgax / a
    return grad_v

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

def _multigaussian_function(v, threshold, a, *_, **__):
    """
        Revised from https://github.com/byin-cwi/Efficient-spiking-networks
    """
    input = v - threshold
    scale = 6.0
    hight = .15
    a = a / 2  # According to the paper, default a = 0.5 in multigauss .
    temp = gaussian(input, mu=0., sigma=a) * (1. + hight) - gaussian(input, mu=a, sigma=scale * a) * hight - gaussian(input, mu=-a, sigma=scale * a) * hight
    grad_v = temp.float() * a

    return grad_v
def _atan_function(v, threshold, a, *_, **__):
    """
    W. Fang \emph{et al.}, Incorporating Learnable Membrane Time Constants to Enhance Learning of Spiking Neural Networks, 2021.
    """
    x = v - threshold
    a = 2 / a # Default a = 2
    grad_v = a / (2 * (1 + (2 / math.pi * a * x) * (2 / math.pi * a * x)))
    return grad_v

__func_config__ = {
    "rectangle": _rectangular_function,
    "triangle" : _triangle_function,
    "multigauss" : _multigaussian_function,
    "sigmoid"  : _sigmoid_function,
    "atan"     : _atan_function,
    "others"   : None, # TODO: Add more surrogate funcions
}


class SurrogateGradient:
    def __init__(self, func_name: str, *args, **kwargs):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, v, *args, **kwargs):
        surro_func = __func_config__.get(self.func_name)
        if surro_func is not None:
            return surro_func(v, *(self.args + args), **{**self.kwargs, **kwargs})
        else:
            raise ValueError("Invalid surrogate gradient function name.")


class TriangleSurroGrad(torch.autograd.Function):
    """Altered from code of Temporal Efficient Training, ICLR 2022 (https://openreview.net/forum?id=_XNtisL32jv)
    max(0, 1 - |ui[t] - θ|)

    FIXME: A function that can be directly merged.
    """

    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.)
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, L)
        return out.float()

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class PMSN_surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, gamma=1.):
        # tm=torch.arange(input.size(-1),device=input.device).repeat(input.size(0),input.size(1),1) * thresh + (2-1e-3) * thresh
        cum_x = input.cumsum(dim=-1)
        cum_x_shift = cum_x.clone()
        cum_x_shift[..., 1:] = cum_x[..., :-1]
        cum_x_shift[..., 0] = 0
        spike_shift = (cum_x_shift / thresh).floor().clamp(min=0)
        out = ((cum_x - spike_shift * thresh) / thresh).floor().clamp(min=0,max=1)
        L = torch.tensor([gamma])
        ctx.save_for_backward(thresh, cum_x - spike_shift * thresh, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (thresh, delta, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        #tmp = (1 / gamma) * (1 / gamma) * ((gamma - abs(delta-thresh)).clamp(min=0))  # triangle
        tmp = (gamma - abs(delta - thresh) > 0) * gamma  # rectangle
        grad_output = grad_input * tmp
        return grad_output, None