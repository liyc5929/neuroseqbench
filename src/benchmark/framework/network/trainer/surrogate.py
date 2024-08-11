import torch
import math


def _rectangular_function(v, threshold, a, b = 1., *_, **__):
    """
        Yujie Wu \emph{et al.}, Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks, 2018.
        $$ h(u) = \frac{1}{a} \cdot \mathrm{sign}(|u - V_\text{th}| < \frac{a}{2}) $$
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
    #grad_v = (1 / a) * torch.exp((v - threshold)/a) / ((1 + torch.exp((v - threshold)/a)) * (1 + torch.exp((v - threshold) / a)))
    return grad_v

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

def _multigaussian_function(v, threshold, a, *_, **__):
    """
        Revised from https://github.com/byin-cwi/Efficient-spiking-networks
    """
    input = v-threshold
    scale = 6.0
    hight = .15
    a = a / 2  # According to the paper, a = 0.5 in multigauss equal to a = 1 in triangle.
    temp = gaussian(input, mu=0., sigma=a) * (1. + hight)  - gaussian(input, mu=a, sigma=scale * a) * hight - gaussian(input, mu=-a, sigma=scale * a) * hight
    grad_v = temp.float() * a
    return grad_v


__func_config__ = {
    "rectangle": _rectangular_function,
    "triangle" : _triangle_function,
    "multigauss" : _multigaussian_function,
    "sigmoid"  : _sigmoid_function,
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
