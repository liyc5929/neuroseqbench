import torch
import torch.nn as nn
import numpy as np
import math


class MutiStepNoisyRateScheduler:

    def __init__(self, init_p=1, reduce_ratio=0.9, milestones=[0.3, 0.7, 0.9, 0.95], num_epoch=100, start_epoch=0):
        self.reduce_ratio = reduce_ratio
        self.p = init_p
        self.milestones = [int(m * num_epoch) for m in milestones]
        self.num_epoch = num_epoch
        self.start_epoch = start_epoch

    def set_noisy_rate(self, p, model):
        for m in model.modules():
            if isinstance(m, EfficientNoisySpike):
                m.p = p

    def __call__(self, epoch, model):
        for one in self.milestones:
            if one + self.start_epoch == epoch:
                self.p *= self.reduce_ratio
                print('change noise rate as ' + str(self.p))
                self.set_noisy_rate(self.p, model)
                break

def get_temperatures(net):
    temperatures = []
    for m in net.modules():
        if isinstance(m, EfficientNoisySpike):
            temperatures.append(m.inv_sg.get_temperature())
    temperatures = torch.cat(temperatures).cpu()
    return temperatures


class InvSigmoid(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=False):
        super(InvSigmoid, self).__init__()
        self.learnable = learnable
        self.alpha = alpha

    def get_temperature(self):
        return self.alpha.detach().clone()

    def forward(self, x):
        if self.learnable and not isinstance(self.alpha, nn.Parameter):
            self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device))
        return torch.sigmoid((1/self.alpha) * x) # in original setting, a= 1/alpha.


class InvRectangle(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=False):
        super(InvRectangle, self).__init__()
        self.learnable = learnable
        self.alpha = alpha

    def get_temperature(self):
        return self.alpha.detach().clone()

    def forward(self, x):
        if self.learnable and not isinstance(self.alpha, nn.Parameter):
            self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device))
        return torch.clamp(x + 0.5 * self.alpha, 0, 1.0 * self.alpha)



class EfficientNoisySpike(nn.Module):
    """
    ASGL https://github.com/Windere/ASGL-SNN
    """
    def __init__(self, inv_sg=InvRectangle(), p=0.1, spike=True):
        super(EfficientNoisySpike, self).__init__()
        self.inv_sg = inv_sg
        self.p = p
        self.reset_mask()

    def create_mask(self, x: torch.Tensor):
        return torch.bernoulli(torch.ones_like(x) * (1 - self.p))

    def forward(self, x):

        sigx = self.inv_sg(x)
        if self.training:
            if self.mask is None:
                self.mask = self.create_mask(x)
            return sigx + (((x >= 0).float() - sigx) * self.mask).detach()
        return (x >= 0).float()

    def reset_mask(self):
        self.mask = None
