import torch
import torch.nn as nn


class _BatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: torch.Tensor):
        if x.dim() != 4 and x.dim() != 3:
            raise ValueError(f'expected x with shape [T, N, C, L] or [T, N, C], but got x with shape {x.shape}!')
        x_seq = x
        stateless_module = super().forward
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y = x_seq.flatten(0, 1)
        if isinstance(stateless_module, (list, tuple, nn.Sequential)):
            for m in stateless_module:
                y = m(y)
        else:
            y = stateless_module(y)
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class TemporalEffectiveBatchNorm1d(nn.Module):
    """
        Chaoteng Duan \emph{et al.}, Temporal Effective Batch Normalization in Spiking Neural Networks, 2022.
    """
    def __init__(
        self,
        T: int,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.bn = _BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.scale = nn.Parameter(torch.ones([T]))

    def forward(self, x_seq: torch.Tensor): # x.shape = [T, B, N]
        return self.bn(x_seq) * self.scale[:x_seq.size(0)].view(-1, 1, 1)
