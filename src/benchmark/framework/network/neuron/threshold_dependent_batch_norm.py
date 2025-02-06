import torch
from torch.nn.modules.batchnorm import _BatchNorm


class ThresholdDependentBatchNorm1d(_BatchNorm):
    """
        Hanle Zheng \emph{et al.}, Going Deeper With Directly-Trained Larger Spiking Neural Networks, 2021.
    """
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.v_th = v_th
        torch.nn.init.constant_(self.weight, alpha * v_th)

    def _check_input_dim(self, input):
        assert input.dim() == 4 - 1 or input.dim() == 3 - 1  # [T * N, C, L]

    def forward(self, x_seq):
        y = x_seq.flatten(0, 1)
        y = super().forward(y)
        y_shape = [x_seq.shape[0], x_seq.shape[1]] + list(y.shape[1:])
        return y.view(y_shape)
