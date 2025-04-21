import torch
import torch.nn as nn


class Distrloss_layer(nn.Module):

    def __init__(self, thresh=1):
        super(Distrloss_layer, self).__init__()
        self.thresh = thresh

    def forward(self, input):
        if input.dim() != 5 and input.dim() != 3:
            raise ValueError('expected 5D or 3D input (got {}D input)'
                             .format(input.dim()))

        T, B, C = input.shape
        distrloss = (input.mean() - 0.5 * self.thresh) ** 2  # also can be changed to distrloss = (input.mean() - 0.5/T) ** 2

        return distrloss
