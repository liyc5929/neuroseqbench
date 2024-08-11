from torch.nn import Module


class MergeDimension(Module):
    def __init__(self):
        super(MergeDimension, self).__init__()

    def forward(self, x):
        if x.dim() == 5 or x.dim() == 3:
            return x.reshape(-1, *x.shape[2:])
        return x


class SplitDimension(Module):
    def __init__(self, time_step):
        super(SplitDimension, self).__init__()
        self.time_step = time_step

    def forward(self, x):
        if x.dim() == 4 or x.dim() == 2:
            return x.reshape(self.time_step, x.shape[0] // self.time_step, *x.shape[1:])
        return x
