from torch.nn import Module, Sequential


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


class ANNSequential(Sequential):
    def __init__(self, *args: Module):
        super(ANNSequential, self).__init__(*args)

    def forward(self, x):
        time_step = x.shape[0]
        x = MergeDimension()(x)
        x = super().forward(x)
        x = SplitDimension(time_step)(x)
        return x


class Permute(Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
