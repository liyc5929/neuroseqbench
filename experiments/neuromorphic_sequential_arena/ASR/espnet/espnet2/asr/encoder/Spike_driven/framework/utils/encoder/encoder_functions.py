import torch


def rate_coding(x, time_step, lower_bound: int = 0, upper_bound: int = 1):
    coded_list = [
        (x > ((upper_bound - lower_bound) * torch.rand(x.shape) + lower_bound)).float()
        for _ in range(time_step)
    ]
    return torch.stack(coded_list)
