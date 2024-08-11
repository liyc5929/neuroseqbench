
def _spatio_temporal_backpropagation_based_hard_update(x, v, y, rest, decay, *_, **__):
    return decay * v * (1.0 - y) + rest * y + x


def _spatio_temporal_backpropagation_based_soft_update(x, v, y, rest, decay, threshold, *_, **__):
    return decay * v + (rest - threshold) * y + x


def _spatio_domain_backpropagation_based_hard_update(x, v, y, rest, decay, *_, **__):
    return decay * v.detach() * (1.0 - y.detach()) + rest * y.detach() + x


def _spatio_domain_backpropagation_based_soft_update(x, v, y, rest, decay, threshold, *_, **__):
    return decay * v.detach() + (rest - threshold) * y.detach() + x


def _no_temporal_dimention_update(x, *_, **__):
    return x


__func_name__ = {
    "STBP_hard": _spatio_temporal_backpropagation_based_hard_update,
    "STBP_soft": _spatio_temporal_backpropagation_based_soft_update,
    "SDBP_soft": _spatio_domain_backpropagation_based_hard_update,
    "SDBP_hard": _spatio_domain_backpropagation_based_soft_update,
    "noTD_soft": _no_temporal_dimention_update,
    "noTD_hard": _no_temporal_dimention_update,
}


class MembraneUpdate:
    def __init__(self, prop_mode: str, reset_mode, *args, **kwargs):
        self.prop_mode  = prop_mode
        self.reset_mode = reset_mode
        self.args       = args
        self.kwargs     = kwargs

    def __call__(self, x, v, y, rest, decay, threshold, *args, **kwargs):
        mem_update_func = __func_name__.get(f"{self.prop_mode}_{self.reset_mode}")
        if mem_update_func is not None:
            return mem_update_func(x, v, y, rest, decay, threshold, *(self.args + args), **{**self.kwargs, **kwargs})
        else:
            raise ValueError("Invalid membrane update strategy.")
