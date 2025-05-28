import torch
import torch.nn as nn
import copy
from typing import Any


class SLIF(nn.Module):
    """
        Serial LIF neuron with storable membrane potential intermediate states.
    """
    def __init__(
        self,
        neuron_rest   = 0.0,
        neuron_decay  = 0.5,
        neuron_thresh = 1.0,
        surro_func    = None,
        hard_reset    = False,
    ):
        super().__init__()
        self._state = {"membrane": neuron_rest,}
        self._state_default = {key: copy.deepcopy(value) for key, value in self._state.items()}

        self.neuron_rest   = neuron_rest
        self.neuron_decay  = torch.tensor(neuron_decay).float()
        self.neuron_thresh = neuron_thresh
        self.hard_reset    = hard_reset
        self.surro_func    = surro_func

    def __getattr__(self, name: str) -> Any:
        inner_state = self.__dict__["_state"]
        if name in inner_state:
            value = inner_state[name]
        else:
            value = super().__getattr__(name)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        inner_state = self.__dict__.get("_state")
        if inner_state is not None and name in inner_state.keys():
            inner_state[name] = value
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        return (
            f"neuron_rest={self.neuron_rest:.2f}, "
            f"neuron_decay={self.neuron_decay:.2f}, "
            f"neuron_thresh={self.neuron_thresh:.2f}, "
            f"hard_reset={self.hard_reset}, "
        )

    def reset(self):
        for key in self._state.keys():
            self._state[key] = copy.deepcopy(self._state_default[key])

    def forward(self, x: torch.Tensor):
        if isinstance(self.membrane, float):
            self.membrane = torch.full_like(x.data, self.membrane)
        self.membrane = self.membrane * self.neuron_decay + x
        spike = self.surro_func(self.membrane - self.neuron_thresh)
        if self.hard_reset:
            self.membrane = self.membrane * (1. - spike)
        else:
            self.membrane = self.membrane - spike * self.neuron_thresh
        return spike
