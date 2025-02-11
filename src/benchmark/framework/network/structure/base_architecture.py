import inspect
import torch
from torch.nn import Module


class OutputConfigCheckMeta(type):
    def __new__(cls, name, bases, dct):
        cls_obj = super().__new__(cls, name, bases, dct)

        original_init = cls_obj.__init__ if "__init__" in dct else None
        def wrapped_init(self, *args, **kwargs):
            if original_init:
                original_init(self, *args, **kwargs)

            # Post-initialization check
            if not hasattr(self, "_output_config") or self._output_config is None:
                raise RuntimeError(
                    f"Initialization of class `{self.__class__.__name__}` is incomplete: the `_output_config` attribute is missing. "
                    f"Please ensure `__init__` method of this class is called correctly."
                )
        cls_obj.__init__ = wrapped_init 

        return cls_obj

class OutputEnforcementMeta(type):
    def __new__(cls, name, bases, dct):
        cls_obj = super().__new__(cls, name, bases, dct)

        # Check if `get_output` is called at least once in `forward`
        forward_method = dct.get("forward", None)
        if forward_method is not None:
            source_lines = inspect.getsource(forward_method).strip().split("\n")
            # Check if `get_output` is called at least once
            called_get_output = any("self.get_output" in line for line in source_lines)
            if not called_get_output:
                raise RuntimeError(f"The `get_output` must be called in `forward` function of the class `{cls_obj.__name__}`.")
        return cls_obj        


class BaseArchitecture(Module, metaclass=type("", (OutputConfigCheckMeta, OutputEnforcementMeta), {})):
    def __init__(self, output_mode: str="sum_time_steps"):
        super(BaseArchitecture, self).__init__()
        self.output_mode    = output_mode 
        self._output_config = {
            "default"       : self._get_sum_time_steps,
            "sum_time_steps": self._get_sum_time_steps,
            "last_time_step": self._get_last_time_step,
            "all_time_steps": self._get_all_time_steps,
        }

    def get_output(self, output: torch.Tensor):
        output_proc = self._output_config.get(self.output_mode)
        if output_proc is not None:
            return output_proc(output)
        else:
            raise ValueError("Invalid `output_mode`.")

    def _get_sum_time_steps(self, output: torch.Tensor):
        return output.sum(axis=0)

    def _get_last_time_step(self, output: torch.Tensor):
        return output[-1]

    def _get_all_time_steps(self, output: torch.Tensor):
        return output
