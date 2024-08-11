import inspect
from torch.nn import Module


class BaseNeuron(Module):
    def __init__(self, exec_mode: str="serial"):
        super(BaseNeuron, self).__init__()
        self.exec_mode    = exec_mode
        self._exec_config = {
            "default": self._serial_process,
            "serial" : self._serial_process,
            "fused"  : self._temporal_fused_process,
        }

    def forward(self, tx, v=None):
        execution_proc = self._exec_config.get(self.exec_mode)
        if execution_proc is not None:
            return execution_proc(tx, v)
        else:
            raise ValueError("Invalid `execution_mode`.")

    def _serial_process(self, _):
        raise NotImplementedError(f"The `{inspect.currentframe().f_code.co_name}` method of the subclass `{type(self).__name__}` needs to be implemented.")

    def _temporal_fused_process(self, _):
        raise NotImplementedError(f"The `{inspect.currentframe().f_code.co_name}` method of the subclass `{type(self).__name__}` needs to be implemented.")
