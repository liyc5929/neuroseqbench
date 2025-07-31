# -----------------------------------------------------------------------------
# This file is adapted from:
# Yik, J., Van den Berghe, K., den Blanken, D. et al. 
# The NeuroBench framework for benchmarking neuromorphic computing algorithms and systems.
# Nat Commun 16, 1545 (2025). https://doi.org/10.1038/s41467-025-56739-4
# -----------------------------------------------------------------------------


from abc import ABC, abstractmethod
import torch
from .base import STATELESS_LAYERS, RECURRENT_LAYERS, RECURRENT_CELLS, NeuroBenchModel


class StaticMetric(ABC):
    """
    Abstract base class for static metrics.

    A static metric computes a value based on the model, and is designed to operate
    without the need for batch processing or iterative accumulation over multiple
    samples. This metric is expected to return a single value when called, which is
    typically computed over the entire model.

    """

    @abstractmethod
    def __call__(self, model: NeuroBenchModel) -> float:
        """
        Compute the static metric for the given model.

        This abstract method must be implemented by any subclass to define
        how the metric should be computed based on the model.

        Args:
            model (NeuroBenchModel): The model whose performance or properties the metric evaluates.

        Returns:
            float: The computed value of the metric for the model.

        Notes:
            - The method should return a single float value representing the
              result of the metric computation. It does not require batch
              processing or tracking multiple samples.
            - This is intended for metrics that compute a single, static
              evaluation of a model, such as those based on its architecture,
              weights, or overall performance.

        """
        pass


class ParameterCount(StaticMetric):
    """A metric that counts the number of parameters in a model."""

    def __call__(self, model):
        """
        Count the number of parameters in a model.

        Args:
            model: A NeuroBenchModel.
        Returns:
            int: Number of parameters in the model.

        """
        return sum(p.numel() for p in model.__net__().parameters())


class ConnectionSparsity(StaticMetric):
    """
    Sparsity of model connections between layers.

    Based on number of zeros
    in supported layers, other layers are not taken into account in the computation:
    Supported layers:
    Linear
    Conv1d, Conv2d, Conv3d
    RNN, RNNBase, RNNCell
    LSTM, LSTMBase, LSTMCell
    GRU, GRUBase, GRUCell

    """

    def __call__(self, model):
        """
        Compute connection sparsity.

        Args:
            model: A NeuroBenchModel.
        Returns:
            float: Connection sparsity, rounded to 3 decimals.

        """

        def get_nr_zeros_weights(module):
            """
            Get the number of zeros in a module's weights.

            Args:
                module: A torch.nn.Module.
            Returns:
                int: Number of zeros in the module's weights.

            """
            children = list(module.children())
            if len(children) == 0:  # it is a leaf
                # print(module)
                if isinstance(module, STATELESS_LAYERS):
                    count_zeros = torch.sum(module.weight == 0)
                    count_weights = module.weight.numel()
                    return count_zeros, count_weights

                elif isinstance(module, RECURRENT_LAYERS):
                    attribute_names = []
                    for i in range(module.num_layers):
                        param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                        if module.bias:
                            param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                        if module.proj_size > 0:  # it is lstm
                            param_names += ["weight_hr_l{}{}"]

                        attribute_names += [x.format(i, "") for x in param_names]
                        if module.bidirectional:
                            suffix = "_reverse"
                            attribute_names += [
                                x.format(i, suffix) for x in param_names
                            ]

                    count_zeros = 0
                    count_weights = 0
                    for attr in attribute_names:
                        attr_val = getattr(module, attr)
                        count_zeros += torch.sum(attr_val == 0)
                        count_weights += attr_val.numel()

                    return count_zeros, count_weights

                elif isinstance(module, RECURRENT_CELLS):
                    attribute_names = ["weight_ih", "weight_hh"]
                    if module.bias:
                        attribute_names += ["bias_ih", "bias_hh"]

                    count_zeros = 0
                    count_weights = 0
                    for attr in attribute_names:
                        attr_val = getattr(module, attr)
                        count_zeros += torch.sum(attr_val == 0)
                        count_weights += attr_val.numel()

                    return count_zeros, count_weights

                else:
                    # print('Module type: ', module, 'not found.')
                    return 0, 0

            else:
                count_zeros = 0
                count_weights = 0
                for child in children:
                    child_zeros, child_weights = get_nr_zeros_weights(child)
                    count_zeros += child_zeros
                    count_weights += child_weights
                return count_zeros, count_weights

        # Pull the layers from the model's network
        layers = model.__net__().children()
        # For each layer, count where the weights are zero
        count_zeros = 0
        count_weights = 0
        for module in layers:
            zeros, weights = get_nr_zeros_weights(module)
            count_zeros += zeros
            count_weights += weights

        # Return the ratio of zeros to weights, rounded to 4 decimals
        return round((count_zeros / count_weights).item(), 4)


class Footprint(StaticMetric):
    """A metric that counts the memory footprint of a model."""

    def __call__(self, model):
        """
        Count the memory footprint of a model.

        Args:
            model: A NeuroBenchModel.
        Returns:
            float: Memory footprint of the model.

        """
        param_size = 0
        for param in model.__net__().parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buffer in model.__net__().buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        return param_size + buffer_size
