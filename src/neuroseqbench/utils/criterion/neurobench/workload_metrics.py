# -----------------------------------------------------------------------------
# This file is adapted from:
# Yik, J., Van den Berghe, K., den Blanken, D. et al. 
# The NeuroBench framework for benchmarking neuromorphic computing algorithms and systems.
# Nat Commun 16, 1545 (2025). https://doi.org/10.1038/s41467-025-56739-4
# -----------------------------------------------------------------------------


from abc import ABC, abstractmethod
from collections import defaultdict
import torch
from torch import Tensor

from .base import NeuroBenchModel
from .macs import single_layer_MACs


class WorkloadMetric(ABC):
    """
    Abstract base class for workload metrics.

    A workload metric is designed to evaluate some aspect of a model's performance
    or behavior, typically during the inference phase, based on its predictions
    and input data. This class defines the basic interface for all workload metrics
    that require computation over batches of data.

    Attributes:
        requires_hooks (bool): Flag indicating if the metric requires hooks for its computation.

    """

    def __init__(self, requires_hooks: bool = False):
        """
        Initialize the WorkloadMetric.

        Args:
            requires_hooks (bool, default=False): Flag indicating if the metric requires hooks

        """
        self._requires_hooks = requires_hooks

    @abstractmethod
    def __call__(
        self, model: NeuroBenchModel, preds: Tensor, data: tuple[Tensor, Tensor]
    ) -> float:
        """
        Compute the workload metric.

        This method must be implemented by any subclass to define how the metric
        should be computed based on the model, predictions, and data.

        Args:
            model (NeuroBenchModel): The model whose performance is being evaluated.
            preds (Tensor): A tensor of model predictions.
            data (tuple[Tensor, Tensor]): A tuple containing the input data (Tensor)
            and the true labels (Tensor).

        Returns:
            float: The computed value of the workload metric.

        """
        pass

    @property
    def requires_hooks(self) -> bool:
        """
        Property indicating whether the metric requires hooks.

        Returns:
            bool: True if the metric requires hooks, False otherwise.

        """
        return self._requires_hooks


class AccumulatedMetric(WorkloadMetric):
    """
    Abstract base class for accumulated workload metrics.

    An accumulated metric tracks values over multiple batches or iterations and computes
    the final metric value after accumulating data. It extends the WorkloadMetric class
    and adds functionality for resetting and computing the accumulated metric over time.

    """

    def __init__(self, requires_hooks: bool = False):
        """
        Initialize the AccumulatedMetric.

        Args:
            requires_hooks (bool, default=False): Flag indicating if the metric requires hooks

        """
        super().__init__(requires_hooks)

    @abstractmethod
    def compute(self) -> float:
        """
        Compute the accumulated metric.

        This method must be implemented by any subclass to compute the accumulated
        value of the metric, typically after processing multiple batches.

        Returns:
            float: The computed accumulated metric value.

        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the accumulated state.

        This method must be implemented by any subclass to reset the metric's
        accumulated state.

        """
        pass


class ActivationSparsity(WorkloadMetric):
    """
    Sparsity of model activations.

    Calculated as the number of zero activations over the total number of activations,
    over all layers, timesteps, samples in data.

    """

    def __init__(self):
        """Initialize the ActivationSparsity metric."""

        super().__init__(requires_hooks=True)

    def __call__(self, model, preds, data):
        """
        Compute activation sparsity.

        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
        Returns:
            float: Activation sparsity

        """
        # TODO: for a spiking model, based on number of spikes over all timesteps over all samples from all layers
        #       Standard FF ANN depends on activation function, ReLU can introduce sparsity.
        total_spike_num = 0  # Count of non-zero activations
        total_neuro_num = 0  # Count of all activations

        for hook in model.activation_hooks:
            # Skip layers with no outputs
            if not hook.activation_outputs:
                continue

            for (
                spikes
            ) in hook.activation_outputs:  # do we need a function rather than a member
                spike_num, neuro_num = torch.count_nonzero(spikes).item(), torch.numel(
                    spikes
                )
                total_spike_num += spike_num
                total_neuro_num += neuro_num

        # Compute sparsity
        if total_neuro_num == 0:  # Prevent division by zero
            return 0.0

        sparsity = (total_neuro_num - total_spike_num) / total_neuro_num
        return sparsity



class SynapticOperations(AccumulatedMetric):
    """
    Number of synaptic operations.

    This metric computes the number of Multiply-Accumulate operations (MACs) for
    Artificial Neural Networks (ANN) and Accumulation operations (ACs) for Spiking
    Neural Networks (SNN).

    """

    def __init__(self):
        """Initialize SynapticOperations metric."""

        super().__init__(requires_hooks=True)
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def reset(self):
        """
        Reset the metric state for a new evaluation.

        Clears all accumulated values for MAC, AC, synaptic operations, and the total
        number of samples.

        """
        self.MAC = 0
        self.AC = 0
        self.total_synops = 0
        self.total_samples = 0

    def __call__(self, model, preds, data):
        """
        Accumulate the Multiply-Accumulate (MAC) operations or Accumulation (AC)
        operations during the forward pass.

        This method accumulates the operations based on the model's connections, and differentiates between
        ANN (MACs) and SNN (ACs) operations based on the spiking activity.


        Args:
            model: A NeuroBenchModel.
            preds: A tensor of model predictions.
            data: A tuple of data and labels.
            inputs: A tensor of model inputs.
        Returns:
            float: Multiply-accumulates.

        """
        for hook in model.connection_hooks:
            inputs = hook.inputs  # copy of the inputs, delete hooks after
            for spikes in inputs:
                # spikes is batch, features, see snntorchmodel wrappper
                # for single_in in spikes:
                if len(spikes) == 1:
                    spikes = spikes[0]
                hook.hook.remove()
                operations, spiking = single_layer_MACs(spikes, hook.layer)
                total_ops, _ = single_layer_MACs(spikes, hook.layer, total=True)
                self.total_synops += total_ops
                if spiking:
                    self.AC += operations
                else:
                    self.MAC += operations
                hook.register_hook()
        # ops_per_sample = ops / data[0].size(0)
        self.total_samples += data[0].size(0)
        return self.compute()

    def compute(self):
        """
        Compute the average number of operations per sample.

        Returns:
            dict: A dictionary containing:

                - "Effective_MACs": The average MACs per sample.

                - "Effective_ACs": The average ACs per sample.

                - "Dense": The average total synaptic operations per sample.

        """

        if self.total_samples == 0:
            return {"Effective_MACs": 0, "Effective_ACs": 0, "Dense": 0}
        ac = self.AC / self.total_samples
        mac = self.MAC / self.total_samples
        total_synops = self.total_synops / self.total_samples
        return {"Effective_MACs": mac, "Effective_ACs": ac, "Dense": total_synops}
