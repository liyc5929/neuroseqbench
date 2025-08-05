# -----------------------------------------------------------------------------
# This file is adapted from:
# Yik, J., Van den Berghe, K., den Blanken, D. et al. 
# The NeuroBench framework for benchmarking neuromorphic computing algorithms and systems.
# Nat Commun 16, 1545 (2025). https://doi.org/10.1038/s41467-025-56739-4
# -----------------------------------------------------------------------------


from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple
from ....network.neuron import SpikeGeneration, PMSN_SpikeGeneration


class NeuronHook(ABC):
    """
    Hook class for an activation layer in a NeuroBenchModel.

    Output of the activation layer in each forward pass will be stored.

    """

    def __init__(self, layer, name=None):
        """
        Initializes the class.

        A forward hook is registered for the activation layer.

        Args:
            layer: The activation layer which is a PyTorch nn.Module.

        """
        self.activation_outputs = []
        self.activation_inputs = []
        self.pre_fire_mem_potential = []
        self.post_fire_mem_potential = []
        self.name = name
        if layer is not None:
            self.hook = layer.register_forward_hook(self.hook_fn)
            self.hook_pre = layer.register_forward_pre_hook(self.pre_hook_fn)
        else:
            self.hook = None
            self.hook_pre = None

        self.layer = layer  # the activation layer
        self.spiking = False

    def pre_hook_fn(self, layer, input):
        """
        Hook function that will be called before each forward pass of the activation
        layer.

        Each input of the activation layer will be stored.

        Args:
            layer: The registered layer
            input: Input of the registered layer

        """
        self.activation_inputs.append(input)
        if self.spiking and hasattr(layer, "mem"):
            self.pre_fire_mem_potential.append(layer.mem)

    def hook_fn(self, layer, input, output):
        """
        Hook function that will be called after each forward pass of the activation
        layer.

        Each output of the activation layer will be stored.

        Args:
            layer: The registered layer
            input: Input of the registered layer
            output: Output of the registered layer

        """
        if self.spiking:
            self.activation_outputs.append(output[0])
            if hasattr(layer, "mem"):
                self.post_fire_mem_potential.append(layer.mem)

        else:
            self.activation_outputs.append(output)

    # def empty_hook(self):
    #     """Deletes the contents of the hooks, but keeps the hook registered."""
    #     self.activation_outputs = []
    #     self.activation_inputs = []

    def reset(self):
        """Resets the stored activation outputs and inputs."""
        self.activation_outputs.clear()
        self.activation_inputs.clear()
        self.pre_fire_mem_potential.clear()
        self.post_fire_mem_potential.clear()

    def close(self):
        """Remove the registered hook."""
        if self.hook:
            self.hook.remove()
        if self.hook_pre:
            self.hook_pre.remove()


class LayerHook(ABC):
    def __init__(self, layer) -> None:
        self.layer = layer
        self.inputs = []
        if layer is not None:
            self.hook = layer.register_forward_pre_hook(self.hook_fn)
        else:
            self.hook = None

    def hook_fn(self, module, input):
        self.inputs.append(input)

    def register_hook(self):
        self.hook = self.layer.register_forward_pre_hook(self.hook_fn)

    def reset(self):
        self.inputs.clear()

    def close(self):
        if self.hook:
            self.hook.remove()


SUPPORTED_ACTIVATIONS = (
    nn.ReLU,
    nn.Sigmoid,
)


STATELESS_LAYERS = (
    nn.Linear,
    nn.Conv2d,
    nn.Conv1d,
    nn.Conv3d,
)

# for neuroseqbench
SPIKING_NEURONS = (
    SpikeGeneration,
    PMSN_SpikeGeneration
)

RECURRENT_CELLS = (nn.RNNCellBase,)
RECURRENT_LAYERS = (nn.RNNBase,)
SUPPORTED_LAYERS = STATELESS_LAYERS + RECURRENT_LAYERS + RECURRENT_CELLS


class NeuroBenchModel(ABC):
    """
    Abstract class for NeuroBench models.

    Individual model frameworks are responsible for defining model inference.

    """

    def __init__(self):
        """
        Init using a trained network.

        Args:
            net: A trained network

        """
        # self.activation_modules = list(SUPPORTED_ACTIVATIONS)
        self.activation_modules = list(SPIKING_NEURONS)
        self.activation_hooks = []
        self.connection_hooks = []
        self.first_layer = None

        # self.supported_layers = (
        #     nn.Linear,
        #     nn.Conv2d,
        #     nn.Conv1d,
        #     nn.Conv3d,
        #     nn.RNNBase,
        #     nn.RNNCellBase,
        # )

    @abstractmethod
    def __call__(self, batch):
        """
        Includes the whole pipeline from data to inference (output should be same format
        as targets).

        Args:
            batch: A batch of data to run inference on

        """
        pass

    @abstractmethod
    def __net__(self):
        """Returns the underlying network."""
        pass

    def set_first_layer(self, layer):
        """Sets the first layer of the network."""
        self.first_layer = layer

    def add_activation_module(self, activaton_module):
        """Add a cutomized activaton_module that can be detected after running the
        preprocessing pipeline for detecting activation functions."""
        self.activation_modules.append(activaton_module)

    def activation_layers(self):
        """
        Retrieve all activation layers in the network, including spiking neurons.

        Returns:
            list: Activation layers.

        """

        def is_activation_layer(module):
            """Check if a module is an activation layer."""
            return any(
                isinstance(module, act_mod) for act_mod in self.activation_modules
            )

        def find_activation_layers(module):
            """Recursively find activation layers in a module."""
            layers = []
            for child_name, child in module.named_children():
                if is_activation_layer(child):
                    layers.append({"layer_name": child_name, "layer": child})
                elif list(child.children()):  # Check for nested submodules
                    layers.extend(find_activation_layers(child))
            return layers

        return find_activation_layers(self.__net__())

    def connection_layers(self):
        """
        Retrieve all connection layers in the network.

        Connection layers include Linear, Conv, and RNN-based layers.

        Returns:
            list: Connection layers.

        """

        def find_connection_layers(module):
            """Recursively find connection layers in a module."""
            layers = []
            for child in module.children():
                if isinstance(child, SUPPORTED_LAYERS):
                    layers.append(child)
                elif list(child.children()):  # Check for nested submodules
                    layers.extend(find_connection_layers(child))
            return layers

        return find_connection_layers(self.__net__())

    def reset_hooks(self):
        """Resets all the hooks (activation hooks and connection hooks) in parallel."""
        for hook in self.activation_hooks + self.connection_hooks:
            hook.reset()

    def close_hooks(self):
        """Closes all the hooks (activation hooks and connection hooks)"""
        for hook in self.activation_hooks + self.connection_hooks:
            hook.close()

    def cleanup_hooks(self):
        """Closes all the hooks (activation hooks and connection hooks)"""
        for hook in self.activation_hooks + self.connection_hooks:
            hook.reset()
            hook.close()

        self.activation_hooks.clear()
        self.connection_hooks.clear()

    def register_hooks(self):
        """Registers hooks for the model."""

        # Registered activation hooks
        for layer in self.activation_layers():
            layer_name = layer["layer_name"]
            layer = layer["layer"]
            self.activation_hooks.append(NeuronHook(layer, layer_name))

        for layer in self.connection_layers():
            self.connection_hooks.append(LayerHook(layer))


from .managers import StaticMetricManager, WorkloadMetricManager
from .static_metrics import ParameterCount, ConnectionSparsity, Footprint
from .workload_metrics import ActivationSparsity, MembraneUpdates, SynapticOperations


class _WrappedNeuroBenchModel(NeuroBenchModel):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def __call__(self, batch):
        # Assume batch is (input, target) tuple
        x, _ = batch
        return self.model(x)

    def __net__(self):
        return self.model  # return the actual `nn.Module` for hook detection


def setup_neurobench_metrics(model: torch.nn.Module) -> Tuple[NeuroBenchModel, StaticMetricManager, WorkloadMetricManager]:
    # Static metrics
    static_metrics = [
        ParameterCount,
        ConnectionSparsity,
        Footprint,
    ]
    static_mgr = StaticMetricManager(static_metrics)

    # Workload metrics
    workload_metrics = [
        ActivationSparsity,
        MembraneUpdates,
        SynapticOperations,
    ]

    wrapped_model = _WrappedNeuroBenchModel(model)
    workload_mgr = WorkloadMetricManager(workload_metrics)
    workload_mgr.register_hooks(wrapped_model)

    return wrapped_model, static_mgr, workload_mgr
