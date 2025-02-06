"""
According to: Ilyass Hammouamri \emph{et al.}, Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings, 2023.
"""

import torch
from torch.nn import Sequential, BatchNorm1d, ConstantPad1d
from DCLS.construct.modules import Dcls1d

from ..neuron import NonSpikingLIF
from . import BaseArchitecture, Permute, ANNSequential


class DCLS_Delays(BaseArchitecture):
    def __init__(self, 
        input_size, 
        hidden_size,
        output_size, 
        time_step,
        max_delay,
        output_mode,
        spiking_neuron = None,
    ):
        super().__init__(output_mode)
        # Register hyperparameters
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.output_size    = output_size
        self.time_step      = time_step
        self.max_delay      = max_delay
        self.left_padding   = self.max_delay - 1
        self.right_padding  = (self.max_delay - 1) // 2
        self.spiking_neuron = spiking_neuron
        self.final_neuron   = NonSpikingLIF(
            decay      = self.spiking_neuron.decay,
            time_step  = self.spiking_neuron.time_step,
            exec_mode  = self.spiking_neuron.exec_mode,
        )
        # Initialize model
        self.features = Sequential(
            # Calculate delays
            Permute(1, 2, 0),
            ConstantPad1d(padding=(self.left_padding, self.right_padding), value=0),
            Dcls1d(self.input_size, self.hidden_size, kernel_count=1, groups=1, dilated_kernel_size=self.max_delay, bias=False, version="gauss"),
            Permute(2, 0, 1),
            # Calculate neurons
            ANNSequential(BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),),
            self.spiking_neuron,

            # Hidden 1
            # Calculate delays
            Permute(1, 2, 0),
            ConstantPad1d(padding=(self.left_padding, self.right_padding), value=0),
            Dcls1d(self.hidden_size, self.hidden_size, kernel_count=1, groups=1, dilated_kernel_size=self.max_delay, bias=False, version="gauss"),
            Permute(2, 0, 1),
            # Calculate neurons
            ANNSequential(BatchNorm1d(self.hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),),
            self.spiking_neuron,

            # Calculate delays
            Permute(1, 2, 0),
            ConstantPad1d(padding=(self.left_padding, self.right_padding), value=0),
            Dcls1d(self.hidden_size, self.output_size, kernel_count=1, groups=1, dilated_kernel_size=self.max_delay, bias=False, version="gauss"),
            Permute(2, 0, 1),
            # Calculate neurons
            self.final_neuron,
        )

    def forward(self, tx: torch.Tensor): # (T, B, C)
        ty = self.features(tx)
        return self.get_output(ty) 
