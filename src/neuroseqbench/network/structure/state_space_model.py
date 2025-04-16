"""
According to: Matei-Ioan Stan~\emph{et al.}, Learning long sequences in spiking neural networks, 2024.
"""
import torch.nn as nn
from . import MergeDimension, SplitDimension


class SSM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, pool=False, dataset = None, spiking_neuron=None, loss=None, neuron_type=None):
        super(SSM, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.flatten = nn.Flatten()
        self.pool = pool
        self.neuron = neuron_type
        self.dataset = dataset
        if self.pool:
            self.max_pool = nn.MaxPool2d(4, 4)
        for hidden_layer_i in range(num_hidden_layers):
            exec("self.spk" + str(
                hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[{0}])".format(hidden_layer_i))
            if hidden_layer_i == 0:
                exec("self.fc" + str(
                    hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size[{0}])".format(
                    hidden_layer_i))
            input_size = hidden_size[hidden_layer_i]
        self.classifier = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x, time_step=None, multi_step=False):
        if time_step is None:
            time_step = x.size(0)
        x = MergeDimension()(x)
        if self.pool:
            x = self.max_pool(x)
        x = self.flatten(x)
        x = SplitDimension(time_step)(x)
        for hidden_layer_i in range(self.num_hidden_layers):
            if hidden_layer_i == 0:
                x = MergeDimension()(x)
                x = eval("self.fc" + str(hidden_layer_i))(x)
                x = SplitDimension(time_step)(x)
            else:
                x = x
            x = eval("self.spk" + str(hidden_layer_i))(x)
        time_step = x.size(0)  # in case the membrane potential output only have T=1
        x = MergeDimension()(x)
        x = self.classifier(x)
        output = SplitDimension(time_step)(x)
        if self.dataset in ['add', 'biadd', 'EEG']: # laststep decision
            output=output[-1, ...].unsqueeze(0)
        return output
