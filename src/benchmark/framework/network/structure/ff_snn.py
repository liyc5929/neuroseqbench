from torch.nn import Module, Linear


class SpikingNet(Module):
    def __init__(self,
        input_size,
        hidden_size,
        output_size,
        spiking_neuron,
        num_hidden_layers = 1,
        args = None
    ):
        super(SpikingNet, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.args = args
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * num_hidden_layers
        else:
            assert len(hidden_size) == num_hidden_layers


        for hidden_layer_i in range(num_hidden_layers):
            exec("self.fc" + str(hidden_layer_i) + " = Linear(in_features=input_size, out_features=hidden_size[hidden_layer_i])")

            if self.args.dataset in ['psmnist'] and hidden_layer_i == (num_hidden_layers - 1):
                exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[hidden_layer_i], recurrent=False)")
            else:
                exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[hidden_layer_i])")
            input_size = hidden_size[hidden_layer_i]
        self.classifier = Linear(in_features=input_size, out_features=output_size)

    def single_step_forward(self, x):
        for hidden_layer_i in range(self.num_hidden_layers):
            x = eval("self.fc" + str(hidden_layer_i))(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        output = self.multi_step_forward(x)
        return output

    def multi_step_forward(self, x):
        for hidden_layer_i in range(0, self.num_hidden_layers):
            x = eval("self.fc" + str(hidden_layer_i))(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        x = self.classifier(x)
        return x
