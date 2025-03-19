import torch
import torch.nn as nn
import math
from functools import partial
from torch import Tensor
from typing import Callable

from src.benchmark.framework.network.architecture import MergeDimension, SplitDimension
from src.benchmark.framework.utils.tools import reset_states
from src.benchmark.framework.utils.criterion.IM_loss import Distrloss_layer
from src.benchmark.framework.network.architecture.layer import BatchNorm1d, ThresholdDependentBatchNorm1d, \
    TemporalEffectiveBatchNorm1d, Dropout
from src.benchmark.framework.network.architecture.tcn import TemporalConvNet
from src.benchmark.framework.network.architecture.rnn import script_lstm


class FFSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, pool=False, dataset=None,
                 spiking_neuron=None, loss=None, neuron_type=None):
        super(FFSNN, self).__init__()
        self.IM_Loss = (loss == 'IM')
        self.ASGL_Loss = (loss == 'ASGL')
        self.num_hidden_layers = num_hidden_layers
        self.flatten = nn.Flatten()
        self.pool = pool
        self.neuron = neuron_type
        self.out_final_step = (dataset == 'add')
        if self.pool:
            self.max_pool = nn.MaxPool2d(4, 4)
        if self.IM_Loss:
            self.distrloss_layers = []
        for hidden_layer_i in range(num_hidden_layers):
            if self.neuron == 'dhsnn':
                if dataset == 'psmnist' and hidden_layer_i == (num_hidden_layers - 1):
                    exec("self.spk" + str(
                        hidden_layer_i) + " = spiking_neuron(input_features=input_size, neuron_num=hidden_size[{0}], recurrent=False)".format
                         (hidden_layer_i))
                else:
                    exec("self.spk" + str(
                        hidden_layer_i) + " = spiking_neuron(input_features=input_size, neuron_num=hidden_size[{0}])".format
                         (hidden_layer_i))

            else:
                exec("self.fc" + str(
                    hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size[{0}])".format(
                    hidden_layer_i))
                if dataset == 'psmnist' and hidden_layer_i == (num_hidden_layers - 1):
                    exec("self.spk" + str(
                        hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[{0}], recurrent=False)".format(
                        hidden_layer_i))
                else:
                    exec("self.spk" + str(hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[{0}])".format(
                        hidden_layer_i))
            input_size = hidden_size[hidden_layer_i]
            if self.IM_Loss:
                self.distrloss_layers.append(Distrloss_layer(thresh=0.5))
        self.classifier = nn.Linear(in_features=input_size, out_features=output_size)
        if self.neuron == 'celif':
            self.TE = nn.Parameter(torch.zeros(max(hidden_size), self.spk0.time_step))
            nn.init.normal_(self.TE, 0.01, 0.01)
            for hidden_layer_i in range(num_hidden_layers):
                exec("self.spk" + str(hidden_layer_i) + " .TE = self.TE".format(hidden_layer_i))

    def single_step_forward(self, x):
        if self.pool:
            x = self.max_pool(x)
        x = self.flatten(x)
        for hidden_layer_i in range(self.num_hidden_layers):
            x = eval("self.fc" + str(hidden_layer_i))(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        x = self.classifier(x)
        return x

    def forward(self, x, time_step=None, multi_step=False):
        self.loss = []
        if time_step is None:
            time_step = x.size(0)
        if multi_step:
            if self.neuron == 'celif':
                output = self.multi_step_forward(x, time_step)
            else:
                output = self.multi_step_forward(x, time_step)
        else:
            reset_states(self)
            output = []
            for t in range(time_step):
                single_step_output = self.single_step_forward(x[t])
                output.append(single_step_output)
            output = torch.stack(output)

        if self.out_final_step:  # for add problem
            output = output[-1, ...].unsqueeze(0)
        if self.IM_Loss:
            disloss = (sum([ele for ele in self.loss])) / len(self.loss)
            return output, disloss
        else:
            return output

    def multi_step_forward(self, x, time_step):
        x = MergeDimension()(x)
        if self.pool:
            x = self.max_pool(x)
        x = self.flatten(x)
        x = SplitDimension(time_step)(x)
        for hidden_layer_i in range(self.num_hidden_layers):
            if self.neuron == 'dhsnn':
                x = x
            else:
                x = MergeDimension()(x)
                x = eval("self.fc" + str(hidden_layer_i))(x)
                x = SplitDimension(time_step)(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
            if self.IM_Loss:
                self.loss.append(self.distrloss_layers[hidden_layer_i](x))

        x = MergeDimension()(x)
        x = self.classifier(x)
        x = SplitDimension(time_step)(x)

        return x


class SpikingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, spiking_neuron=None, bn=None,
                 recurrent=False, args=None, vocab_size=None, dropout_embedding=0., pad_idx=None, dropout=0.0):
        super(SpikingNet, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.args = args
        self.neuron_type = args.neuron
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * num_hidden_layers
        else:
            assert len(hidden_size) == num_hidden_layers

        self.dropout_embedding = dropout_embedding
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=pad_idx)
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        else:
            self.embedding = None

        if bn == 'bn':
            self.bns = nn.ModuleList([BatchNorm1d(hidden_size[l]) for l in range(num_hidden_layers)])
        elif bn == 'ln':
            self.bns = nn.ModuleList([nn.LayerNorm(hidden_size[l]) for l in range(num_hidden_layers)])
        elif bn == 'tdbn':
            self.bns = nn.ModuleList([ThresholdDependentBatchNorm1d(alpha=1., v_th=spiking_neuron().threshold, num_features=hidden_size[l])
                   for l in range(num_hidden_layers)])
        elif bn == 'tebn':
            self.bns = nn.ModuleList([TemporalEffectiveBatchNorm1d(T=spiking_neuron().time_step, num_features=hidden_size[l]) for l in
                   range(num_hidden_layers)])
        elif bn is None:
            self.bns = None
        else:
            raise NotImplementedError

        for hidden_layer_i in range(num_hidden_layers):
            exec("self.fc" + str(
                hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size[hidden_layer_i])")
            exec("self.dropout" + str(
                hidden_layer_i) + " = Dropout(dropout)")

            if self.args.dataset in ['psmnist', 'binadd', 'dvslip'] and hidden_layer_i == (num_hidden_layers - 1):
                exec("self.spk" + str(
                    hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[hidden_layer_i], recurrent=False)")
            else:
                exec("self.spk" + str(
                    hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size[hidden_layer_i])")
            input_size = hidden_size[hidden_layer_i]
        self.classifier = nn.Linear(in_features=input_size, out_features=output_size)
        if self.neuron_type == 'celif':
            self.TE = nn.Parameter(torch.zeros(max(hidden_size),self.spk0.time_step))
            nn.init.normal_(self.TE, 0.01, 0.01)
            for hidden_layer_i in range(num_hidden_layers):
                exec("self.spk" + str(hidden_layer_i) + " .TE = self.TE".format(hidden_layer_i))

    def single_step_forward(self, x):
        for hidden_layer_i in range(self.num_hidden_layers):
            x = eval("self.fc" + str(hidden_layer_i))(x)
            if self.bns is not None:
                x = self.bns[hidden_layer_i](x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        x = self.classifier(x)
        return x

    def forward(self, x, time_step=None, multi_step=True):
        if time_step is None:
            time_step = x.size(0)
        x = x.view(x.size(0), x.size(1), -1)
        if multi_step:
            reset_states(self)
            output = self.multi_step_forward(x, time_step)
        else:
            reset_states(self)
            output = []
            for t in range(time_step):
                single_step_output = self.single_step_forward(x[t])
                output.append(single_step_output)
            output = torch.stack(output)
        return output

    def multi_step_forward(self, x, time_step):
        if self.embedding is not None:
            x = embedded_dropout(self.embedding, x,
                                 dropout=self.dropout_embedding if self.training else 0)
        for hidden_layer_i in range(0, self.num_hidden_layers):
            if self.training and (self.args.learning_rule == 'eprop'):
                t_trace = []
                trace = torch.zeros_like(x[0].detach())
                for each_step_x in x:
                    trace = self.args.decay * trace.detach() + each_step_x
                    t_trace.append(trace)
                t_trace = torch.stack(t_trace)
                t_trace_output = eval("self.fc" + str(hidden_layer_i))(t_trace)
                x = eval("self.fc" + str(hidden_layer_i))(
                    x.detach()).detach() + t_trace_output - t_trace_output.detach()
            else:
                x = eval("self.fc" + str(hidden_layer_i))(x)
            if self.bns is not None:
                x = self.bns[hidden_layer_i](x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
            x = eval("self.dropout" + str(hidden_layer_i))(x)
        x = self.classifier(x)
        return x

class SSMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, spiking_neuron=None):
        super(SSMNet, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        for hidden_layer_i in range(num_hidden_layers):
            exec("self.spk" + str(
                hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size)")
            if hidden_layer_i == 0:
                exec("self.fc" + str(
                    hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size)")
            input_size = hidden_size
        self.classifier = nn.Linear(in_features=input_size, out_features=output_size)
    def forward(self, x, time_step=None, multi_step=True):
        assert multi_step
        x = x.view(x.size(0), x.size(1), -1)
        for hidden_layer_i in range(self.num_hidden_layers):
            if hidden_layer_i == 0:
                x = eval("self.fc" + str(hidden_layer_i))(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        x = self.classifier(x)

        return x

def seq_to_ann_forward(x_seq: Tensor, stateless_module: nn.Module or list or tuple or nn.Sequential or Callable):
    """
    * :ref:`API in English <seq_to_ann_forward-en>`

    .. _seq_to_ann_forward-cn:

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: Tensor
    :param stateless_module: 单个或多个无状态网络层
    :type stateless_module: torch.nn.Module or list or tuple or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: Tensor

    * :ref:`中文 API <seq_to_ann_forward-cn>`

    .. _seq_to_ann_forward-en:

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: Tensor
    :param stateless_module: one or many stateless modules
    :type stateless_module: torch.nn.Module or list or tuple or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: Tensor

    Applied forward on stateless modules

    """
    y_shape = [x_seq.shape[0], x_seq.shape[1]]
    y = x_seq.flatten(0, 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    y_shape.extend(y.shape[1:])
    return y.view(y_shape)


class SeqToANN(nn.Module):
    def __init__(self, module):
        super(SeqToANN, self).__init__()
        self.module = module

    def forward(self, x):
        x = seq_to_ann_forward(x, self.module)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, spiking_neuron=None):
        super(VGG, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.features = self.make_layers(cfg=self.cfg, spiking_neuron=spiking_neuron)
        # if self.cfg == 'VGGSNN':
        #     self.avgpool = nn.Identity()
        #     self.fc = nn.Linear(4608, num_classes)
        # elif self.cfg == 'A':
        #     self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        #     self.fc = nn.Linear(512*7*7, num_classes)
        # else:
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Sequential(nn.Linear(512 * 2 * 2, 1024),
                                BatchNorm1d(1024),
                                spiking_neuron(),
                                nn.Linear(1024, 1024),
                                BatchNorm1d(1024),
                                spiking_neuron())
        self.cls = nn.Linear(1024*2, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.features(x)
        # x = seq_to_ann_forward(x, self.avgpool)
        # print(f"before flatten: {x.size()}")
        x = x.view(x.size(0), x.size(1), -1)
        # print(f"after flatten: {x.size()}")
        # x = self.fc(x)
        x = self.cls(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, self.norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, spiking_neuron):
        layers = []
        in_channels = 2
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [SeqToANN(nn.AvgPool2d(kernel_size=2, stride=2))]
            else:
                if i == 0:
                    layers += [BasicLayer(in_channels, x, spiking_neuron, kernel_size=7, padding=3, stride=2)]
                else:
                    layers += [BasicLayer(in_channels, x, spiking_neuron)]
                in_channels = x
        return nn.Sequential(*layers)


class BasicLayer(nn.Module):

    def __init__(self, in_channels, out_channels, spiking_neuron, kernel_size=3, padding=1, stride=1):
        super(BasicLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.spk = spiking_neuron()

    def forward(self, x):
        x = seq_to_ann_forward(x, [self.conv1, self.bn1])
        # x = self.conv1(x)
        # x = self.bn1(x)
        x = self.spk(x)
        return x


class DVSLIPNet(nn.Module):
    def __init__(self, num_classes, spiking_neuron=None, args=None):
        super(DVSLIPNet, self).__init__()
        # self.num_hidden_layers = num_hidden_layers
        self.args = args
        self.snn = VGG(num_classes=num_classes, spiking_neuron=spiking_neuron)
        # if isinstance(hidden_size, int):
        #     hidden_size = [hidden_size] * num_hidden_layers
        # else:
        #     assert len(hidden_size) == num_hidden_layers

    # def single_step_forward(self, x):
    #     for hidden_layer_i in range(self.num_hidden_layers):
    #         x = eval("self.fc" + str(hidden_layer_i))(x)
    #         x = eval("self.spk" + str(hidden_layer_i))(x)
    #     x = self.classifier(x)
    #     return x

    def forward(self, x, time_step=None, multi_step=True):
        # if time_step is None:
        #     time_step = x.size(0)
        # x = x.view(x.size(0), x.size(1), -1)
        # if multi_step:
        # print(f"input: {x.size()}")
        reset_states(self)
        output = self.snn(x)
        # output = self.multi_step_forward(x, time_step)
        # else:
        #     reset_states(self)
        #     output = []
        #     for t in range(time_step):
        #         single_step_output = self.single_step_forward(x[t])
        #         output.append(single_step_output)
        #     output = torch.stack(output)
        return output

    # def multi_step_forward(self, x, time_step):
    #     for hidden_layer_i in range(0, self.num_hidden_layers):
    #         x = eval("self.fc" + str(hidden_layer_i))(x)
    #
    #         x = eval("self.spk" + str(hidden_layer_i))(x)
    #         x = eval("self.dropout" + str(hidden_layer_i))(x)
    #     x = self.classifier(x)
    #     return x


class DvsGestureSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1, spiking_neuron=None, bn=None,
                 final_step_cls=False, args=None):
        super(DvsGestureSNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.args = args
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(4, 4)
        if final_step_cls:
            last_layer = True
        else:
            last_layer = False

        if bn == 'bn':
            bns = [BatchNorm1d(hidden_size) for l in range(num_hidden_layers)]
        elif bn == 'ln':
            bns = [nn.LayerNorm(hidden_size) for l in range(num_hidden_layers)]
        elif bn == 'tdbn':
            bns = [ThresholdDependentBatchNorm1d(alpha=1., v_th=spiking_neuron().threshold, num_features=hidden_size)
                   for l in range(num_hidden_layers)]
        elif bn == 'tebn':
            bns = [TemporalEffectiveBatchNorm1d(T=spiking_neuron().time_step, num_features=hidden_size) for l in
                   range(num_hidden_layers)]
        elif bn is None:
            bns = [None for _ in range(num_hidden_layers)]
        else:
            raise NotImplementedError

        for hidden_layer_i in range(num_hidden_layers):
            exec("self.fc" + str(hidden_layer_i) + " = nn.Linear(in_features=input_size, out_features=hidden_size)")
            if hidden_layer_i + 1 == num_hidden_layers:
                exec("self.spk" + str(
                    hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size, bn=bns[hidden_layer_i], last_layer=last_layer)")
            else:
                exec("self.spk" + str(
                    hidden_layer_i) + " = spiking_neuron(neuron_num=hidden_size, bn=bns[hidden_layer_i])")
            input_size = hidden_size
        self.classifier = nn.Linear(in_features=input_size, out_features=output_size)

    def single_step_forward(self, x):
        x = self.max_pool(x)
        x = self.flatten(x)
        for hidden_layer_i in range(self.num_hidden_layers):
            x = eval("self.fc" + str(hidden_layer_i))(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)
        x = self.classifier(x)
        return x

    def forward(self, x, time_step=None, multi_step=False):
        if time_step is None:
            time_step = x.size(0)
        if multi_step:
            output = self.multi_step_forward(x, time_step)
        else:
            reset_states(self)
            output = []
            for t in range(time_step):
                single_step_output = self.single_step_forward(x[t])
                output.append(single_step_output)
            output = torch.stack(output)
        return output

    def multi_step_forward(self, x, time_step):
        x = MergeDimension()(x)

        x = self.max_pool(x)
        x = self.flatten(x)
        x = SplitDimension(time_step)(x)
        for hidden_layer_i in range(self.num_hidden_layers):
            # x = MergeDimension()(x)
            if self.training and (self.args.learning_rule == 'eprop'):
                t_trace = []
                trace = torch.zeros_like(x[0].detach())
                for each_step_x in x:
                    trace = self.args.decay * trace.detach() + each_step_x
                    t_trace.append(trace)
                t_trace = torch.stack(t_trace)
                t_trace_output = eval("self.fc" + str(hidden_layer_i))(t_trace)
                x = eval("self.fc" + str(hidden_layer_i))(
                    x.detach()).detach() + t_trace_output - t_trace_output.detach()
            else:
                x = eval("self.fc" + str(hidden_layer_i))(x)
            # x = SplitDimension(time_step)(x)
            x = eval("self.spk" + str(hidden_layer_i))(x)

        # x = MergeDimension()(x)
        x = self.classifier(x)
        # x = SplitDimension(time_step)(x)

        return x


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type, num_hidden_layers=1, spiking_neuron=None,
                 spiking=False, dvs_pooling=False, use_flatten=False):
        super(LSTMNet, self).__init__()

        self.dvs_pooling = dvs_pooling
        if dvs_pooling:
            self.max_pool = nn.Sequential(nn.MaxPool2d(4, 4), nn.Flatten())

        self.rnn_type = rnn_type
        self.time_window = 2
        self.nlayers = num_hidden_layers
        self.use_flatten = use_flatten
        # self.use_bn = use_bn
        if isinstance(hidden_size, int):
            self.hidden_size = [hidden_size] * num_hidden_layers
        else:
            assert len(hidden_size) == num_hidden_layers
            self.hidden_size = hidden_size
        # spiking_neuron = partial(spiking_neuron,
        #                          time_step=self.time_window)
        self.spiking = spiking
        if self.spiking:
            self.spiking_neuron = spiking_neuron()

        if rnn_type == 'lstm':
            if self.spiking:
                self.rnns = [script_lstm(input_size if l == 0 else self.hidden_size[l - 1],
                                         self.hidden_size[l],
                                         num_layers=1, batch_first=False, spiking=self.spiking,
                                         spiking_neuron=spiking_neuron)
                             for l in range(num_hidden_layers)]
            else:
                self.rnns = [nn.LSTM(input_size if l == 0 else self.hidden_size[l - 1],
                                     self.hidden_size[l],
                                     num_layers=1, batch_first=False)
                             for l in range(num_hidden_layers)]
        elif rnn_type == 'gru':
            if self.spiking:
                self.rnns = [script_lstm(input_size if l == 0 else self.hidden_size[l - 1],
                                         self.hidden_size[l],
                                         num_layers=1, batch_first=False, spiking=self.spiking, GRU=True)
                             for l in range(num_hidden_layers)]
            else:
                self.rnns = [nn.GRU(input_size if l == 0 else self.hidden_size[l - 1],
                                    self.hidden_size[l],
                                    num_layers=1, batch_first=False)
                             for l in range(num_hidden_layers)]
        else:
            raise NotImplementedError
        # if self.use_bn:
        #     self.bns = nn.ModuleList([BatchNorm1d(hidden_size[l]) for l in range(num_hidden_layers)])
        self.rnns = nn.ModuleList(self.rnns)
        self.classifier = nn.Linear(self.hidden_size[-1], output_size)

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            if self.spiking:
                return [[(weight.new_zeros(batch_size,
                                           self.hidden_size[l]).to(device),
                          weight.new_zeros(batch_size,
                                           self.hidden_size[l]).to(device))]
                        for l in range(self.nlayers)]
            else:
                return [(weight.new_zeros(1, batch_size,
                                          self.hidden_size[l]).to(device),
                         weight.new_zeros(1, batch_size,
                                          self.hidden_size[l]).to(device))
                        for l in range(self.nlayers)]
        elif self.rnn_type == 'gru':
            if self.spiking:
                return [[(weight.new_zeros(batch_size,
                                           self.hidden_size[l]).to(device),
                          weight.new_zeros(batch_size,
                                           self.hidden_size[l]).to(device))]
                        for l in range(self.nlayers)]
            else:
                # return [[weight.new_zeros(batch_size,
                #                           self.hidden_size).to(device)]
                #         for l in range(self.nlayers)]
                return [[weight.new_zeros(1, batch_size,
                                          self.hidden_size[l]).to(device)]
                        for l in range(self.nlayers)]
        else:
            raise NotImplementedError(
                f"Model '{self.rnn_type}' not implemented.")

    def forward(self, inputs, **kwargs):
        hiddens = inputs
        if self.use_flatten:
            hiddens = hiddens.view(hiddens.size(0), hiddens.size(1), -1)
        state = self.init_hidden(batch_size=hiddens.size(1), device=hiddens.device)
        if self.dvs_pooling:
            time_step = hiddens.size(0)
            hiddens = MergeDimension()(hiddens)
            hiddens = self.max_pool(hiddens)
            hiddens = SplitDimension(time_step)(hiddens)
        # if self.spiking:
        #     # print(f"hiddens: {hiddens.size()}")
        #     hiddens = (hiddens.unsqueeze(1)).repeat(1, self.time_window, 1, 1)
        for l, rnn in enumerate(self.rnns):
            if self.spiking:
                hiddens, final_states = rnn(hiddens, state[l], threshold=self.spiking_neuron.threshold,
                                            surrogate_function=self.spiking_neuron.surrogate_function)
            else:
                hiddens, final_states = rnn(hiddens, state[l])
        # if self.spiking:
        #     hiddens = hiddens.mean(1)
        return self.classifier(hiddens)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse
                                      )
    return X


class TextLSTMNet(nn.Module):
    def __init__(self,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 output_size,
                 rnn_type,
                 num_hidden_layers=1,
                 dropout_embedding=0.,
                 pad_idx=None,
                 spiking_neuron=None,
                 spiking=False):
        super(TextLSTMNet, self).__init__()

        self.hidden_size = hidden_dim

        self.rnn_type = rnn_type
        self.nlayers = num_hidden_layers

        self.dropout_embedding = dropout_embedding

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if rnn_type == 'lstm':
            self.rnns = [nn.LSTM(emb_dim if l == 0 else hidden_dim,
                                 hidden_dim,
                                 num_layers=1, batch_first=False)
                         for l in range(num_hidden_layers)]
        elif rnn_type == 'gru':
            self.rnns = [nn.GRU(emb_dim if l == 0 else hidden_dim,
                                hidden_dim,
                                num_layers=1, batch_first=False)
                         for l in range(num_hidden_layers)]
        else:
            raise NotImplementedError
        self.rnns = nn.ModuleList(self.rnns)
        self.classifier = nn.Linear(hidden_dim, output_size)

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        if self.rnn_type == 'lstm':
            return [(weight.new_zeros(1, batch_size,
                                      self.hidden_size).to(device),
                     weight.new_zeros(1, batch_size,
                                      self.hidden_size).to(device))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'gru':
            return [[weight.new_zeros(1, batch_size,
                                      self.hidden_size).to(device)]
                    for l in range(self.nlayers)]
        else:
            raise NotImplementedError(
                f"Model '{self.rnn_type}' not implemented.")

    def forward(self, inputs, **kwargs):
        state = self.init_hidden(batch_size=inputs.size(1), device=inputs.device)
        hiddens = embedded_dropout(self.embedding, inputs,
                                   dropout=self.dropout_embedding if self.training else 0)
        for l, rnn in enumerate(self.rnns):
            hiddens, final_states = rnn(hiddens, state[l])
        return self.classifier(hiddens)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0., spiking_neuron=None,
                 output_last_step=False,
                 use_pool=False, use_flatten=False, t_internal=4):
        super(TCN, self).__init__()
        self.use_pool = use_pool
        if self.use_pool:
            self.flatten = nn.Flatten()
            self.max_pool = nn.MaxPool2d(4, 4)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout,
                                   spiking_neuron=spiking_neuron)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        self.spiking = spiking_neuron is not None
        self.output_last_step = output_last_step
        self.use_flatten = use_flatten
        self.t_internal = t_internal

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x, **kwargs):
        if self.use_flatten:
            x = x.view(x.size(0), x.size(1), -1)
        if self.use_pool:
            time_step = x.size(0)
            x = MergeDimension()(x)
            x = self.max_pool(x)
            x = self.flatten(x)
            x = SplitDimension(time_step)(x)
        x = x.permute(1, 2, 0).contiguous()  # [T, B, N] -> B, N, T
        if self.spiking:
            reset_states(self)
            T = self.t_internal
            for t in range(T):
                y1 = self.tcn(x)
                if self.output_last_step:
                    if t == 0:
                        output = self.linear(y1[:, :, -1]).unsqueeze(0)
                    else:
                        output = output + self.linear(y1[:, :, -1]).unsqueeze(0)
                else:
                    y1 = y1.permute(2, 0, 1).contiguous()
                    if t == 0:
                        output = self.linear(y1)
                    else:
                        output = output + self.linear(y1)
            output = output / T
        else:
            y1 = self.tcn(x)
            if self.output_last_step:
                output = self.linear(y1[:, :, -1]).unsqueeze(0)
            else:
                y1 = y1.permute(2, 0, 1).contiguous()
                output = self.linear(y1)
        return output


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def _in_projection_packed(
        q,
        k,
        v,
        w,
        b=None,
):
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return nn.functional.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (nn.functional.linear(q, w_q, b_q),) + nn.functional.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return nn.functional.linear(q, w_q, b_q), nn.functional.linear(k, w_k, b_k), nn.functional.linear(v, w_v, b_v)


def _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
):
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = nn.functional.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = nn.functional.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def SDSA3(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        scale=0.125
):
    assert attn_mask is None
    assert dropout_p == 0.0
    T, B, Nt, E = q.shape
    # q = q / math.sqrt(E)
    if True:
        attn = k.transpose(-2, -1) @ v
        # print(f"attn: {attn.size()}")
        # if dropout_p > 0.0:
        #     attn = nn.functional.dropout(attn, p=dropout_p)
        output = (q @ attn) * scale
    else:
        # q = q / math.sqrt(E)
        attn = q @ k.transpose(-2, -1)
        # attn = nn.functional.softmax(attn, dim=-1)
        # print(f"attn: {attn.size()}")
        # print(f"q: {q.size()}")
        output = (attn @ v) * scale

    # print(f"attn output: {output.size()}")
    return output, attn


def spk_multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim_to_check,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        dropout_p,
        out_proj_weight,
        out_proj_bias,
        q_lif=None,
        k_lif=None,
        v_lif=None,
        attn_lif=None,
        training=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # set up shape vars
    T, tgt_len, bsz, embed_dim = query.shape
    T, src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    q = q_lif(q)
    k = k_lif(k)
    v = v_lif(v)
    # print(f"q: {q.size()}, k: {k.size()}, v: {v.size()}")
    # prep attention mask
    assert attn_mask is None, "Not implemented!"

    # prep key padding mask
    assert key_padding_mask is None, "Not implemented"

    # add bias along batch dimension (currently second)

    assert bias_k is None
    assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(T, tgt_len, bsz * num_heads, head_dim).transpose(1, 2)
    k = k.contiguous().view(T, k.shape[1], bsz * num_heads, head_dim).transpose(1, 2)
    v = v.contiguous().view(T, v.shape[1], bsz * num_heads, head_dim).transpose(1, 2)
    # print(f"****q: {q.size()}, k: {k.size()}, v: {v.size()}")

    # add zero attention along batch dimension (now first)

    # update source sequence length after adjustments
    src_len = k.size(2)

    # merge key padding and attention masks
    assert key_padding_mask is None

    # convert mask to float

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = SDSA3(q, k, v, attn_mask, dropout_p)
    # print(f"attn_output: {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous().view(T, tgt_len, bsz, embed_dim)
    attn_output = attn_lif(attn_output)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(T, bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=2) / num_heads
    else:
        return attn_output, None


class SpkMultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']

    # bias_k: Optional[torch.Tensor]
    # bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 batch_first=False, spiking_neuron=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpkMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.bias_k = self.bias_v = None

        self._reset_parameters()

        assert spiking_neuron is not None
        self.scale = 0.125
        self.head_lif = spiking_neuron()
        self.q_lif = spiking_neuron()
        self.k_lif = spiking_neuron()
        self.v_lif = spiking_neuron()
        self.attn_lif = spiking_neuron()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    # def __setstate__(self, state):
    #     # Support loading old MultiheadAttention checkpoints generated by v1.1.0
    #     if '_qkv_same_embed_dim' not in state:
    #         state['_qkv_same_embed_dim'] = True
    #
    #     super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """
        assert self.batch_first is False
        # if self.batch_first:
        #     query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = spk_multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            q_lif=self.q_lif, k_lif=self.k_lif, v_lif=self.v_lif, attn_lif=self.attn_lif,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class SpkTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=torch.nn.functional.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, spiking_neuron=None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SpkTransformerEncoderLayer, self).__init__()
        self.self_attn = SpkMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               spiking_neuron=spiking_neuron,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.head_lif = spiking_neuron()
        self.fc1_lif = spiking_neuron()
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.fc2_lif = spiking_neuron()
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        # if isinstance(activation, str):
        #     self.activation = _get_activation_fn(activation)
        # else:
        self.activation = activation

    # def __setstate__(self, state):
    #     if 'activation' not in state:
    #         state['activation'] = F.relu
    #     super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = self.head_lif(x)
        # print(f"head_lif: {x.size()}")
        assert self.norm_first is False
        # if self.norm_first:
        #     x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        #     x = x + self._ff_block(self.norm2(x))
        # else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        # print(f"norm1: {x.size()}")
        x = self.norm2(x + self._ff_block(x))
        # print(f"norm2: {x.size()}")

        return x

    # self-attention block
    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.fc1_lif(x)
        x = self.linear1(x)
        x = self.fc2_lif(x)
        x = self.linear2(x)
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)

    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)


class TransformerNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 nhead,
                 num_hidden_layers=1,
                 dropout=0,
                 use_pool=False):
        super(TransformerNet, self).__init__()
        self.use_pool = use_pool
        if self.use_pool:
            self.flatten = nn.Flatten()
            self.max_pool = nn.MaxPool2d(4, 4)
        self.encoder = nn.Linear(input_size, hidden_size)

        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
                                                dropout=dropout,
                                                )
        encoder_norm = nn.LayerNorm(hidden_size)
        custom_encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers, encoder_norm)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
                                          num_encoder_layers=num_hidden_layers, dropout=dropout,
                                          custom_encoder=custom_encoder).encoder
        # self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
        #                                   num_encoder_layers=num_hidden_layers, dropout=dropout).encoder

        self.linear = nn.Linear(hidden_size, output_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=0.)

    def forward(self, x, **kwargs):
        if self.use_pool:
            time_step = x.size(0)
            x = MergeDimension()(x)
            x = self.max_pool(x)
            x = self.flatten(x)
            x = SplitDimension(time_step)(x)
        # print(f"x: {x.size()}")
        # x = x.transpose(-2, -1)
        x = self.encoder(x)
        # x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        # print(f"x: {x.size()}")
        x = self.transformer(x)  # input should have dimension (N, C, L)
        # print(f"x: {x.size()}")
        # x = x.transpose(0, 1)
        # x = x.transpose(-2, -1)
        # print(f"x: {x.size()}")
        # exit()
        # o = self.linear(x)
        return self.linear(x)


class SpkTransformerNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 nhead,
                 num_hidden_layers=1,
                 dropout=0,
                 spiking_neuron=None,
                 T=4,
                 use_pool=False,
                 use_flatten=False):
        super(SpkTransformerNet, self).__init__()
        self.use_pool = use_pool
        if self.use_pool:
            self.flatten = nn.Flatten()
            self.max_pool = nn.MaxPool2d(4, 4)
        self.use_flatten = use_flatten
        self.encoder = nn.Linear(input_size, hidden_size)

        encoder_layer = SpkTransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
                                                   dropout=dropout, spiking_neuron=spiking_neuron
                                                   )
        encoder_norm = nn.LayerNorm(hidden_size)
        custom_encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers, encoder_norm)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
                                          num_encoder_layers=num_hidden_layers, dropout=dropout,
                                          custom_encoder=custom_encoder).encoder
        # self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
        #                                   num_encoder_layers=num_hidden_layers, dropout=dropout).encoder
        self.linear_spk = spiking_neuron()
        self.time_window = T
        self.linear = nn.Linear(hidden_size, output_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=0.)

    def forward(self, x, **kwargs):
        reset_states(self)
        if self.use_flatten:
            x = x.view(x.size(0), x.size(1), -1)
        if self.use_pool:
            time_step = x.size(0)
            x = MergeDimension()(x)
            x = self.max_pool(x)
            x = self.flatten(x)
            x = SplitDimension(time_step)(x)
        # print(f"x: {x.size()}")
        # x = x.transpose(-2, -1)
        x = self.encoder(x)
        # x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        # print(f"x: {x.size()}")
        x = (x.unsqueeze(0)).repeat(self.time_window, 1, 1, 1)
        # print(f"x: {x.size()}")
        x = self.transformer(x)  # input should have dimension (N, C, L)
        # print(f"x: {x.size()}")
        # x = x.transpose(0, 1)
        # x = x.transpose(-2, -1)
        # print(f"x: {x.size()}")
        # exit()
        # o = self.linear(x)
        x = self.linear_spk(x)
        # print(f"x: {x.size()}")
        # exit()
        return self.linear(x).mean(0)


class TextTransformerNet(nn.Module):
    def __init__(self,
                 nhead,
                 emb_dim,
                 hidden_dim,
                 vocab_size,
                 output_size,
                 num_hidden_layers=1,
                 spiking_neuron=None,
                 spiking=False):
        super(TextTransformerNet, self).__init__()

        self.hidden_size = hidden_dim

        self.nlayers = num_hidden_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                                dropout=0.,
                                                )
        encoder_norm = nn.LayerNorm(emb_dim)
        custom_encoder = nn.TransformerEncoder(encoder_layer, self.nlayers, encoder_norm)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                          num_encoder_layers=self.nlayers, dropout=0.,
                                          custom_encoder=custom_encoder).encoder
        self.decoder = nn.Linear(emb_dim, output_size)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout=0.)

    def forward(self, inputs, **kwargs):
        emb = embedded_dropout(self.embedding, inputs,
                               dropout=0 if self.training else 0)
        emb = self.pos_encoder(emb)
        y = self.transformer(emb)
        y = self.decoder(y)
        return y
