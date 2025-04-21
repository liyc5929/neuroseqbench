from ..trainer import SurrogateGradient # TODO: Add default surrogate gradient function
from .module import MergeDimension, SplitDimension

import torch
from torch.nn import Module, Conv2d, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear, BatchNorm2d, GroupNorm, Sequential


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, time_step, stride=1, downsample=None, norm_layer=None, spiking_neuron=None):
        super(BasicBlock, self).__init__()
        self.spiking_neuron = spiking_neuron
        self.conv1          = Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1            = norm_layer(planes)
        self.sn1            = self.spiking_neuron
        self.conv2          = Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2            = norm_layer(planes)
        self.sn2            = self.spiking_neuron
        self.downsample     = downsample
        self.time_step      = time_step

    def forward(self, x):
        x = MergeDimension()(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = SplitDimension(self.time_step)(x)

        x = self.sn1(x)

        x = MergeDimension()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = SplitDimension(self.time_step)(x)
        
        x = self.sn2(x)

        return x


class SpikingResNet(Module):
    def __init__(self, block, layers, time_step, num_classes, spiking_neuron=None):
        super(SpikingResNet, self).__init__()
        norm_layer          = BatchNorm2d
        self._norm_layer    = norm_layer
        self.time_step      = time_step
        self.spiking_neuron = spiking_neuron
        self.inplanes       = 64
        self.conv1          = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1            = norm_layer(self.inplanes)
        self.sn1            = self.spiking_neuron
        self.maxpool        = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1         = self._make_layer(block, 64,  layers[0], stride=1, spiking_neuron=self.spiking_neuron)
        self.layer2         = self._make_layer(block, 128, layers[1], stride=2, spiking_neuron=self.spiking_neuron)
        self.layer3         = self._make_layer(block, 256, layers[2], stride=2, spiking_neuron=self.spiking_neuron)
        self.layer4         = self._make_layer(block, 512, layers[3], stride=2, spiking_neuron=self.spiking_neuron)
        self.avgpool        = AdaptiveAvgPool2d((1, 1))
        self.flat           = Flatten(start_dim=1, end_dim=-1)
        self.fc             = Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (BatchNorm2d, GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, spiking_neuron=None):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(self.inplanes, planes * block.expansion, stride=stride, kernel_size=1, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.time_step, stride, downsample, norm_layer, spiking_neuron))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.time_step, norm_layer=norm_layer, spiking_neuron=spiking_neuron))
        return Sequential(*layers)

    def forward(self, x):
        x = MergeDimension()(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = SplitDimension(self.time_step)(x)

        x = self.sn1(x)

        x = MergeDimension()(x)
        x = self.maxpool(x)
        x = SplitDimension(self.time_step)(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = MergeDimension()(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)
        x = SplitDimension(self.time_step)(x)

        return x


# User interfaces
def spiking_resnet18(time_step, num_classes, spiking_neuron):
    return SpikingResNet(block=BasicBlock, layers=[2, 2, 2, 2], time_step=time_step, num_classes=num_classes, spiking_neuron=spiking_neuron)
