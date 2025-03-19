import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import base
from torch import Tensor
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_any_t
from typing import Optional, List, Tuple, Union
from typing import Callable
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np

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


def multi_step_forward(x_seq: Tensor, single_step_module: nn.Module or list[nn.Module] or tuple[nn.Module] or nn.Sequential or Callable):
    """
    * :ref:`API in English <multi_step_forward-en>`

    .. _multi_step_forward-cn:

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: Tensor
    :param single_step_module: 一个或多个单步模块
    :type single_step_module: torch.nn.Module or list[nn.Module] or tuple[nn.Module] or torch.nn.Sequential or Callable
    :return: ``shape=[T, batch_size, ...]`` 的输出tensor
    :rtype: torch.Tensor

    在单步模块 ``single_step_module`` 上使用多步前向传播。

    * :ref:`中文 API <multi_step_forward-cn>`

    .. _multi_step_forward-en:

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: torch.Tensor
    :param single_step_module: one or many single-step modules
    :type single_step_module: torch.nn.Module or list[nn.Module] or tuple[nn.Module] or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: torch.torch.Tensor

    Applies multi-step forward on ``single_step_module``.

    """
    y_seq = []
    if isinstance(single_step_module, (list, tuple, nn.Sequential)):
        for t in range(x_seq.shape[0]):
            x_seq_t = x_seq[t]
            for m in single_step_module:
                x_seq_t = m(x_seq_t)
            y_seq.append(x_seq_t)
    else:
        for t in range(x_seq.shape[0]):
            y_seq.append(single_step_module(x_seq[t]))

    return torch.stack(y_seq)

class MultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(
                        f"{m} supports for step_mode == 's', which should not be contained by MultiStepContainer!")

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: ``shape=[T, batch_size, ...]``
        :type x_seq: Tensor
        :return: y_seq with ``shape=[T, batch_size, ...]``
        :rtype: Tensor
        """
        return multi_step_forward(x_seq, super().forward)


class SeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(
                        f"{m} supports for step_mode == 's', which should not be contained by SeqToANNContainer!")

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[T, batch_size, ...]
        :type x_seq: Tensor
        :return: y_seq, shape=[T, batch_size, ...]
        :rtype: Tensor
        """
        return seq_to_ann_forward(x_seq, super().forward)


class StepModeContainer(nn.Sequential, base.StepModule):
    def __init__(self, stateful: bool, *args):
        super().__init__(*args)
        self.stateful = stateful
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(
                        f"{m} supports for step_mode == 's', which should not be contained by StepModeContainer!")
        self.step_mode = 's'

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return super().forward(x)
        elif self.step_mode == 'm':
            if self.stateful:
                return multi_step_forward(x, super().forward)
            else:
                return seq_to_ann_forward(x, super().forward)


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: Tensor):
        if x.dim() != 4 and x.dim() != 3:
            raise ValueError(f'expected x with shape [T, N, C, L] or [T, N, C], but got x with shape {x.shape}!')
        return seq_to_ann_forward(x, super().forward)



class _ThresholdDependentBatchNormBase(_BatchNorm, base.MultiStepModule):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.v_th = v_th
        assert self.affine, "ThresholdDependentBatchNorm needs to set `affine = True`!"
        torch.nn.init.constant_(self.weight, alpha * v_th)

    def forward(self, x_seq):
        return seq_to_ann_forward(x_seq, super().forward)


class ThresholdDependentBatchNorm1d(_ThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm1d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm1d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm1d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm1d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm1d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm1d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 4 - 1 or input.dim() == 3 - 1  # [T * N, C, L]


class TemporalEffectiveBatchNorm1d(nn.Module):
    bn_instance = BatchNorm1d

    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super().__init__()
        self.bn = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.scale = nn.Parameter(torch.ones([T]))
    def forward(self, x_seq: torch.Tensor):
        # x.shape = [T, B, N]
        return self.bn(x_seq) * self.scale[:x_seq.size(0)].view(-1, 1, 1)
















class BatchNorm2d(nn.BatchNorm2d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm2d-en>`

        .. _BatchNorm2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm2d`

        * :ref:`中文 API <BatchNorm2d-cn>`

        .. _BatchNorm2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm2d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            return seq_to_ann_forward(x, super().forward)


class BatchNorm3d(nn.BatchNorm3d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm3d-en>`

        .. _BatchNorm3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm3d`

        * :ref:`中文 API <BatchNorm3d-cn>`

        .. _BatchNorm3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm3d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f'expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!')
            return seq_to_ann_forward(x, super().forward)




class ThresholdDependentBatchNorm2d(_ThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm2d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm2d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm2d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm2d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm2d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm2d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 5 - 1  # [T * N, C, H, W]


class ThresholdDependentBatchNorm3d(_ThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm3d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm3d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm3d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm3d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm3d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm3d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 6 - 1  # [T * N, C, H, W, D]





# OTTT modules

class ReplaceforGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)


class GradwithTrace(nn.Module):
    def __init__(self, module):
        """
        * :ref:`API in English <GradwithTrace-en>`

        .. _GradwithTrace-cn:

        :param module: 需要包装的模块

        用于随时间在线训练时，根据神经元的迹计算梯度
        出处：'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        * :ref:`中文 API <GradwithTrace-cn>`

        .. _GradwithTrace-en:

        :param module: the module that requires wrapping

        Used for online training through time, calculate gradients by the traces of neurons
        Reference: 'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]

        with torch.no_grad():
            out = self.module(spike).detach()

        in_for_grad = ReplaceforGrad.apply(spike, trace)
        out_for_grad = self.module(in_for_grad)

        x = ReplaceforGrad.apply(out_for_grad, out)

        return x


class SpikeTraceOp(nn.Module):
    def __init__(self, module):
        """
        * :ref:`API in English <SpikeTraceOp-en>`

        .. _SpikeTraceOp-cn:

        :param module: 需要包装的模块

        对脉冲和迹进行相同的运算，如Dropout，AvgPool等

        * :ref:`中文 API <GradwithTrace-cn>`

        .. _SpikeTraceOp-en:

        :param module: the module that requires wrapping

        perform the same operations for spike and trace, such as Dropout, Avgpool, etc.

        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]

        spike = self.module(spike)
        with torch.no_grad():
            trace = self.module(trace)

        x = [spike, trace]

        return x


class OTTTSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        for module in self:
            if not isinstance(input, list):
                input = module(input)
            else:
                if len(list(module.parameters())) > 0:  # e.g., Conv2d, Linear, etc.
                    module = GradwithTrace(module)
                else:  # e.g., Dropout, AvgPool, etc.
                    module = SpikeTraceOp(module)
                input = module(input)
        return input


# weight standardization modules

# class WSConv2d(Conv2d):
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: _size_2_t,
#             stride: _size_2_t = 1,
#             padding: Union[str, _size_2_t] = 0,
#             dilation: _size_2_t = 1,
#             groups: int = 1,
#             bias: bool = True,
#             padding_mode: str = 'zeros',
#             step_mode: str = 's',
#             gain: bool = True,
#             eps: float = 1e-4
#     ) -> None:
#         """
#         * :ref:`API in English <WSConv2d-en>`
#
#         .. _WSConv2d-cn:
#
#         :param gain: 是否对权重引入可学习的缩放系数
#         :type gain: bool
#
#         :param eps: 预防数值问题的小量
#         :type eps: float
#
#         其他的参数API参见 :class:`Conv2d`
#
#         * :ref:`中文 API <WSConv2d-cn>`
#
#         .. _WSConv2d-en:
#
#         :param gain: whether introduce learnable scale factors for weights
#         :type step_mode: bool
#
#         :param eps: a small number to prevent numerical problems
#         :type eps: float
#
#         Refer to :class:`Conv2d` for other parameters' API
#         """
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
#                          step_mode)
#         if gain:
#             self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
#         else:
#             self.gain = None
#         self.eps = eps
#
#     def get_weight(self):
#         fan_in = np.prod(self.weight.shape[1:])
#         mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
#         var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
#         weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
#         if self.gain is not None:
#             weight = weight * self.gain
#         return weight
#
#     def _forward(self, x: Tensor):
#         return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#     def forward(self, x: Tensor):
#         if self.step_mode == 's':
#             x = self._forward(x)
#
#         elif self.step_mode == 'm':
#             if x.dim() != 5:
#                 raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
#             x = seq_to_ann_forward(x, self._forward)
#
#         return x
#
# class TemporalEffectiveBatchNorm2d(TemporalEffectiveBatchNormNd):
#     bn_instance = BatchNorm2d
#
#     def __init__(
#             self,
#             T: int,
#             num_features,
#             eps=1e-5,
#             momentum=0.1,
#             affine=True,
#             track_running_stats=True,
#             step_mode='s'
#     ):
#         """
#         * :ref:`API in English <TemporalEffectiveBatchNorm2d-en>`
#
#         .. _TemporalEffectiveBatchNorm2d-cn:
#
#         :param T: 总时间步数
#         :type T: int
#
#         其他参数的API参见 :class:`BatchNorm2d`
#
#         `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。
#
#         TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
#         学习的参数。
#
#         * :ref:`中文 API <TemporalEffectiveBatchNorm2d-cn>`
#
#         .. _TemporalEffectiveBatchNorm2d-en:
#
#         :param T: the number of time-steps
#         :type T: int
#
#         Refer to :class:`BatchNorm2d` for other parameters' API
#
#         Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.
#
#         TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.
#
#         """
#         super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)
#
#     def multi_step_forward(self, x_seq: torch.Tensor):
#         # x.shape = [T, N, C, H, W]
#         return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1, 1)
#
#
# class TemporalEffectiveBatchNorm3d(TemporalEffectiveBatchNormNd):
#     bn_instance = BatchNorm3d
#
#     def __init__(
#             self,
#             T: int,
#             num_features,
#             eps=1e-5,
#             momentum=0.1,
#             affine=True,
#             track_running_stats=True,
#             step_mode='s'
#     ):
#         """
#         * :ref:`API in English <TemporalEffectiveBatchNorm3d-en>`
#
#         .. _TemporalEffectiveBatchNorm3d-cn:
#
#         :param T: 总时间步数
#         :type T: int
#
#         其他参数的API参见 :class:`BatchNorm3d`
#
#         `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。
#
#         TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
#         学习的参数。
#
#         * :ref:`中文 API <TemporalEffectiveBatchNorm3d-cn>`
#
#         .. _TemporalEffectiveBatchNorm3d-en:
#
#         :param T: the number of time-steps
#         :type T: int
#
#         Refer to :class:`BatchNorm3d` for other parameters' API
#
#         Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.
#
#         TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.
#
#         """
#         super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)
#
#     def multi_step_forward(self, x_seq: torch.Tensor):
#         # x.shape = [T, N, C, H, W, D]
#         return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1, 1, 1)

class Dropout(base.MemoryModule):
    def __init__(self, p=0.5, step_mode='m'):
        """
        * :ref:`API in English <Dropout.__init__-en>`

        .. _Dropout.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        与 ``torch.nn.Dropout`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        .. tip::

            这种Dropout最早由 `Enabling Spike-based Backpropagation for Training Deep Neural Network Architectures
            <https://arxiv.org/abs/1903.06379>`_ 一文进行详细论述：

            There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
            training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
            are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
            iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
            the output error and modify the network parameters only at the last time step. For dropout to be effective in
            our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
            data is not changed, such that the neural network is constituted by the same random subset of units during
            each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
            each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
            iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
            are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
            time window within an iteration.

        * :ref:`中文API <Dropout.__init__-cn>`

        .. _Dropout.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        This layer is almost same with ``torch.nn.Dropout``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        .. admonition:: Tip
            :class: tip

            This kind of Dropout is firstly described in `Enabling Spike-based Backpropagation for Training Deep Neural
            Network Architectures <https://arxiv.org/abs/1903.06379>`_:

            There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
            training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
            are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
            iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
            the output error and modify the network parameters only at the last time step. For dropout to be effective in
            our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
            data is not changed, such that the neural network is constituted by the same random subset of units during
            each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
            each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
            iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
            are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
            time window within an iteration.
        """
        super().__init__()
        self.step_mode = step_mode
        assert 0 <= p < 1
        self.register_memory('mask', None)
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def create_mask(self, x: Tensor):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def single_step_forward(self, x: Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return x * self.mask
        else:
            return x

    def multi_step_forward(self, x_seq: Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x_seq[0])

            return x_seq * self.mask
        else:
            return x_seq


class Dropout2d(Dropout):
    def __init__(self, p=0.2, step_mode='s'):
        """
        * :ref:`API in English <Dropout2d.__init__-en>`

        .. _Dropout2d.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        与 ``torch.nn.Dropout2d`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        关于SNN中Dropout的更多信息，参见 :ref:`layer.Dropout <Dropout.__init__-cn>`。

        * :ref:`中文API <Dropout2d.__init__-cn>`

        .. _Dropout2d.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        This layer is almost same with ``torch.nn.Dropout2d``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        For more information about Dropout in SNN, refer to :ref:`layer.Dropout <Dropout.__init__-en>`.
        """
        super().__init__(p, step_mode)

    def create_mask(self, x: Tensor):
        self.mask = F.dropout2d(torch.ones_like(x.data), self.p, training=True)