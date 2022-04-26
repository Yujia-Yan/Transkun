import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import torch.distributed as dist

"""
adapted from https://github.com/atranitell/Synchronized-BatchNorm-PyTorch-Horovod/blob/master/sync_bn.py
For customized distributed training 
"""

class AllReduce_autograd_enabled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        X = torch.clone(input)
        dist.all_reduce(X)
        return X

    @staticmethod
    def backward(ctx, grad_output):
        G = torch.clone(grad_output)
        dist.all_reduce(G)
        return G
    

allreduce_autograd_enabled = AllReduce_autograd_enabled.apply

class _SynchronizedBatchNorm(_BatchNorm):
  r"""Synchronized BatchNorm align with pytorch 1.2 batchnorm syntax.
    1) training=true, track_running_stats=true: 
      running_mean and running_var are just tracked and do not use.
      mean and var are computed by batch samples.
    2) training=true, track_running_stats=false:
      running_mean and running_var do not track and use.
      mean and var are computed by batch samples.
    3) training=false, track_running_stats=true:
      using running_mean and running_var instead of mean/var by batches.
    4) training=false, track_running_stats=false:
      using batch mean/var instead of running_mean and running_var.
  """

  def __init__(self, num_features,
               eps=1e-5,
               momentum=0.01,
               affine=True,
               track_running_stats=True):
    super(_SynchronizedBatchNorm, self).__init__(
        num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats)

    

  def forward(self, inputs):
    self._check_input_dim(inputs)

    # exponential_average_factor is self.momentum set to
    # (when it is available) only so that if gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
      exponential_average_factor = 0.0
    else:
      exponential_average_factor = self.momentum

    # setting momentum
    if self.track_running_stats:
      # TODO: if statement only here to tell the jit to skip emitting this when it is None
      if self.num_batches_tracked is not None:
        self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
          exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
          exponential_average_factor = self.momentum

    # if not self.training:
    #   return F.batch_norm(
    #       inputs, self.running_mean, self.running_var, self.weight, self.bias,
    #       self.training or not self.track_running_stats,
    #       exponential_average_factor, self.eps)

    # Resize the input to (B, C, -1).
    ch = self.num_features
    inputs_shape = inputs.size()
    inputs = inputs.reshape(inputs.size(0), ch, -1)

    # reshape
    if self.affine:
      weight = self.weight.view(1, ch, 1)
      bias = self.bias.view(1, ch, 1)

    # verification inference version (the only is used to test function)
    if not self.training:
      if self.track_running_stats:
        mean = self.running_mean.view(1, ch, 1)
        inv_std = 1 / (self.running_var + self.eps).sqrt().view(1, ch, 1)
        if self.affine:
          outputs = weight * inv_std * (inputs - mean) + bias
        else:
          outputs = inv_std * (inputs - mean)
      else:
        var, mean = torch.var_mean(
            inputs, unbiased=False, dim=[0, 2], keepdim=True)
        inv_std = 1 / (var + self.eps).sqrt()
        if self.affine:
          outputs = weight * inv_std * (inputs - mean) + bias
        else:
          outputs = inv_std * (inputs - mean)
      return outputs.reshape(inputs_shape)

    # Compute the sum and square-sum.
    world_size = dist.get_world_size()
    sum_size = inputs.size(0) * inputs.size(2)*world_size
    stat_sum = inputs.sum(dim=[0, 2])
    stat_ssum = inputs.pow(2).sum(dim=[0, 2])

    # Reduce-and-broadcast the statistics.
    # concat in order to broadcast once
    stats = torch.stack([stat_sum, stat_ssum]).to(inputs.device)
    sync_sum, sync_ssum = allreduce_autograd_enabled(stats).split(1)



    # VAR = E(X^2) - (EX)^2
    mean = sync_sum.view(1, ch, 1) / sum_size
    var = sync_ssum.view(1, ch, 1) / sum_size - mean.pow(2)
    inv_std = 1. / (var + self.eps).sqrt()

    # track running stat
    if self.track_running_stats:
      with torch.no_grad():
        m = exponential_average_factor
        uvar = sync_ssum / (sum_size - 1) - (sync_sum / (sum_size)).pow(2)
        self.running_mean = (1 - m) * self.running_mean + m * mean.view(-1)
        self.running_var = (1 - m) * self.running_var + m * uvar.view(-1)

    # affine
    if self.affine:
      outputs = weight * inv_std * (inputs - mean) + bias
    else:
      outputs = (inputs - mean) * inv_std
    return outputs.reshape(inputs_shape)


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
  def _check_input_dim(self, inputs):
    if inputs.dim() != 2 and inputs.dim() != 3:
      raise ValueError('expected 2D or 3D inputs (got {}D inputs)'
                       .format(inputs.dim()))


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
  def _check_input_dim(self, inputs):
    if inputs.dim() != 4:
      raise ValueError('expected 4D inputs (got {}D inputs)'
                       .format(inputs.dim()))


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
  def _check_input_dim(self, inputs):
    if inputs.dim() != 5:
      raise ValueError('expected 5D inputs (got {}D inputs)'
                       .format(inputs.dim()))
