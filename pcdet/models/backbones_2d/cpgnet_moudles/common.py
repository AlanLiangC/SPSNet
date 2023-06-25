#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    : 2022/06/24 10:15:22
@Author  : Hongda Chang
@Version : 0.1
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def arctan2(x1, x2):
    x1_zero = x1 == 0
    x2_zero = x2 == 0
    x2_neg = x2 < 0

    # to prevent `nan` in output of `torch.arctan`.
    # we expect arctan(0/0) to be 0, and arctan(0/any) happens to be 0.
    x2 = x2 + (x1_zero & x2_zero)
    phi = torch.arctan(x1/x2)

    add_pi = ((x1 > 0) & x2_neg) | (x1_zero & x2_neg)
    neg_pi = (x1 < 0) & x2_neg

    phi += add_pi * math.pi
    phi -= neg_pi * math.pi

    return phi


class CBN2d(nn.Module):
    """ Conv2d + BacthNorm2d + NoLinear
    """
    
    def __init__(self, 
                 in_channels, out_channels: int, kernel_size=3, stride=1, padding=1, bias=True,
                 no_linear=nn.ReLU()):
        super(CBN2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.no_linear = no_linear

    def forward(self, inputs):
        outputs = self.bn(self.conv(inputs))
        
        if isinstance(self.no_linear, nn.Module):
            outputs = self.no_linear(outputs)

        return outputs


class MLP(nn.Module):
    """ Very simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, end_nolinear=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self.end_nolinear = end_nolinear

    def forward(self, inputs):
        """
        Args:
            inputs: shape with :math:`(..., D)`
        
        Return:
            outputs: shape with :math:`(..., E)`
        """
        for i, layer in enumerate(self.layers):
            inputs = F.relu(layer(inputs)) if i < self.num_layers - 1 else layer(inputs)
        return F.relu(inputs) if self.end_nolinear else inputs