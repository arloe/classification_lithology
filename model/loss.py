import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def ce_loss(output, target):
    return F.cross_entropy(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def mae_loss(output, target):
    return F.l1_loss(output, target)


def hub_loss(output, target):
    return F.smooth_l1_loss(output, target)


def kldiv_loss(output, target):
    return F.kl_div(output, target)


class MeanPoweredErrorLoss(nn.Module):
    def __init__(self, exponent):
        super(MeanPoweredErrorLoss, self).__init__()
        self.exponent = exponent

    def forward(self, output, target):
        # return MAE, MSE, or MTE depending on the exponent
        return torch.mean(torch.pow(torch.abs(output - target), self.exponent))
