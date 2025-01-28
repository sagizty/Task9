"""
MPL SoftCrossEntropy loss Script  ver： Sep 17th 19:00

update
SoftlabelCrossEntropy loss for soft-label based augmentations
现在相当于是对（B，CLS)与（B，cls）label进行loss
但是你需要做成多个patch同时的：（B，N，CLS)与（B，N，cls）label
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


'''
# define SoftlabelCrossEntropy loss for soft-label based augmentations
def SoftCrossEntropy(input, target, reduction='sum'):  # reduction='sum' fixme 好像说有问题？查一下warning
    log_likelihood = -F.log_softmax(input, dim=1)
    batch = input.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


class SoftlabelCrossEntropy(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'sum') -> None:
        super(SoftlabelCrossEntropy, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return SoftCrossEntropy(input, target, reduction=self.reduction)
'''
