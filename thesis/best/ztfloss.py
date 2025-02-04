"""
作者：ztf08
名称：losses.py
说明：
日期：2022/5/19 23:59
"""
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         """
#         Args:
#             inputs (tensor): model outputs
#             targets (tensor): image labels
#             smooth (int, optional): smooth factor. Defaults to 1.
#
#         Returns:
#             loss
#         """
#         # comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)
#
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#
#         return 1 - dice


# class w_FocalLoss(nn.Module):
#     def __init__(self):
#         super(w_FocalLoss, self).__init__()
#
#     def forward(self, inputs, targets, w0, gamma=2):
#         w1 = 1 - w0
#         weights = torch.tensor([w0, w1]).cuda().float()
#         logpt = F.cross_entropy(inputs, targets, weight=weights)
#         pt = torch.exp(logpt)
#         # compute the loss
#         loss = ((1 - pt) ** gamma) * logpt
#         return loss


class Jonint_Loss(nn.Module):
    def __init__(self):
        super(Jonint_Loss, self).__init__()

    def w_FocalLoss(self, inputs, targets, w0, gamma=2):
        targets = F.one_hot(targets,num_classes=2).float()
        w1 = 1 - w0
        # weights = torch.tensor([1, 1]).cuda().float()
        weights = torch.tensor([w0, w1]).cuda().float()
        logpt = F.binary_cross_entropy_with_logits(inputs, targets, weight=weights)
        # logpt = F.cross_entropy(inputs, targets)
        # pt = torch.exp(logpt)
        # compute the loss
        # loss = ((1 - pt) ** gamma) * log
        loss = logpt
        # print('logpt:',loss)
        return loss

    # todo 修改dice损失函数
    def DiceLoss(self, output, target, eps=1e-5):  # soft dice loss

        target = target.float()

        num = 2 * (output * target).sum() + eps
        den = output.sum() + target.sum() + eps

        return 1.0 - num / den

    def softmax_dice_loss(self, output, target, alpha=1e-5):
        output = F.softmax(output, dim=1)
        A = torch.unique(target)
        # for i in range(target.shape[0]):
        #     assert len(A)>1, '无标签'
        # torch.unique:去除数组中的重复数字，并进行排序之后输出。
        if torch.any(A == 0):
            loss0 = self.DiceLoss(output[:, 0, ...], (target == 0).float(), eps=alpha)
        else:
            loss0 = 0
        if torch.any(A == 1):
            loss1 = self.DiceLoss(output[:, 1, ...], (target == 1).float(), eps=alpha)
        else:
            loss1 = 0
        if torch.any(A == 2):
            loss2 = self.DiceLoss(output[:, 2, ...], (target == 2).float(), eps=alpha)
        else:
            loss2 = 0
        if torch.any(A == 3):
            loss3 = self.DiceLoss(output[:, 3, ...], (target == 3).float(), eps=alpha)
        else:
            loss3 = 0
        if torch.any(A == 4):
            loss4 = self.DiceLoss(output[:, 4, ...], (target == 4).float(), eps=alpha)
        else:
            loss4 = 0
        if torch.any(A == 5):
            loss5 = self.DiceLoss(output[:, 5, ...], (target == 5).float(), eps=alpha)
        else:
            loss5 = 0
        if torch.any(A == 6):
            loss6 = self.DiceLoss(output[:, 6, ...], (target == 6).float(), eps=alpha)
        else:
            loss6 = 0
        if torch.any(A == 7):
            loss7 = self.DiceLoss(output[:, 7, ...], (target == 7).float(), eps=alpha)
        else:
            loss7 = 0
        if torch.any(A == 8):
            loss8 = self.DiceLoss(output[:, 8, ...], (target == 8).float(), eps=alpha)
        else:
            loss8 = 0
        if torch.any(A == 9):
            loss9 = self.DiceLoss(output[:, 9, ...], (target == 9).float(), eps=alpha)
        else:
            loss9 = 0

        diceloss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 +loss9) / 10
        # print(diceloss)

        return diceloss

    def forward(self, inputs_seg, targets_seg):
        diceloss = self.softmax_dice_loss(inputs_seg, targets_seg)
        return diceloss
