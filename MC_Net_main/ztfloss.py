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
        if output.shape[1]>1:
            output = F.softmax(output, dim=1)
        else:
            output = F.sigmoid(output)
        A = torch.unique(target)
        # for i in range(target.shape[0]):
        #     assert len(A)>1, '无标签'
        # torch.unique:去除数组中的重复数字，并进行排序之后输出。
        loss = 0
        if len(A)==2:
            loss += self.DiceLoss((output.squeeze()).float(), (target).float(), eps=alpha)
        else:
            for i in A:
                i = i.item()
                if i!=0:
                    loss += self.DiceLoss(output[:, i, ...], (target == i).float(), eps=alpha)

        diceloss = loss / (len(A)-1)

        return diceloss

    def forward(self, inputs_seg, targets_seg):
        diceloss = self.softmax_dice_loss(inputs_seg, targets_seg)
        return diceloss


class TI_Loss(torch.nn.Module):
    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        """
        :param dim: 2 if 2D; 3 if 3D
        :param connectivity: 4 or 8 for 2D; 6 or 26 for 3D
        :param inclusion: list of [A,B] classes where A is completely surrounded by B.
        :param exclusion: list of [A,C] classes where A and C exclude each other.
        :param min_thick: Minimum thickness/separation between the two classes. Only used if connectivity is 8 for 2D or 26 for 3D
        """
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        for inc in inclusion:
            temp_pair = []
            temp_pair.append(True)  # type inclusion
            temp_pair.append(inc[0])
            temp_pair.append(inc[1])
            self.interaction_list.append(temp_pair)

        for exc in exclusion:
            temp_pair = []
            temp_pair.append(False)  # type exclusion
            temp_pair.append(exc[0])
            temp_pair.append(exc[1])
            self.interaction_list.append(temp_pair)

    def set_kernel(self):
        """
        Sets the connectivity kernel based on user's sepcification of dim, connectivity, min_thick
        """
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))

        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array([
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
                ])
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))

        self.kernel = torch_kernel = torch.from_numpy(np.expand_dims(np.expand_dims(np_kernel, axis=0), axis=0))

    def topological_interaction_module(self, P):
        """
        Given a discrete segmentation map and the intended topological interactions, this module computes the critical voxels map.
        :param P: Discrete segmentation map
        :return: Critical voxels map
        """

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Get Masks
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()

            # Get Neighbourhood Information
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding='same')
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding='same')
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Get the pixels which induce errors
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(critical_voxels_map, violating).double()

        return critical_voxels_map

    def forward(self, x, y):
        """
        The forward function computes the TI loss value.
        :param x: Likelihood map of shape: b, c, x, y(, z) with c = total number of classes
        :param y: GT of shape: b, c, x, y(, z) with c=1. The GT should only contain values in [0,L) range where L is the total number of classes.
        :return:  TI loss value
        """

        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete离散 segmentation map
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(), dim=1)
        del x_softmax

        # Call the Topological Interaction Module
        critical_voxels_map = self.topological_interaction_module(P)

        # Compute the TI loss value
        ce_tensor = torch.unsqueeze(self.ce_loss_func(x.double(), y[:, 0].long()), dim=1)
        ce_tensor[:, 0] = ce_tensor[:, 0] * torch.squeeze(critical_voxels_map, dim=1)
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()

        return ce_loss_value

