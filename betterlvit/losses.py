# -*- coding: utf-8 -*-
"""Segmentation loss functions.

Currently only WeightedDiceBCE is wired into the training loop
(scripts/train.py); the other classes are kept verbatim from the legacy
utils.py for ablation reuse.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'WeightedBCE',
    'WeightedDiceLoss',
    'BinaryDiceLoss',
    'MultiClassDiceLoss',
    'DiceLoss',
    'WeightedDiceCE',
    'WeightedDiceBCE_unsup',
    'WeightedDiceBCE',
]


class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # print("====",logit_pixel.size())
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert (logit.shape == truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (logit.shape == truth.shape)
        p = logit.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        w = truth.detach()
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
        p = w * (p)
        t = w * (t)
        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice = 1 - (2 * intersection + smooth) / (union + smooth)

        loss = dice.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, inputs, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat + targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.dice_loss = WeightedDiceLoss()

    def forward(self, inputs, targets):
        # print(inputs.shape)
        assert inputs.shape == targets.shape, "predict & target shape do not match"
        total_loss = 0
        # logits = F.softmax(inputs, dim=1)
        for i in range(5):
            dice_loss = self.dice_loss(inputs[:, i], targets[:, i])
            total_loss += dice_loss
            total_loss = total_loss / 5
        return total_loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        dice1 = self._dice_loss(inputs[:, 1], target[:, 1]) * weight[1]
        dice2 = self._dice_loss(inputs[:, 2], target[:, 2]) * weight[2]
        dice3 = self._dice_loss(inputs[:, 3], target[:, 3]) * weight[3]
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes, dice1, dice2, dice3


class WeightedDiceCE(nn.Module):
    def __init__(self, dice_weight=0.5, CE_weight=0.5):
        super(WeightedDiceCE, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(4)  # OSIC: 4, RITE: 5
        self.CE_weight = CE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        # inputs = inputs.argmax(dim=1)
        dice, dice1, dice2, dice3 = self.dice_loss(inputs, targets)
        hard_dice_coeff = 1 - dice
        dice01 = 1 - dice1
        dice02 = 1 - dice2
        dice03 = 1 - dice3
        torch.cuda.empty_cache()
        return hard_dice_coeff, dice01, dice02, dice03

    def forward(self, inputs, targets):
        targets = targets.long()
        dice_CE_loss = self.dice_loss(inputs, targets)
        torch.cuda.empty_cache()
        return dice_CE_loss


class WeightedDiceBCE_unsup(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1):
        super(WeightedDiceBCE_unsup, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets, LV_loss):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE + 0.1 * LV_loss

        return dice_BCE_loss


class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=1, BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss
