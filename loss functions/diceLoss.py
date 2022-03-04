import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(DiceLoss, self)._init_()

    def backward_propagate(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice