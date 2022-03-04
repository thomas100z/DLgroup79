import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss2(nn.Module):
    def __init__(self, gamma=1, alpha=1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Adjusted version of binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)  # nonlogit version gives error
        focal_loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        return focal_loss
