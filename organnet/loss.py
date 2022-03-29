import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def _init_(self):
        super(DiceLoss, self)._init_()

    def forward(self, inputs, targets, smooth=1):
        n, c, _, _, _ = inputs.shape
        inputs = torch.sigmoid(inputs)

        intersection = torch.mul(inputs, targets).sum([2, 3, 4])
        dice = (2. * intersection) / (inputs.sum([2, 3, 4]) + targets.sum([2, 3, 4]) + smooth)
        dice = dice.sum() / c

        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.alpha = alpha.to(DEVICE)

        self.n = 2

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        focal_loss = self.alpha * ((1. - inputs) ** self.gamma) * (targets * torch.log(inputs + 1e-7))
        return - (focal_loss.sum([0,1]) / self.n).mean()


if __name__ == "__main__":
    ALPHA = torch.tensor([1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(1, 10, 1, 1, 1)
    GAMMA = 2
    f = FocalLoss(GAMMA, ALPHA)

    d = DiceLoss()
    a = torch.rand(2, 10, 15, 15, 5) * 50
    b = torch.ones(2, 10, 15, 15, 5)
    print(f(a, b))
    print(f(b, b))
