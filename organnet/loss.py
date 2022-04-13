import torch
import torch.nn as nn
from collections import Counter


class DiceLoss(nn.Module):
    def __init__(self, channels=10):
        super().__init__()
        self.c = channels

    def forward(self, inputs, targets):
        intersection = torch.mul(inputs, targets).sum([2, 3, 4])
        dice = (2. * intersection) / (inputs.sum([2, 3, 4]) + targets.sum([2, 3, 4]))
        dice = dice.sum() / (self.c * inputs.shape[0])

        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha, shape=(2, 10, 256, 256, 48)):
        super().__init__()
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.alpha = alpha.to(self.DEVICE)
        self.n = shape[2] * shape[3] * shape[4]

    def forward(self, inputs, targets):

        focal_loss = self.alpha * torch.pow((1. - inputs), self.gamma) * (targets * torch.log(inputs + 1e-7))
        focal_loss = focal_loss.sum(dim=1).sum([1, 2, 3]) / self.n

        return - (focal_loss.sum() / inputs.shape[0])


if __name__ == "__main__":
    ALPHA = torch.tensor([0.01, 3.0, 12.0, 3.0, 12.0, 12.0, 3.0, 3.0, 5.0, 5.0]).reshape(1, 10, 1, 1, 1)
    GAMMA = 2

    f = FocalLoss(GAMMA, ALPHA, shape=(2, 10, 256, 256, 48))
    d = DiceLoss()

    a = torch.zeros(2, 10, 256, 256, 48)
    b = torch.ones(2, 10, 256, 256, 48)

    for i in range(10):
        print(d(a + 0.1 * i, b))

    for i in range(10):
        print(f(a + 0.1 * i, b))
