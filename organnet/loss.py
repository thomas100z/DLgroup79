import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(DiceLoss, self)._init_()

    def forward(self, inputs, targets, smooth=1):
        n, c, _, _, _ = inputs.shape
        inputs = torch.sigmoid(inputs)

        intersection = torch.mul(inputs, targets).sum([2, 3, 4])
        dice = (2. * intersection) / (inputs.sum([2, 3, 4]) + targets.sum([2, 3, 4]) + smooth)
        dice = dice.sum() / c

        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=1):
        super().__init__()

        if alpha is None:
            # alpha = [0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0]
            alpha = torch.tensor([1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            alpha =  alpha.reshape(1,10,1,1,1)

        self.gamma = gamma
        self.alpha = alpha
        self.n = 2

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy(inputs, targets)
        focal_loss = - self.alpha * (1 - inputs) ** self.gamma * bce_loss

        return - (focal_loss / self.n)


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=1, alpha=1):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#
#     def forward(self, inputs, targets):
#         # Adjusted version of binary cross entropy
#         inputs[inputs == 0] = 1e-20
#         bce_loss = torch.log(
#             inputs)  # F.binary_cross_entropy_with_logits(inputs, targets, reduce = False) #Negative log likelihood
#
#         if self.gamma == 0.0:
#             modulator = 1.0
#         else:
#             modulator = torch.pow((1 - torch.exp(-bce_loss)), self.gamma)
#             # modulator = torch.exp(-self.gamma * targets * inputs - self.gamma * torch.log(1 +
#             #     torch.exp(-1.0 * inputs)))
#
#         loss = modulator * bce_loss * targets
#         loss = self.alpha * loss
#
#         focal_loss = torch.sum(loss, dim=(2, 3, 4))
#         focal_loss = torch.sum(focal_loss, dim=1)
#
#         focal_loss = focal_loss.sum() / 2
#
#         return focal_loss


if __name__ == "__main__":
    f = FocalLoss()
    d = DiceLoss()
    b = torch.ones(2, 10, 15, 15, 5)
    c = torch.arange(10)
    print(c)
    c = c.reshape(1,10,1,1,1)
    dsfsdf = torch.mul(c,b)
    print('')
    # print(f(b, b).shape)
