import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(DiceLoss, self)._init_()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        intersection = torch.mul(inputs, targets).sum([0, 2, 3, 4])
        dice = (2. * intersection) / (inputs.sum([0, 2, 3, 4]) + targets.sum([0, 2, 3, 4]) + smooth)
        dice = dice.sum() / 10

        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, gamma=1, alpha=1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Adjusted version of binary cross entropy
        inputs[inputs == 0] = 1e-20
        bce_loss = torch.log(inputs) #F.binary_cross_entropy_with_logits(inputs, targets, reduce = False) #Negative log likelihood
        
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.pow((1-torch.exp(-bce_loss)), self.gamma)
            # modulator = torch.exp(-self.gamma * targets * inputs - self.gamma * torch.log(1 + 
            #     torch.exp(-1.0 * inputs)))

        loss = modulator * bce_loss * targets
        loss = self.alpha*loss

        focal_loss = torch.sum(loss, dim=(2,3,4))
        focal_loss = torch.sum(focal_loss, dim = 1)

        focal_loss = focal_loss.sum()/2

        return focal_loss

if __name__ == "__main__":
    f = FocalLoss()
    d = DiceLoss()
    a = torch.zeros(2, 10, 256, 256, 48)+0.9999
    b = torch.ones(2, 10, 256, 256, 48)
    #print(d(b, b))
    #print(d(b, a))
    #print('')
    print('loss', f(a, b))
    #print(f(b, a))


# class FocalLoss_wrong(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss_wrong, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1 - alpha])  # original:(float, int, long)
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)

#         logpt = F.log_softmax(input, dim=-1)  # dim=1 ??
#         logpt = logpt.gather(1, target.type(torch.int64))
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


