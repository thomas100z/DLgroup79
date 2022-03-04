import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import models
from torchsummary import summary

vgg = models.vgg16()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvResu2():
    pass


class OrganNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.SHAPE = __name__ == "__main__"

        self.conv1 = nn.Conv3d(1, 8, kernel_size=(1, 3, 3))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(1, 3, 3))

        self.pool1 = nn.MaxPool3d((1, 2, 2))

    def forward(self, x):
        if self.SHAPE: print(x.shape)
        x = self.conv1(x)
        if self.SHAPE: print(x.shape)
        x = self.conv2(x)
        blue_concat = x
        if self.SHAPE: print(x.shape)
        x = self.pool1(x)
        if self.SHAPE: print(x.shape)

        return x

    def save_checkpoint(self, optimizer, filename="my_checkpoint.pth"):
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, checkpoint_file, optimizer, lr):
        checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == "__main__":
    net = OrganNet()

    input_tensor = torch.rand((2, 1, 256, 256, 48))
    SHAPE = __name__ == "__main__"

    output = net(input_tensor)
    print("----------------------------------------------------------------")
    summary(net, (1, 256, 256, 48))
