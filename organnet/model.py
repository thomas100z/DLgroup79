import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import models
from torchsummary import summary

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvResu2(nn.Module):
    def __init__(self, channel_in: int, channel_out: int):
        super().__init__()

        half_out = channel_in + (int((channel_out - channel_in )/ 2))

        self.conv_block = nn.Sequential(
            nn.Conv3d(channel_in, half_out, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(half_out),
            nn.Conv3d(half_out, channel_out, kernel_size=(3, 3, 3)),
            nn.ReLU()
        )

        self.pool_dense = nn.Sequential(
            nn.AvgPool3d((1, 2, 2)),
            nn.Linear(1, 1),  # TODO
            nn.Linear(1, 1),  # TODO
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv_block(x)
        print(x1.shape)
        x2 = self.pool_dense(x1)
        print(x2.shape)
        return torch.add(torch.mul(x1, x2),x1)  # TODO


class OrganNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.SHAPE = __name__ == "__main__"

        # 2xConv 1,3,3 : green arrows
        self.conv3d2_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 3, 3)),
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3))
        )
        self.conv3d2_2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3)),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3))
        )

        # pool and transpose layers : white arrows
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.transpose = nn.ConvTranspose3d(64, 32, (2, 2, 2))
        self.transpose = nn.ConvTranspose3d(32, 16, (1, 2, 2))

        # 2xConv 3,3,3 Resu
        self.convresu2_1 = ConvResu2(16, 32)
        self.convresu2_2 = ConvResu2(32, 64)
        self.convresu2_3 = ConvResu2(128, 64)
        self.convresu2_3 = ConvResu2(64, 32)

        # Conv 1 kernel
        self.con1_1 = nn.Conv3d(256, 128, kernel_size=(1, 1, 1))
        self.conv1_2 = nn.Conv3d(128, 64, kernel_size=(1, 1, 1))
        self.conv1_3 = nn.Conv3d(32, 25, kernel_size=(1, 1, 1))

    def forward(self, x):

        x = self.conv3d2_1(x)

        if self.SHAPE: print(x.shape)

        x = self.pool1(x)

        if self.SHAPE: print(x.shape)

        x = self.convresu2_1(x)

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
    # net = OrganNet()
    block = ConvResu2(16, 32)

    input_block = torch.rand((2, 16, 256, 256, 48))
    output_block = block(input_block)

    # input_tensor = torch.rand((2, 1, 256, 256, 48))
    # output = net(input_tensor)
    #
    # print("----------------------------------------------------------------")
    # summary(net, (1, 256, 256, 48))
