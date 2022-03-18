import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import models
from torchsummary import summary
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvResu2(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, HDC: bool, depth: int, reduction=16) -> None:
        super().__init__()

        self.channel_out = channel_out
        half_out = channel_in + (int((channel_out - channel_in) / 2))

        if not HDC:
            self.conv_block = nn.Sequential(
                nn.Conv3d(channel_in, half_out, kernel_size=(3, 3, 3), padding='valid'),
                nn.ReLU(),
                nn.BatchNorm3d(half_out),
                nn.Conv3d(half_out, channel_out, kernel_size=(3, 3, 3), padding='valid'),
                nn.ReLU(),
            )
            self.target_depth = depth - 4
            target_shape = (1, 1, self.target_depth)


        else:
            self.conv_block = nn.Sequential(
                nn.Conv3d(channel_in, channel_out, kernel_size=(3, 3, 3), dilation=(3, 3, 3), padding='valid'),
                nn.ReLU(),
            )
            self.target_depth = depth - 6
            target_shape = (1, 1, self.target_depth)

        self.pool_dense = nn.Sequential(
            nn.AdaptiveAvgPool3d(target_shape),
            nn.Linear(channel_out, channel_out // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_out // channel_out, channel_out, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv_block(x)
        x2 = self.pool_dense(x1)
        mat_mul = torch.add(torch.mul(x1, x2), x1)
        return mat_mul


class OrganNet(nn.Module):

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.SHAPE = __name__ == "__main__"

        # 2xConv 1,3,3 : green arrows
        self.conv3d2_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 3, 3), padding='valid'),
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3), padding='valid')
        )
        self.conv3d2_2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding='valid'),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding='valid')
        )

        # pool and transpose layers : white arrows
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=2)
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=2)
        self.transpose_2 = nn.ConvTranspose3d(64, 32, (2, 2, 2))
        self.transpose_1 = nn.ConvTranspose3d(32, 16, (1, 2, 2))

        # 2xConv 3,3,3 Resu and HDC
        depth = int((depth - 4 - 1))
        self.convresu2_1 = ConvResu2(16, 32, False, depth)
        depth = int((depth - 4 - 1))
        self.convresu2_2 = ConvResu2(32, 64, False, depth)
        depth = int((depth - 4))
        self.hdc_1 = ConvResu2(64, 128, True, depth)
        depth = int((depth - 6))
        self.hdc_2 = ConvResu2(128, 256, True, depth)
        depth = int((depth - 6))

        # self.convresu2_3 = ConvResu2(128, 64, False, depth)
        # depth =
        # self.convresu2_3 = ConvResu2(64, 32, False, depth)

        # Conv 1 kernel
        self.conv1_1 = nn.Conv3d(256, 128, kernel_size=(1, 1, 1), padding='valid')
        self.conv1_2 = nn.Conv3d(128, 64, kernel_size=(1, 1, 1), padding='valid')
        self.conv1_3 = nn.Conv3d(32, 25, kernel_size=(1, 1, 1), padding='valid')

        # HDC kernel
        hdc_depth = 57

        # self.hdc_2 = ConvResu2(128, 256, True, depth)
        # self.hdc_3 = ConvResu2(256, 128, True, depth)

    def forward(self, x):
        blue_concat = self.conv3d2_1(x)

        x = self.pool1(blue_concat)

        yellow_concat = self.convresu2_1(x)
        x = self.pool2(yellow_concat)

        green_concat = self.convresu2_2(x)

        orange_concat = self.hdc_1(green_concat)

        # x = self.hdc_2(orange_concat)

        # x = self.conv1_1(x)

        print(x.shape, orange_concat.shape)
        # x = torch.cat((x, orange_concat), 1)

        # x = self.hdc_3(x)
        #
        # x = self.conv1_2(x)
        #
        # x = torch.cat((green_concat, x), 1)
        #
        # x = self.convresu2_2(x)
        #
        # x = self.transpose_2(x)
        #
        # x = torch.cat((yellow_concat, x), 1)
        #
        # x = self.convresu2_3(x)
        #
        # x = self.transpose_1(x)
        #
        # x = torch.cat((blue_concat, x), 1)
        #
        # x = self.conv3d2_2(x)
        #
        # x = self.conv1_3(x)

        return x

    def save_checkpoint(self, optimizer, filename=os.path.join('models', 'model_checkpoint.pth')):
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
    net = OrganNet(48)
    #
    #
    # input_block = torch.ones((2, 1, 256, 256, 48))
    #
    # output_block = net(input_block)
    #
    # y = torch.mul(input_block, torch.rand(1, 1, 48))
    #
    # print('')
    z = 48
    # #
    summary(net, (1, 256, 256, z))

    # input_tensor = torch.rand((2, 1, 256, 256, 48))
    # output = net(input_tensor)
    #
    # print("----------------------------------------------------------------")
    # summary(net, (1, 256, 256, 48))

    # conv3d2_1 = nn.Sequential(
    #     nn.Conv3d(1, 8, kernel_size=(1, 3, 3), padding=1, stride=(1, 1, 1))
    #     # nn.Conv3d(8, 16, kernel_size=(1, 3, 3), padding=1)
    # )
    #
    # x = conv3d2_1(input_block)
    # print(x.shape)
