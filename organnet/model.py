import torch.nn as nn
import torch
from torchsummary import summary
import os
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvResu2(nn.Module):
    def __init__(self, channel_in: int, channel_out: int, HDC: bool, reduction=4) -> None:
        super().__init__()

        self.channel_out = channel_out
        half_out = channel_in + (int((channel_out - channel_in) / 2))
        self.h = half_out
        if not HDC:
            self.conv_block = nn.Sequential(
                nn.Conv3d(channel_in, half_out, kernel_size=(3, 3, 3), padding='same'),
                nn.ReLU(),
                nn.BatchNorm3d(half_out),
                nn.Conv3d(half_out, channel_out, kernel_size=(3, 3, 3), padding='same'),
                nn.ReLU(),
            )

        else:
            self.conv_block = nn.Sequential(
                nn.Conv3d(channel_in, channel_out, kernel_size=(3, 3, 3), dilation=(3, 3, 3), padding='same'),
                nn.ReLU(),
            )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.pool_dense = nn.Sequential(
            nn.Linear(channel_out, channel_out // reduction),
            nn.ReLU(),
            nn.Linear(channel_out // reduction, channel_out),
            nn.Sigmoid()
        )

    def forward(self, x):

        x1 = self.conv_block(x)
        b, c, _, _, _ = x1.size()
        x2 = self.global_pool(x1).view(b, c)
        x2 = self.pool_dense(x2).view(b, c, 1, 1, 1)

        return torch.add(torch.mul(x1, x2), x1)


class OrganNet(nn.Module):

    def __init__(self, channel: int = 10) -> None:
        super().__init__()

        # 2xConv 1,3,3 : green arrows
        self.conv3d2_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 3, 3), padding='same'),
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3), padding='same')
        )

        self.conv3d2_2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding='same'),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding='same')
        )

        # pool and transpose layers : white arrows
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.transpose_2 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.transpose_1 = nn.ConvTranspose3d(32, 16, (1, 2, 2), stride=(1, 2, 2))

        # 2xConv 3,3,3 Resu and HDC
        self.convresu2_1 = ConvResu2(16, 32, False)
        self.convresu2_2 = ConvResu2(32, 64, False)
        self.hdc_1 = ConvResu2(64, 128, True)
        self.hdc_2 = ConvResu2(128, 256, True)
        self.convresu2_3 = ConvResu2(128, 64, False)
        self.convresu2_4 = ConvResu2(64, 32, False)

        # Conv 1 kernel
        self.conv1_1 = nn.Conv3d(256, 128, kernel_size=(1, 1, 1), padding='same')
        self.conv1_2 = nn.Conv3d(128, 64, kernel_size=(1, 1, 1), padding='same')
        self.conv1_3 = nn.Conv3d(32, channel, kernel_size=(1, 1, 1), padding='same')

        # HDC kernel
        self.hdc_2 = ConvResu2(128, 256, True)
        self.hdc_3 = ConvResu2(256, 128, True)

        # Softmax the output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        blue_concat = self.conv3d2_1(x)

        x = self.pool1(blue_concat)

        yellow_concat = self.convresu2_1(x)

        x = self.pool2(yellow_concat)

        green_concat = self.convresu2_2(x)
        orange_concat = self.hdc_1(green_concat)

        x = self.hdc_2(orange_concat)
        x = self.conv1_1(x)

        x = torch.cat((x, orange_concat), 1)
        x = self.hdc_3(x)

        x = self.conv1_2(x)

        x = torch.cat((green_concat, x), 1)
        x = self.convresu2_3(x)

        x = self.transpose_2(x)

        x = torch.cat((yellow_concat, x), 1)
        x = self.convresu2_4(x)

        x = self.transpose_1(x)

        x = torch.cat((blue_concat, x), 1)
        x = self.conv3d2_2(x)
        x = self.conv1_3(x)

        return self.softmax(x)

    def save_checkpoint(self, optimizer, filename=os.path.join('models', 'model_checkpoint.pth')):
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, checkpoint_file, optimizer, lr=0.0001):
        checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
        # print(checkpoint["state_dict"])
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == "__main__":
    net = OrganNet()
    summary(net, (1, 256, 256, 24))
