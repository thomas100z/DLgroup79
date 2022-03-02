import torch.nn as nn
import torch.nn.functional as F


class OrganNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(2, 2))




    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        return x



if __name__ == "__main__":
    net = OrganNet()