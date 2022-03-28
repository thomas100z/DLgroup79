import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from organnet.dataloader import MICCAI
from organnet.model import OrganNet
from organnet.loss import FocalLoss, DiceLoss
import torch
import numpy as np

# load model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOAD_PATH = './models/23-21:15OrganNet.pth'

net = OrganNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.load_checkpoint(LOAD_PATH, optimizer, 0.001)

# load data
test_dataloader = DataLoader(MICCAI('train_additional'), batch_size=1, shuffle=True)

# focal loss + dice loss
criterion_focal = FocalLoss()
criterion_dice = DiceLoss()
losses = []
val_losses = []


def dice_score(inputs, targets):
    n, c, h, w, d = inputs.shape
    assert n == 1
    inputs = inputs.reshape((c, h, w, d))
    targets = targets.reshape((c, h, w, d))

    c_max_input = torch.argmax(inputs, 0)
    smooth = 1.0

    result = torch.empty(c, h, w, d)
    for i in range(c):
        result[i] = torch.where(c_max_input == i, 1, 0)

    intersection = torch.mul(inputs, targets).sum([1, 2, 3])
    dice = (2. * intersection) / (inputs.sum([1, 2, 3]) + targets.sum([1, 2, 3]) + smooth)

    return dice * 100

DSC = []

with torch.no_grad():
    test_loss = 0
    with torch.no_grad():
        for test_sample in test_dataloader:
            inputs, labels = test_sample[0].to(DEVICE), test_sample[1].to(DEVICE)
            labels = labels.type(torch.float)
            inputs = inputs.type(torch.float)
            outputs = net(inputs)
            loss_dice = criterion_dice(outputs.float(), labels.float())
            loss_focal = criterion_focal(outputs.float(), labels.float())
            loss = loss_dice + loss_focal

            test_loss += loss.item()

            _, c, h, w, d = labels.size()
            data = labels.view(c, h, w, d)
            dsc = net.dice_coef_multi_class(outputs.float(), labels.float())
            DSC.append(dsc)
            print(DSC)

    print(f'TEST LOSS: {test_loss}')
