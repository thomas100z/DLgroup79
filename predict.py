import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from organnet.dataloader import MICCAI
from organnet.model import OrganNet
from organnet.loss import FocalLoss, DiceLoss
import torch

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


# def dice_coef(y_pred,y_true):
#     smooth = 1.0
#     y_true_f = torch.flatten(y_true)
#     y_pred_f = torch.flatten(torch.argmax(y_pred, axis=1))
#     intersection = torch.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

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

            # for channel in data:
            #     print(channel.shape)

            import plotly.graph_objects as go
            import numpy as np

            # values = data
            # print(values)
            # print("jo1", values[0].flatten())
            # print("jo2", values[:, 1].flatten())
            # print("jo3", values[:, 2].flatten())
            # print("jo4", values[:, 3].flatten())
            # print("jo5", values[:, 4].flatten())

            # fig = go.Figure(data=go.Volume(
            #     x=values[:, 1].flatten(),
            #     y=values[:, 2].flatten(),
            #     z=values[:, 3].flatten(),
            #     value=values[:, 0].flatten(),
            #     isomin=0.1,
            #     isomax=0.8,
            #     opacity=0.1,  # needs to be small to see through all surfaces
            #     surface_count=17,  # needs to be a large number for good volume rendering
            # ))
            # fig.show()

    print(f'TEST LOSS: {test_loss}')
