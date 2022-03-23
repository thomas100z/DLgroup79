import matplotlib.pyplot as plt
from organnet.dataloader import get_data
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from organnet.dataloader import get_data
from organnet.focalLoss2 import FocalLoss2
from organnet.model import OrganNet
from organnet.diceLoss import *
from organnet.focalLoss import *

# load model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOAD_PATH = './models/18-14:08OrganNet.pth'

net = OrganNet(1)
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.load_checkpoint(LOAD_PATH, optimizer, 0.001)

# load data
training_data, test_data = get_data()
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# focal loss + dice loss
criterion_focal = FocalLoss2()
criterion_dice = DiceLoss()
losses = []
val_losses = []

with torch.no_grad():
    test_loss = 0
    with torch.no_grad():
        for test_sample in test_dataloader:
            inputs, labels = test_sample['t1']['data'].to(DEVICE), test_sample['label']['data'].to(DEVICE)
            labels = labels.type(torch.float)
            inputs = inputs.type(torch.float)
            outputs = net(inputs)

            loss_dice = criterion_dice(outputs.float(), labels.float())
            loss_focal = criterion_focal(outputs.float(), labels.float())
            loss = loss_dice + loss_focal

            test_loss += loss.item()

            _ , _ , h , w , d = labels.size()
            data = labels.view(h,w,d)

            import plotly.graph_objects as go
            import numpy as np

            values = data

            fig = go.Figure(data=go.Volume(
                x=data[:, 0].flatten(),
                y=data[:, 1].flatten(),
                z=data[:, 2].flatten(),
                value=data.flatten(),
                isomin=0.1,
                isomax=0.8,
                opacity=0.1,  # needs to be small to see through all surfaces
                surface_count=17,  # needs to be a large number for good volume rendering
            ))
            fig.show()




    print(f'TEST LOSS: {test_loss}')


