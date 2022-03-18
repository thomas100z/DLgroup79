from torch.utils.data import DataLoader
import torch.optim as optim
from organnet.dataloader import get_data
from organnet.focalLoss2 import FocalLoss2
from organnet.model import OrganNet
from datetime import datetime
from organnet.diceLoss import *
from organnet.focalLoss import *
import torch

EPOCH = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_CHANNEL = 1
LOAD_PATH = None #'./models/18-14:08OrganNet.pth'

# get the data from the dataloader, paper: batch size = 2
training_data, test_data = get_data()
train_size = int(0.9 * len(training_data))
val_size = len(training_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(training_data, [train_size, val_size])
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)

# OrganNet model
net = OrganNet(OUT_CHANNEL)
net.to(DEVICE)

# paper: adam 0.001 initial, reduced by factor 10 every 50 epoch
optimizer = optim.Adam(net.parameters(), lr=0.001)

# optional restore checkpoint
if LOAD_PATH:
    net.load_checkpoint(LOAD_PATH,optimizer, 0.001)

# focal loss + dice loss
criterion = torch.nn.CrossEntropyLoss()
criterion_focal = FocalLoss2()
criterion_dice = DiceLoss()
losses = []
val_losses = []

# train model on train set
for epoch in range(EPOCH):
    running_loss = 0.0
    validation_loss = 0.0

    for i, data in enumerate(train_dataloader):
        inputs, labels = data['t1']['data'].to(DEVICE), data['label']['data'].to(DEVICE)
        optimizer.zero_grad()
        labels = labels.type(torch.float)
        inputs = inputs.type(torch.float)
        outputs = net(inputs)

        loss_dice = criterion_dice(outputs.float(), labels.float())
        loss_focal = criterion_focal(outputs.float(), labels.float())

        loss = loss_dice + loss_focal
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"[EPOCH {epoch + 1}] sample: ({i}/{len(train_dataloader)})\tcombined loss: {loss.item()}\tloss_focal: {loss_focal.item()}\tloss_dice: {loss_dice.item()}")

    with torch.no_grad():
        for j, data in enumerate(validation_dataloader):
            inputs, labels = data['t1']['data'].to(DEVICE), data['label']['data'].to(DEVICE)
            labels = labels.type(torch.float)
            inputs = inputs.type(torch.float)
            outputs = net(inputs)

            loss_dice = criterion_dice(outputs.float(), labels.float())
            loss_focal = criterion_focal(outputs.float(), labels.float())
            loss = loss_dice + loss_focal

            validation_loss += loss.item()

    losses.append(running_loss/i)
    val_losses.append(validation_loss/j)

    print(f"[EPOCH {epoch + 1 }] running loss: {running_loss/i}\tvalidation loss: {validation_loss/j}")

# save the model
print("-------------------------------------------------------")
print('Finished Training')

now = datetime.now()

path = './models/' + now.strftime("%d-%H:%M") + "OrganNet.pth"
net.save_checkpoint(optimizer, path)

print("Model saved")
print("-------------------------------------------------------")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(len(losses)), losses, 'r-')
plt.plot(range(len(val_losses)), val_losses, 'b-')
plt.grid()
plt.xlabel('EPOCH')
plt.ylabel('LOSS')

# evaluate the model on the test set
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

print(f'TEST LOSS: {test_loss}')

plt.show()
