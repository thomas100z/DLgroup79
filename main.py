import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from organnet.dataloader import MICCAI
from organnet.model import OrganNet
from datetime import datetime
from organnet.loss import FocalLoss, DiceLoss
import torch

EPOCH = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_CHANNEL = 10
LOAD_PATH = None #'models/29-11:33-OrganNet.pth'
ALPHA = torch.tensor([0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0]).reshape(1,10,1,1,1)
GAMMA = 2

# get the data from the dataloader, paper: batch size = 2
load_data_set = True if 'trainimages.pickle' in os.listdir('data') else False
training_data = MICCAI('train', load=load_data_set)
train_size = int(0.9 * len(training_data))
val_size = len(training_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(training_data, [train_size, val_size])
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

# OrganNet model
net = OrganNet(OUT_CHANNEL).to(DEVICE)

# paper: adam 0.001 initial, reduced by factor 10 every 50 epoch
optimizer = optim.Adam(net.parameters(), lr=0.001)

# optional restore checkpoint
if LOAD_PATH:
    net.load_checkpoint(LOAD_PATH, optimizer, 0.001)

# focal loss + dice loss
criterion_focal = FocalLoss(GAMMA, ALPHA)
criterion_dice = DiceLoss()

losses = []
val_losses = []
# train model on train set
try:
    for epoch in range(EPOCH):
        running_loss = 0.0
        validation_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss_dice = criterion_dice(outputs, labels)
            loss_focal = criterion_focal(outputs, labels)

            loss = loss_dice + loss_focal
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"[EPOCH {epoch + 1}] sample: ({i}/{len(train_dataloader)})\t"
                  f"combined loss: {loss.item()}\tloss_focal: {loss_focal.item()}\tloss_dice: {loss_dice.item()}")

        with torch.no_grad():
            for j, data in enumerate(validation_dataloader):
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                outputs = net(inputs)

                loss_dice = criterion_dice(outputs, labels)
                loss_focal = criterion_focal(outputs, labels)
                loss = loss_dice + loss_focal

                validation_loss += loss.item()

        losses.append(running_loss / len(train_dataloader))
        val_losses.append(validation_loss / len(validation_dataloader))

        # adjust the learning rate every 50 epochs according to the paper
        if epoch % 50 == 0 and epoch > 1:
            if optimizer.param_groups[0]['lr'] > 0.00001:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10

        print(f"[EPOCH {epoch + 1}] running loss: {running_loss / len(train_dataloader)}\t"
              f"validation loss: {validation_loss / len(validation_dataloader)}")

except KeyboardInterrupt:
    print("Terminating training")

# save the model
print("-------------------------------------------------------")
print('Finished Training')

now = datetime.now()

path = './models/' + now.strftime("%d-%H:%M") + "-OrganNet.pth"
net.save_checkpoint(optimizer, path)

print("Model saved")
print("-------------------------------------------------------")

plt.figure()
plt.plot(range(len(losses)), losses, 'r-')
plt.plot(range(len(val_losses)), val_losses, 'b-')
plt.grid()
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.show()
