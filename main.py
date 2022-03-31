import os, sys
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
LOAD_PATH = sys.argv[1] if sys.argv[1] else None
ALPHA = torch.tensor([0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0]).reshape(1,10,1,1,1)
GAMMA = 2
BATCH_SIZE = 2

# get the data from the dataloader, paper: batch size = 2
load_train_set = True if 'train.pickle' in os.listdir('data') else False
training_data = MICCAI('train', load=load_train_set)

load_val_data = True if 'train_additional.pickle' in os.listdir('data') else False
validation_data = MICCAI('train_additional', load=load_val_data)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True)

# OrganNet model
net = OrganNet(OUT_CHANNEL).to(DEVICE)

# paper: adam 0.001 initial, reduced by factor 10 every 50 epoch
optimizer = optim.Adam(net.parameters(), lr=0.001)

# optional restore checkpoint
if LOAD_PATH:
    net.load_checkpoint(LOAD_PATH, optimizer, 0.001)

# focal loss + dice loss
criterion_focal = FocalLoss(GAMMA, ALPHA, shape=(BATCH_SIZE, *training_data[0][0].shape))
criterion_dice = DiceLoss(channels=OUT_CHANNEL)

losses = []
val_losses = []

# train model on train set
try:
    for epoch in range(EPOCH):
        running_loss = 0.0
        validation_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs, labels, patient = data[0].to(DEVICE), data[1].to(DEVICE), data[2]

            optimizer.zero_grad()
            outputs = net(inputs)

            loss_dice = criterion_dice(outputs, labels)
            loss_focal = criterion_focal(outputs, labels)

            loss = loss_dice + loss_focal
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f"[EPOCH {epoch + 1}] sample: ({patient})\t"
                  f"combined loss: {loss.item()}\tloss_focal: {loss_focal.item()}\tloss_dice: {loss_dice.item()}")

        with torch.no_grad():
            for j, data in enumerate(validation_dataloader):
                inputs, labels, patient = data[0].to(DEVICE), data[1].to(DEVICE), data[2]

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
