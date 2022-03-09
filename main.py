from torch.utils.data import DataLoader
import torch.optim as optim
from organnet.dataloader import get_data
from organnet.focalLoss2 import FocalLoss2
from organnet.model import OrganNet
from datetime import datetime
from organnet.loss import *
from organnet.diceLoss import *
from organnet.focalLoss import *
import torch

EPOCH = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get the data from the dataloader, paper: batch size = 2
training_data, test_data = get_data()
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)

# OrganNet model
net = OrganNet()
net.to(DEVICE)

# paper: adam 0.001 initial, reduced by factor 10 every 50 epoch
optimizer = optim.Adam(net.parameters(), lr=0.001)

# focal loss + dice loss
criterion_focal = FocalLoss2()
criterion_dice = DiceLoss()
losses = []
val_losses = []

# train model on train set
for epoch in range(EPOCH):
    running_loss = 0.0

    for i, data in enumerate(train_dataloader):
        inputs, labels = data['t1']['data'].to(DEVICE), data['label']['data'].to(DEVICE)
        optimizer.zero_grad()
        outputs = inputs
        print(outputs.shape, labels.shape)
        loss_dice = criterion_dice(outputs.float(), labels.float())
        loss_focal = criterion_focal(outputs.float(), labels.float())

        loss = loss_dice + loss_focal
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    losses.append(running_loss)
    print(f"[EPOCH {epoch}] loss: {running_loss}")

# save the model
print("-------------------------------------------------------")
print('Finished Training')

now = datetime.now()

PATH = './models/' + now.strftime("%d-%H:%M") + "OrganNet.pth"
torch.save(net.state_dict(), PATH)

print("Model saved")
print("-------------------------------------------------------")

# evaluate the model on the test set
with torch.no_grad():
    for test_sample in test_dataloader:
        inputs, labels = test_sample['t1']['data'].to(DEVICE), test_sample['label']['data'].to(DEVICE)

        print(f'input:{inputs.shape}\tlabel:{labels.shape}')
