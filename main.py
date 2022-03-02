from torch.utils.data import DataLoader
import torch.optim as optim
from dataloader import get_data
from model import OrganNet
from loss import *
import torch

EPOCH = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# get the data from the dataloader
training_data, test_data = get_data()
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# OrganNet model
net = OrganNet()

# paper: adam 0.001 initial, reduced by factor 10 every 50 epoch
optimizer = optim.Adam(net.parameters(), lr=0.001)

# focal loss + dice loss
focal_loss = None
losses = []
val_losses = []

# train model on train set
for epoch in range(EPOCH):
    running_loss = 0.0

    for i, data in enumerate(train_dataloader):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    losses.append(running_loss)
    print(f"[EPOCH {epoch}] loss: {running_loss}")




# evaluate the model on the test set
with torch.no_grad():
    for test_sample in test_dataloader:
        pass