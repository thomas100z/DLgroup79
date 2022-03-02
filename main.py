from torch.utils.data import DataLoader
import torch.optim as optim
from dataloader import get_data
from model import OrganNet
import torch

EPOCH = 1000

# get the data from the dataloader
training_data, test_data = get_data()
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# OrganNet model
net = OrganNet()

# paper: adam 0.001 initial, reduced by factor 10 every 50 epoch
optimizer = optim.Adam(net.parameters(), lr=0.001)

# focal loss + dice loss
loss = None
losses = []
val_losses = []

# train model on train set
for epoch in range(EPOCH):

    for train_sample in train_dataloader:
        pass




# evaluate the model on the test set
with torch.no_grad():
    for test_sample in test_dataloader:
        pass