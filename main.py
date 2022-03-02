from torch.utils.data import DataLoader
from dataloader import get_data

training_data, test_data = getData()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
