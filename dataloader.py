from torch.utils.data import Dataset

class OrganDataset(Dataset):

    def __init__(self) -> None:
        super(OrganDataset, self).__init__()

        # load to class x,y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
