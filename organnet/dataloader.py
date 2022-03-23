from torch.utils.data import Dataset
import torchio as tio
import torch
import os


class MICCAI(Dataset):
    images = []
    labels = []

    channel_encode = 10

    def __init__(self, data_set: str):
        super(MICCAI, self).__init__()

        resize = tio.Resize((256, 256, 48))

        data_path = os.path.join('data', data_set, 'data_3D')

        for subject_code in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, subject_code)):
                patient_path = os.path.join(data_path, subject_code)
                files = os.listdir(patient_path)
                patient_data = None
                label = None
                for file in files:
                    if 'img' in file and 'mha' in file and 'resampled' not in file:
                        patient_data = file

                    if 'mask' in file and 'resampled' not in file:
                        label = file
                label = resize(tio.LabelMap(os.path.join(patient_path, label), type=tio.LABEL))
                label = label.tensor.type(torch.LongTensor)
                self.labels.append(label)
                self.images.append(resize(tio.ScalarImage(os.path.join(patient_path, patient_data))).tensor.float())

    def __len__(self):
        return len(self.labels)

    def _encode(self, tensor):
        result = torch.empty(10, 256, 256, 48)
        for i in range(self.channel_encode):
            result[i] = torch.where(tensor == i, 1, 0)

        return result

    def __getitem__(self, index):
        return self.images[index], self._encode(self.labels[index])


if __name__ == "__main__":
    a = MICCAI('train_additional')

    for i in a:

        print('')





