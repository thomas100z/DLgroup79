from torch.utils.data import Dataset
import torchio as tio
import torch
import os
import pickle


class MICCAI(Dataset):
    images = []
    labels = []
    patients = []
    channel_encode = 10

    def __init__(self, data_set: str, load: bool = False):
        super(MICCAI, self).__init__()

        if load:
            with open('data/' + data_set + 'images.pickle', 'rb') as handle:
                self.images = pickle.load(handle)

            with open('data/' + data_set + 'labels.pickle', 'rb') as handle:
                self.labels = pickle.load(handle)

            with open('data/' + data_set + 'patients.pickle', 'rb') as handle:
                self.patients = pickle.load(handle)

        else:

            resize = tio.Resize((256, 256, 48))
            norm = tio.RescaleIntensity(out_min_max=(0, 1))

            data_path = os.path.join('data', data_set, 'data_3D')

            for subject_code in os.listdir(data_path):
                if os.path.isdir(os.path.join(data_path, subject_code)):
                    patient_path = os.path.join(data_path, subject_code)
                    files = os.listdir(patient_path)
                    patient_data = None
                    label = None
                    for file in files:
                        if 'img' in file and 'mha' in file and 'resampled' in file:
                            patient_data = file

                        if 'mask' in file and 'resampled' in file:
                            label = file
                    label = resize(tio.LabelMap(os.path.join(patient_path, label), type=tio.LABEL))
                    label = label.tensor.type(torch.LongTensor)

                    image = resize(tio.ScalarImage(os.path.join(patient_path, patient_data)))
                    image = norm(image.tensor.float())
                    assert image.max() <= 1

                    self.labels.append(label)
                    self.images.append(image)
                    self.patients.append(patient_path.split('/')[-1])

            with open('data/' + data_set + 'images.pickle', 'wb') as handle:
                pickle.dump(self.images, handle)

            with open('data/' + data_set + 'labels.pickle', 'wb') as handle:
                pickle.dump(self.labels, handle)

            with open('data/' + data_set + 'patients.pickle', 'wb') as handle:
                pickle.dump(self.patients, handle)

    def __len__(self):
        return len(self.labels)

    def _encode(self, tensor):
        result = torch.empty(10, 256, 256, 48)
        for i in range(self.channel_encode):
            result[i] = torch.where(tensor == i, 1, 0)

        return result

    def __getitem__(self, index):
        return self.images[index], self._encode(self.labels[index]), self.patients[index]


if __name__ == "__main__":
    a = MICCAI('train', load=False)
