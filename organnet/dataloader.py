from torch.utils.data import Dataset
import torchio as tio
import os


class MICCAI(Dataset):
    images = []
    labels = []

    channel_encode = tio.transforms.OneHot(10)

    def __init__(self, data_set: str) -> None:
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

                self.labels.append(resize(tio.LabelMap(os.path.join(patient_path, label), type=tio.LABEL)))
                self.images.append(resize(tio.ScalarImage(os.path.join(patient_path, patient_data))))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index].tensor, self.channel_encode(self.labels[index]).tensor


if __name__ == "__main__":
    a = MICCAI('train_additional')
