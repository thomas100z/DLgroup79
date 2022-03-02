import torchio as tio
import os
from torch.utils.data import Dataset

transforms = [
    tio.Resize((256, 256, 48))
]


def get_data() -> tuple[Dataset, Dataset]:
    subjects_list = []
    subjects_dataset = {}

    for data_set in ['train', 'train_additional']:
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

                subjects_list.append(tio.Subject(
                    t1=tio.ScalarImage(os.path.join(patient_path, patient_data), ),
                    label=tio.LabelMap(os.path.join(patient_path, label), ),
                ))

        transform = tio.Compose(transforms)
        subjects_dataset[data_set] = tio.SubjectsDataset(subjects_list, transform=transform)

    return subjects_dataset['train'], subjects_dataset['train_additional']


if __name__ == "__main__":
    get_data()
