import torchio as tio
import os
from torch.utils.data import Dataset


def get_data() -> tuple[Dataset, Dataset]:
    subjects_list = []
    data_path = os.path.join('data', 'train', 'data_3D')
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

    transforms = [
    ]
    transform = tio.Compose(transforms)
    subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

    return subjects_dataset, subjects_dataset
