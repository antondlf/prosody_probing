import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import pandas as pd

########
### Load feature labels
### Align labels with neural feat indices
### Generate splits for 10-fold CV
### create dataloader object

class BaseDataset(Dataset):
    """Base dataset for probing tasks."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the features.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        #self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample