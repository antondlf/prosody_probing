import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
#from torch import io
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from ast import literal_eval
"""Run this file as a script to test the dataset loading and make sure all files load correctly.
Only one locally stored forward pass layer of the tested corpus is necessary to test. Watch for float
writing and reading issues."""
########
### Load feature labels
### Align labels with neural feat indices
### Generate splits for 10-fold CV?


class BaseDataset(Dataset):
    """Base dataset for probing tasks."""

    def __init__(self, csv_file, root_dir, tensor=False, transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the features.
            tensor (bool): Whether to return features as tensor or np array.
            tranform (func): transform functiont to apply to features if necessary.
            target_transform (func): transform to apply to label set (usually mapping to some int from strings)
        """
        self.transform = transform
        self.target_transform = target_transform
        self.annotations = csv_file
        self.labels = pd.read_csv(csv_file)#, dtype={'start_end_indices': list})
        self.labels['start_end_indices'] = self.labels.start_end_indices.map(literal_eval)
        self.labels = self.labels.explode('start_end_indices', ignore_index=True).dropna().reset_index()
        self.file_list = self.labels.file_id.unique()
        self.features = np.concatenate([np.load(root_dir / f'{file}.npy') for file in self.file_list], axis=0)
        
        if tensor:
            self.features = torch.from_numpy(self.features)
        #self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        sample = self.labels.iloc[idx]
        label = sample.label.item()
        feat = self.features[idx, :]
        
        if self.transform:
            feat = self.transform(feat)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        
        return feat, label
    

    def test_data_loading(self, root_dir):
        counter = 0
        for file in tqdm(self.file_list):
            counter += 1
            file_subset = self.labels.loc[self.labels.file_id == file]
            
            dataloader_feats = self.features[file_subset.index, :]
            print(self.features.shape, file_subset.index) 
            check_feats = np.load(root_dir / f'{file}.npy')
            check_csv = pd.read_csv('data/switchboard/phonwords/sw4033B_t44.csv')
            print(f"Identity of feats is {(dataloader_feats[:-2, :] == check_feats).all()}")
             
            if not dataloader_feats == check_feats:
                raise AssertionError(f"{file} has thrown an error, it was file no. {counter}. It's loader shape was {dataloader_feats.shape}, its programatic shape was {check_feats.shape}")
            
        
        print('All files processed correctly')
    

def main():
    
    annotations = Path('data/switchboard/aligned_tasks/phones_accents.csv')
    feat_root = Path('data/feats/switchboard/wav2vec2-base/layer-1') 
    print('Loading Dataset...')
    timestart = time.time()
    dataset = BaseDataset(annotations, feat_root)
    timeend = time.time()
    print(f"loading took {timeend-timestart}")
    print('Testing Dataset...')
    print(f"dataset len is {len(dataset)}")
    dataset.test_data_loading(feat_root)
    
            
        

if __name__ == '__main__':
    main()        