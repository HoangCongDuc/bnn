import sys
sys.path.append("..") # Adds higher directory to python modules path.
from utils import worker_init_reset_seed

from operator import concat
import os
from tkinter.messagebox import NO 
import torch
import numpy as np
import pandas as pd
import glob

from sklearn.model_selection import train_test_split
import torch.utils.data as torchdata
from torchvision import transforms

ROOT_UCI = '/lclhome/cnguy049/projects/bnn/bnn/data/uci_wine'

class WineDataset(torchdata.Dataset):
    def __init__(self, root_dir, test_size=0.2, train=True, seed = 42) -> None:
        super().__init__()

        self.files = glob.glob(f"{root_dir}/*.csv")

        data_train = []
        data_test = []
        all_data = []

        for file in self.files:
            data = pd.read_csv(file, delimiter=';')
            all_data.append(data)

            train_df, test_df = train_test_split(data, test_size=test_size, random_state=seed)
            data_train.append(train_df)
            data_test.append(test_df)
        
        all_data = pd.concat(all_data, ignore_index=True, sort=False)
        data_train = pd.concat(data_train, ignore_index=True, sort=False)
        data_test = pd.concat(data_test, ignore_index=True, sort=False)

        print(f"Number of records: {len(all_data)}")
        print(f"Number of training data: {len(data_train)}")
        print(f"Number of validation data: {(len(data_test))}")

        # normalized dataset 
        
        for col in all_data.columns[:-1]:
            data_train[col] = (data_train[col] - all_data[col].mean()) / all_data[col].std()
            data_test[col] = (data_test[col] - all_data[col].mean()) / all_data[col].std()
        
        # print(train)

        if train:
            self.data = data_train
        else:
            self.data = data_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        item = self.data.iloc[idx]
        input = item[:-1].to_numpy()
        label = item[-1]
        
        input = torch.from_numpy(input).float()

        return {"inputs": input, "targets": label}


def build_uci_loaders(cfg):
    

    num_workers = cfg.num_workers
    batch_size = cfg.batch_size

    train_dataset = WineDataset(root_dir=ROOT_UCI, test_size=cfg.test_size, seed=cfg.seed)
    valid_dataset = WineDataset(root_dir=ROOT_UCI, test_size=cfg.test_size, train=False, seed=cfg.seed)

    train_sampler = torchdata.RandomSampler(train_dataset)
    valid_sampler = torchdata.SequentialSampler(valid_dataset)

    train_loader = torchdata.DataLoader(
                        train_dataset, 
                        batch_size, 
                        sampler=train_sampler,
                        num_workers=num_workers,
                        worker_init_fn=worker_init_reset_seed,
                        drop_last=True
                        )
    valid_loader = torchdata.DataLoader(
                        valid_dataset,
                        batch_size,
                        sampler=valid_sampler,
                        num_workers=num_workers,
                        )

    return train_loader, valid_loader