from operator import concat
import os 
import torch
import numpy as np
import pandas as pd
import glob

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torchvision import transforms

class WineDataset(Dataset):
    def __init__(self, root_dir, test_size=0.2, train=True, seed = 42) -> None:
        super().__init__()

        self.files = glob.glob(f"{root_dir}/*.csv")

        data_train = []
        data_test = []
        all_data = []

        for file in self.files:
            data = pd.read_csv(file)
            train, test = train_test_split(data, test_size=test_size, random_state=seed)
            all_data = data.append(data)
            data_train.append(train)
            data_test.append(test)
        
        all_data = pd.concat(all_data, ignore_index=True, sort=False)
        data_train = pd.concat(data_train, ignore_index=True, sort=False)
        data_test = pd.concat(data_test, ignore_index=True, sort=False)

        # normalized dataset 

        for col in data[:-1]:
            data_train[col] = (data_train[col] - data[col].mean()) / data[col].std()
            data_test[col] = (data_test[col] - data[col].mean()) / data[col].std()

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
        
        # input to tensor
        input = torch.Tensor(input)

        return {"input": input, "gt": label}
