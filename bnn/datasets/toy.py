
from utils import worker_init_reset_seed

import os
from tkinter.messagebox import NO 
import torch
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as torchdata
from torchvision import transforms
from torch.autograd import Variable
import random 

random.seed(42)
np.random.seed(42)

class ToyRegression(torchdata.Dataset):
    def __init__(self, size, train=True):
        if train:
            x = np.random.uniform(-0.1, 0.61, size=size)
            noise = np.random.normal(0, 0.02, size=size) #metric as mentioned in the paper
            y = x + 0.3 * np.sin( 2 * np.pi * (x+noise)) + 0.3 * np.sin( 4 * np.pi * (x+noise)) + noise
        else:
            x = np.linspace(-0.5, 1, size)
            y = x + 0.3 * np.sin( 2 * np.pi * x) + 0.3 * np.sin( 4 * np.pi * x)
        self.x = x
        self.y = y

        # self.Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
    
    def toTensor(self, x):
        return torch.from_numpy(np.asarray(x)).type(torch.FloatTensor)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        data = {
            'inputs': self.toTensor(self.x[index]).unsqueeze(0),
            'targets': self.toTensor(self.y[index])
        }
        return data

def build_toy_loaders(args):
    num_workers = args.num_workers
    batch_size = args.batch_size

    train_dataset = ToyRegression(size=2000, train=True)
    valid_dataset = ToyRegression(size=50, train=False)
    train_sampler = torchdata.SequentialSampler(train_dataset)
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
