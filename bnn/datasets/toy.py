
from utils import worker_init_reset_seed
import torch
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as torchdata
from sklearn.datasets import make_moons

import random 

random.seed(42)
np.random.seed(42)

class ToyRegression(torchdata.Dataset):
    def __init__(self, size, train=True):
        if train:
            # x = np.random.uniform(-0.1, 0.61, size=size)
            # noise = np.random.normal(0, 0.02, size=size)
            # y = x + 0.3 * np.sin( 2 * np.pi * (x+noise)) + 0.3 * np.sin( 4 * np.pi * (x+noise)) + noise
            # x[x <= 0] -= 2
            # x[x > 0] += 2
            # y = np.sin(x) + noise
            x = np.random.uniform(-4, 4, size=size)
            noise = np.random.normal(0, 3, size=size) #metric as mentioned in the paper
            y = (x) ** 3 + noise
        else:
            # x = np.linspace(-0.5, 1, size)
            # y = x + 0.3 * np.sin( 2 * np.pi * x) + 0.3 * np.sin( 4 * np.pi * x)
            x = np.linspace(-5, 5, size)
            y = (x) ** 3
            # y = np.sin(x)
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
            'targets': self.toTensor(self.y[index]).unsqueeze(0)
        }
        return data

class ToyClassification(torchdata.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def toTensor(self, x):
        return torch.from_numpy(np.asarray(x)).type(torch.FloatTensor)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        data = {
            'inputs': self.toTensor(self.x[index]).unsqueeze(0),
            'targets': self.toTensor(self.y[index]).unsqueeze(0)
        }
        return data

def build_toy_loaders(cfg):
    num_workers = cfg.num_workers
    batch_size = cfg.batch_size
    data_info = cfg.data_info

    if cfg.task == 'regression':
        train_dataset = ToyRegression(size=2000, train=True)
        valid_dataset = ToyRegression(size=50, train=False)
    else:
        X, Y = make_moons(n_samples=data_info['total_size'], noise=0.1, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=data_info['train_size'], random_state=42)
        train_dataset = ToyClassification(x=x_train,
                                        y=y_train)

        valid_dataset = ToyClassification(x=x_test,
                                        y=y_test)
        
    train_sampler = torchdata.SequentialSampler(train_dataset)
    valid_sampler = torchdata.SequentialSampler(valid_dataset)

    train_loader = torchdata.DataLoader(
                        train_dataset, 
                        batch_size, 
                        sampler=train_sampler,
                        num_workers=num_workers,
                        worker_init_fn=worker_init_reset_seed,
                        drop_last=False
                        )

    valid_loader = torchdata.DataLoader(
                        valid_dataset,
                        batch_size,
                        sampler=valid_sampler,
                        num_workers=num_workers,
                        )

    return train_loader, valid_loader
