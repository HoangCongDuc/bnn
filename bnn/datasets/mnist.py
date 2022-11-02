import torch.utils.data as torchdata
from torchaudio import transforms
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from utils import worker_init_reset_seed
import torchvision.datasets as visdata

ROOT_MNIST = ''

def build_mnist_loaders(args):
    
    num_workers = args.num_workers
    batch_size = args.batch_size 

    train_dataset = visdata.MNIST(
                        ROOT_MNIST, 
                        True, 
                        transforms=None, 
                        download=True
                        )

    valid_dataset = visdata.MNIST(
                        ROOT_MNIST, 
                        True, 
                        transforms=None
                        )

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