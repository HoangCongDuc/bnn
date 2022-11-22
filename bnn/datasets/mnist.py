import torch.utils.data as torchdata
from torchaudio import transforms
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from utils import worker_init_reset_seed
import torchvision.datasets as visdata

ROOT_MNIST = ''

class MNIST_base(visdata.MNIST):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

class CustomMNIST(MNIST_base):
    def __init__(self, cfg, train=True):
        self.mnist_base = MNIST_base(root=cfg.data_path,
                                        train=train,
                                        download=cfg.download,
                                        transform=cfg.transform)

    def __len__(self):
        return len(self.mnist_base)

    def __getitem__(self, index):
        data, target = self.mnist_base[index]        
        return {'inputs' : data, 
                'targets': target}

    
def build_mnist_loaders(cfg):
    
    

    num_workers = cfg.num_workers
    batch_size = cfg.batch_size 

    train_dataset = CustomMNIST(cfg,train=True)

    valid_dataset = CustomMNIST(cfg, train=False)

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