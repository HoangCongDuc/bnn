import torch
import torch.nn as nn
from utils import *
from tqdm import tqdm
from models import MLP
from datasets import build_uci_loaders, build_mnist_loaders
from torch.utils.data import DataLoader
import os.path as osp
from utils import get_optimizer, get_scheduler, parse_args
from utils import get_timestamp, setup_logger
# model returns KL and forward

CHECKPOINT_PATH = 'checkpoints'
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, args):
             
        if args.dataset == 'uci':
            self.train_loader, self.valid_loader = build_uci_loaders(args)
        elif args.dataset == 'mnist':
            self.train_loader, self.valid_loader = build_mnist_loaders(args)
        else:
            raise Exception(f"Do not support {args.dataset} dataset")

        self.exp_name = args.exp_name
        self.initialize_training_folders(True)

        self.num_batches = len(self.train_loader)

        self.model = MLP()
        
        self.optimizer = get_optimizer(self.model, args)
        self.scheduler = get_scheduler(self.optimizer, args)

        self.val_iterval = 2
        self.num_epochs = args.num_epochs
        self.num_samples = args.num_samples
        self.current_epoch = 0
        self.current_iter = 0
        
        self.device = torch.device('cuda')

        self.metric = get_metric(args.task)

    def initialize_training_folders(self, from_scratch):
        exp_path = osp.join(CHECKPOINT_PATH, self.exp_name)

        if from_scratch and osp.isdir(exp_path):
            timestamp = get_timestamp()
            os.rename(exp_path, osp.join(osp.dirname(exp_path), self.exp_name + '_' + timestamp))

        exp_path = osp.join(CHECKPOINT_PATH, self.exp_name)

        if from_scratch:
            os.makedirs(exp_path)
            os.makedirs(osp.join(exp_path, 'models'), exist_ok=True)
            os.makedirs(osp.join(exp_path, 'training_states'), exist_ok=True)

        self.writer = SummaryWriter(log_dir=exp_path)

        setup_logger('base', exp_path, screen=True, tofile=True)
        self.logger = logging.getLogger('base')
        
        
    def train(self):
        
        self.logger.info(f'Running epoch {self.current_epoch}/{self.num_epochs}')

        while self.current_epoch < self.num_epochs:
            if self.current_epoch % self.val_iterval == 0:
                self.validation() 
            self.current_iter = 0
            self.train_one_epoch()
            self.scheduler.step()
            self.current_epoch += 1
    
    def weight_kl(self):
        '''
            Calculate weight to reweight KL by batch index
            args:
                idx: current batch index
            returns:
                weight: weight of current batch
        '''
        weight = (2 ** (self.num_batches - self.current_iter)) / (2**(self.num_batches - 1))
        return weight
    
    def train_one_epoch(self):
        self.model.train()
        for data in tqdm(self.train_loader):
            
            print(data['inputs'])
            print(data['targets'])

            self.model.zero_grad()
            kl_div = self.model.KL()
            print(type(kl_div))
            nll = self.model.nll(data, self.num_samples)
            print(type(nll))
            # reweight loss by current weight of KL
            elbo = self.weight_kl() * kl_div + nll
            elbo.backward()
            self.optimizer.step()
            self.current_iter += 1
            
    def validation(self):
        self.model.eval()
        pass
            
def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
    