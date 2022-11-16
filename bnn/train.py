import torch
import torch.nn as nn
from utils import *
from tqdm import tqdm
from models import MLP
from datasets import build_uci_loaders, build_mnist_loaders, build_toy_loaders
from torch.utils.data import DataLoader
import os.path as osp
from utils import get_optimizer, get_scheduler, parse_args
from utils import get_timestamp, setup_logger
from visualize import visualize_toy
# model returns KL and forward

CHECKPOINT_PATH = 'checkpoints'
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, args):
             
        if args.dataset == 'uci':
            self.train_loader, self.valid_loader = build_uci_loaders(args)
        elif args.dataset == 'mnist':
            self.train_loader, self.valid_loader = build_mnist_loaders(args)
        elif args.dataset == 'toy':
            self.train_loader, self.valid_loader = build_toy_loaders(args)
        else:
            raise Exception(f"Do not support {args.dataset} dataset")

        self.exp_name = args.exp_name
        self.initialize_training_folders(True)
        self.dataset = args.dataset

        self.num_batches = len(self.train_loader)
        
        self.device = args.device
        self.model = MLP(logstd=(0, -6), mixture_weights=(3, 1)).to(self.device)
        
    
        self.optimizer = get_optimizer(self.model, args)
        self.scheduler = get_scheduler(self.optimizer, args)
        
        self.kl_reweight = args.kl_reweight
        self.val_iterval = 2
        self.num_epochs = args.num_epochs
        self.num_samples = args.num_samples
        self.current_epoch = 0
        self.current_iter = 0
        
        self.task = args.task
        self.metric = get_metric(self.task)

        self.best_metric = {
            'metric' : 100000000,
            'state_dict' : None
        }

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
        while self.current_epoch < self.num_epochs:
            self.logger.info(f'Running epoch {self.current_epoch}/{self.num_epochs}')
            if self.current_epoch > 0 and self.current_epoch % self.val_iterval == 0:
                # self.validation() 
                val_metric, mean, std = self.validation() 
                if val_metric < self.best_metric['metric']:
                    self.best_metric['metric'] = val_metric
                    self.best_metric['mean_std'] = (mean, std)
                    self.logger.info(f"Save model with best performance: {self.best_metric['metric']}")
                
                if self.task == 'regression':
                    self.logger.info(f'Epoch: {self.current_epoch}/{self.num_epochs} -  Val MSE: {val_metric:.4f}')
                else:
                    self.logger.info(f'Epoch: {self.current_epoch}/{self.num_epochs} -  Val Acc: {val_metric:.4f}')
            self.current_iter = 1
            self.train_one_epoch()
            # self.scheduler.step()
            self.current_epoch += 1

        if self.dataset == 'toy':
            mean, std = self.best_metric['mean_std']
            
            self.visualize_toy_dataset(mean, std)
    
    def visualize_toy_dataset(self, mean, std):
        x = []
        y = []
        for data in self.train_loader:
            x.append(data['inputs'])
            y.append(data['targets'])
        x = torch.cat(x, dim=0).squeeze(1).cpu().numpy()
        y = torch.cat(y, dim=0).cpu().numpy()

        x_test = []
        y_test = []

        for data in self.valid_loader:
            x_test.append(data['inputs'])
            y_test.append(data['targets'])

        x_test = torch.cat(x_test, dim=0).squeeze(1).cpu().numpy()
        y_test = torch.cat(y_test, dim=0).cpu().numpy()

        save_name = self.exp_name + '.png'
        visualize_toy(x, 
                    y, 
                    x_test, 
                    y_test,
                    mean,
                    std,
                    save_name)

    def weight_kl(self):
        '''
            Calculate weight to reweight KL by batch index
            args:
                idx: current batch index
            returns:
                weight: weight of current batch
        '''
        weight = (2 ** (self.num_batches - self.current_iter)) / (2**self.num_batches - 1)
        return weight
    
    def train_one_epoch(self):
        self.model.train()
        for data in self.train_loader:
            data['targets'] = data['targets'].type(torch.FloatTensor)

            data['inputs'] = data['inputs'].to(self.device)
            data['targets'] = data['targets'].to(self.device)

            # reweight loss by current weight of KL
            if self.kl_reweight:
                weight_kl = self.weight_kl()
                # weight_kl = 1 / self.num_batches
            else:
                weight_kl = 1

            self.model.zero_grad()
            kl_div = self.model.KL()
            nll = self.model.nll(data, self.num_samples)
            elbo =  weight_kl * kl_div + nll
            elbo.backward()
            
            if self.current_iter % 20 == 0:
                self.logger.info(f'step: {self.current_iter}/{self.num_batches} -  EBLO: {elbo.item():.4f} - KL: {kl_div.item():.4f} - NLL: {nll.item():.4f}')
            self.optimizer.step()
            self.current_iter += 1
            
    # @torch.no_grad()        
    # def validation(self):
    #     self.model.eval()
    #     outputs = []
    #     targets = []
    #     for data in self.valid_loader:
    #         data['targets'] = data['targets'].type(torch.FloatTensor)

    #         data['inputs'] = data['inputs'].to(self.device)
    #         data['targets'] = data['targets'].to(self.device)

    #         targets.append(data['targets'])
    #         b_outputs = torch.zeros_like(data['targets'])
    #         for _ in range(self.num_samples):
    #             preds = self.model(data['inputs'])
    #             preds = torch.squeeze(preds, dim=-1)
    #             b_outputs += preds
    #         b_outputs = b_outputs / self.num_samples
    #         outputs.append(b_outputs)
    #     outputs = torch.cat(outputs, dim=0).cpu().numpy()
    #     targets = torch.cat(targets, dim=0).cpu().numpy()
    #     metric = self.metric(targets, outputs)
    #     return metric
    @torch.no_grad()
    def validation(self):
        self.model.eval()
        all_outputs = []
        targets = []

        for idx in range(self.num_samples):
            outputs = [] 
            for data in self.valid_loader:
                data['inputs'] = data['inputs'].to(self.device)
                data['targets'] = data['targets'].type(torch.FloatTensor).to(self.device)
                preds = self.model(data['inputs'])
                preds = torch.squeeze(preds, dim=-1)
                outputs.append(preds)
                if idx == 0:
                    targets.append(data['targets'])
            outputs = torch.cat(outputs, dim=0).unsqueeze(0)
            all_outputs.append(outputs)

        all_outputs = torch.cat(all_outputs, dim=0) 
        targets = torch.cat(targets, dim=0)
        preds_mean = all_outputs.mean(0).cpu().numpy()
        preds_std = all_outputs.std(0).cpu().numpy()
        # calculate final outputs by mean of all outputs
        final_outputs = torch.sum(all_outputs, dim=0) / self.num_samples
        targets = targets.cpu().numpy()
        final_outputs = final_outputs.cpu().numpy()
        metric = self.metric(targets, final_outputs)

        return metric, preds_mean, preds_std

def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
    