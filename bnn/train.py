import os
import os.path as osp
import logging

import numpy as np
import torch

from models import build_model
from datasets import build_uci_loaders, build_mnist_loaders, build_toy_loaders
from utils import get_timestamp, setup_logger, read_config
from utils import get_optimizer, get_scheduler
from utils import  get_metric
from visualize import visualize_toy

CHECKPOINT_PATH = 'checkpoints'
# from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, cfg):
             
        if cfg.dataset == 'uci':
            self.train_loader, self.valid_loader = build_uci_loaders(cfg)
        elif cfg.dataset == 'mnist':
            self.train_loader, self.valid_loader = build_mnist_loaders(cfg)
        elif cfg.dataset == 'toy':
            self.train_loader, self.valid_loader = build_toy_loaders(cfg)
        else:
            raise NotImplementedError(f"Do not support {cfg.dataset} dataset")

        self.exp_name = cfg.exp_name
        self.checkpoint_path = getattr(cfg, 'checkpoint_path', CHECKPOINT_PATH)
        self.initialize_training_folders(True)
        self.dataset = cfg.dataset

        self.num_batches = len(self.train_loader)
        
        self.device = cfg.device
        self.model = build_model(cfg.model)
        self.model.to(self.device)
    
        self.optimizer = get_optimizer(self.model, cfg)
        self.scheduler = get_scheduler(self.optimizer, cfg)
        
        self.kl_weight = cfg.kl_weight
        self.num_samples_train = cfg.num_samples_train
        self.num_samples_val = cfg.num_samples_val

        self.num_epochs = cfg.num_epochs
        self.num_warmup_epochs = cfg.num_warmup_epochs
        self.val_iterval = cfg.val_interval
        self.current_epoch = 1
        
        self.task = cfg.task
        self.metric = get_metric(self.task)

    def initialize_training_folders(self, from_scratch):
        exp_path = osp.join(self.checkpoint_path, self.exp_name)

        if from_scratch and osp.isdir(exp_path):
            timestamp = get_timestamp()
            os.rename(exp_path, osp.join(osp.dirname(exp_path), self.exp_name + '_' + timestamp))

        self.checkpoint_path = exp_path

        if from_scratch:
            os.makedirs(exp_path)
            os.makedirs(osp.join(exp_path, 'models'), exist_ok=True)
            os.makedirs(osp.join(exp_path, 'training_states'), exist_ok=True)

        # self.writer = SummaryWriter(log_dir=exp_path)

        setup_logger('base', exp_path, screen=True, tofile=True)
        self.logger = logging.getLogger('base')

    def get_state_dict(self):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
      
    def train(self):
        self.train_warmup()
        best_metric = 1e6

        while self.current_epoch <= self.num_epochs:
            self.logger.info(f'Running epoch {self.current_epoch}/{self.num_epochs}')
            self.train_one_epoch()

            if self.current_epoch % self.val_iterval == 0:
                val_metric, mean, std = self.validation() 
                if val_metric < best_metric:
                    best_metric = val_metric
                    best_checkpoint = self.get_state_dict()
                    self.logger.info(f"Achieve new best performance: {best_metric}")

                    if self.dataset == 'toy':
                        self.visualize_toy_dataset(mean, std)
                
                metric_name = 'NLL' if self.task == 'regression' else 'Acc'
                self.logger.info(f'Epoch: {self.current_epoch}/{self.num_epochs} -  Val {metric_name}: {val_metric:.4f}')
            
            self.scheduler.step()
            self.current_epoch += 1

        torch.save(best_checkpoint['model'], osp.join(self.checkpoint_path, 'models', 'best_model.pth'))

    def train_warmup(self):
        for epoch in range(self.num_warmup_epochs):
            total_loss = 0
            total_samples = 0
            for data in self.train_loader:
                data['inputs'] = data['inputs'].to(self.device)
                data['targets'] = data['targets'].to(self.device)

                self.model.zero_grad()
                loss = self.model.nll(data, 0)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_samples += len(data['inputs'])

            self.logger.info(f'Warmup epoch: {epoch}/{self.num_warmup_epochs} - Avg loss: {total_loss / total_samples:.4f}')
                

    def visualize_toy_dataset(self, mean, std):
        x = []
        y = []
        for data in self.train_loader:
            x.append(data['inputs'])
            y.append(data['targets'])
        x = torch.cat(x, dim=0).cpu().numpy()
        y = torch.cat(y, dim=0).cpu().numpy()

        x_test = []
        y_test = []

        for data in self.valid_loader:
            x_test.append(data['inputs'])
            y_test.append(data['targets'])

        x_test = torch.cat(x_test, dim=0).cpu().numpy()
        y_test = torch.cat(y_test, dim=0).cpu().numpy()

        save_name = osp.join(self.checkpoint_path, 'training_states', f'Epoch_{self.current_epoch}.png')
        visualize_toy(x.flatten(), 
                    y.flatten(), 
                    x_test.flatten(), 
                    y_test.flatten(),
                    mean.flatten(),
                    std.flatten(),
                    save_name)

    def train_one_epoch(self):
        self.model.train()
        current_iter = 1

        for data in self.train_loader:
            data['inputs'] = data['inputs'].to(self.device)
            data['targets'] = data['targets'].to(self.device)

            self.model.zero_grad()
            kl_div = self.model.KL(self.num_samples_train)
            nll = self.model.nll(data, self.num_samples_train)
            elbo =  self.kl_weight * kl_div + nll
            elbo.backward()
            
            if current_iter % 20 == 0:
                self.logger.info(f'step: {current_iter}/{self.num_batches} -  EBLO: {elbo.item():.4f} - KL: {kl_div.item():.4f} - NLL: {nll.item():.4f}')
            self.optimizer.step()
            current_iter += 1
            
    @torch.no_grad()
    def validation(self):
        self.model.eval()
        all_outputs = []
        targets = []

        for idx in range(self.num_samples_val):
            outputs = [] 
            for data in self.valid_loader:
                data['inputs'] = data['inputs'].to(self.device)
                data['targets'] = data['targets'].type(torch.float32).to(self.device)
                preds = self.model(data['inputs'])
                outputs.append(preds)
                if idx == 0:
                    targets.append(data['targets'])
            outputs = torch.cat(outputs, dim=0)
            all_outputs.append(outputs)

        all_outputs = torch.stack(all_outputs, dim=0) 
        targets = torch.cat(targets, dim=0)

        preds_mean = all_outputs.mean(dim=0).cpu().numpy()
        preds_std = all_outputs.std(dim=0).cpu().numpy()
        targets = targets.cpu().numpy()
        metric = self.metric(targets, preds_mean, preds_std)

        return metric, preds_mean, preds_std

def main():
    # args = parse_args()
    cfg = read_config()
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
    