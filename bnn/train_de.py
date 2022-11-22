import torch
from utils import *
from models import build_model
from datasets import build_uci_loaders, build_mnist_loaders, build_toy_loaders
import os.path as osp
from utils import get_optimizer, get_scheduler
from utils import get_timestamp, setup_logger
from visualize import visualize_toy
# model returns KL and forward

CHECKPOINT_PATH = 'checkpoints'
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, cfg):
             
        if cfg.dataset == 'uci':
            self.train_loader, self.valid_loader = build_uci_loaders(cfg)
        elif cfg.dataset == 'mnist':
            self.train_loader, self.valid_loader = build_mnist_loaders(cfg)
        elif cfg.dataset == 'toy':
            self.train_loader, self.valid_loader = build_toy_loaders(cfg)
        else:
            raise Exception(f"Do not support {cfg.dataset} dataset")

        self.cfg = cfg

        self.exp_name = cfg.exp_name
        self.initialize_training_folders(True)
        self.dataset = cfg.dataset

        self.num_batches = len(self.train_loader)
        
        self.device = cfg.device
        
        self.model_list = self.create_models(cfg)
    
        self.num_epochs = cfg.num_epochs

        self.current_epoch = 0
        self.current_iter = 0
        
        self.task = cfg.task
        self.metric = get_metric(self.task)

        self.best_metric = {
            'model': None,
            'metric' : 100000000,
            'state_dict' : None
        }

        self.cfg = cfg

    def create_models(self, cfg):
        model_list = []
        for _ in range(cfg.model['n_models']):
            model = build_model(cfg.model)
            model_list.append(model)
        return model_list
    
    def create_optimizer(self, model, cfg):
        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        return optimizer, scheduler
        

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
        
        
    def train_one_model(self, model, optimizer, scheduler):
        self.current_epoch = 0
        while self.current_epoch < self.num_epochs:
            self.logger.info(f'Running epoch {self.current_epoch}/{self.num_epochs}')
            # if self.current_epoch > 0 and self.current_epoch % self.val_iterval == 0:

            #     val_metric, mean, std = self.validation() 
            #     if val_metric < self.best_metric['metric']:
            #         self.best_metric['metric'] = val_metric
            #         self.best_metric['mean_std'] = (mean, std)
            #         self.logger.info(f"Save model with best performance: {self.best_metric['metric']}")

            #         if self.dataset == 'toy':
            #             self.visualize_toy_dataset(mean, std)

            #     if self.task == 'regression':
            #         self.logger.info(f'Epoch: {self.current_epoch}/{self.num_epochs} -  Val MSE: {val_metric:.4f}')
            #     else:
            #         self.logger.info(f'Epoch: {self.current_epoch}/{self.num_epochs} -  Val Acc: {val_metric:.4f}')

            self.current_iter = 1
            self.train_one_epoch(model, optimizer)
            scheduler.step()
            self.current_epoch += 1
        return model

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

        save_name = osp.join(CHECKPOINT_PATH, self.exp_name, f'{self.exp_name}.png')
        visualize_toy(x, 
                    y, 
                    x_test, 
                    y_test,
                    mean,
                    std,
                    save_name)
    
    def train_one_epoch(self, model, optimizer):
        model.train()
        hist_loss = 0
        for data in self.train_loader:
            data['targets'] = data['targets'].type(torch.FloatTensor)

            data['inputs'] = data['inputs'].to(self.device)
            data['targets'] = data['targets'].to(self.device)
            
            
            nll_loss, _ = model(data['inputs'], data['targets']) 
            hist_loss += nll_loss.item() 
            nll_loss.backward()
            optimizer.step()
            # if self.current_iter % 20 == 0:
            #     self.logger.info(f'step: {self.current_iter}/{self.num_batches} - \
            #         NLL: {nll_loss.item():.4f}')
            self.current_iter += 1
        hist_loss = hist_loss / len(self.train_loader)
        if self.current_epoch % 5 == 0:
            self.logger.info(f'Epoch: {self.current_epoch}/{self.num_epochs} - NLL: {hist_loss:.4f}')

    @torch.no_grad()
    def validation(self, model):
        model.eval()
        targets = []
        outputs = [] 
        for data in self.valid_loader:
            data['inputs'] = data['inputs'].to(self.device)
            data['targets'] = data['targets'].type(torch.FloatTensor).to(self.device)
            _, preds = model(data['inputs'], data['targets'])
            preds = torch.squeeze(preds, dim=-1)

            outputs.append(preds)
            targets.append(data['targets'])
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)
        targets = torch.cat(targets, dim=0)

        return outputs, targets
    
    def train_models(self):
        for idx, model in enumerate(self.model_list):
            self.logger.info(f"Training model {idx}")
            optimizer, scheduler = self.create_optimizer(model, self.cfg)
            model = self.train_one_model(model, optimizer, scheduler)
            self.model_list[idx] = model
        self.ensemble()

    def ensemble(self):
        final_outputs = []
        for idx, model in enumerate(self.model_list):
            if idx == 0:
                model_outputs, targets = self.validation(model)
            else:
                model_outputs, _ = self.validation(model)
            final_outputs.append(model_outputs)
        final_outputs = torch.cat(final_outputs, dim=0)
        mean = final_outputs.mean(0).cpu().numpy()
        std = final_outputs.std(0).cpu().numpy()
        # metric = (((mean - targets) / std) ** 2 + np.log(2 * np.pi * std)).mean()

        if self.dataset == 'toy':
            self.visualize_toy_dataset(mean, std)
        
        return mean, std

def main():
    cfg = read_config()
    trainer = Trainer(cfg)
    trainer.train_models()

if __name__ == '__main__':
    main()
    