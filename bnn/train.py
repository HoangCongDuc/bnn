import torch
import torch.nn as nn
from tqdm import tqdm
from bnn.models import MLP, CNN


# model returns KL and forward

class Trainer:

    def __init__(self, args):        
        self.num_epochs = args.n_epochs
        
        self.train_dataloader = None
        self.num_batches = len(self.train_dataloader)

        self.model = None
        self.optimizer = None
        self.n_samples = None
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.train_one_epoch()
    
    def weight_kl(self, idx):
        '''
            Calculate weight to reweight KL by batch index
            args:
                idx: current batch index
            returns:
                weight: weight of current batch
        '''
        weight = (2 ** (self.num_batches - idx)) / (2**(self.num_batches-1))
        return weight
    
    def elbo(self, kl, nll, idx):
        '''
            Calculate elbo for one batch 
                loss = weight_kl * kl + nll
            args: 
                kl: kl divergence
                nll: negative loglikelihood

            returns:
                eblo: elbo loss
        '''
        weight_kl = self.weight_kl(idx)
        return weight_kl * kl + nll

    def train_one_epoch(self):
        self.net.train()
        for idx, data in enumerate(tqdm(self.train_dataloader)):
            self.net.zero_grad()
            kl_div = self.net(data)
            nll = self.net(data, self.n_samples)
            elbo = self.elbo(kl_div, nll, idx)
            elbo.backward()
            self.optimizer.step()
            
    def validation(self):
        pass

def main():
    pass

if __name__ == '__main__':
    main()
    