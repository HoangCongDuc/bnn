import torch
import torch.nn as nn
import torch.functional as F
from layers import BayesianLinear

class BayesianNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.l1 = BayesianLinear(28*28, 400)
        self.l2 = BayesianLinear(400, 400)
        self.l3 = BayesianLinear(400, 10)
    
    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x
    
    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior
    
    def sample_elbo(self, input, target, samples=None):
        outputs = torch.zeros(samples, self.cfg.batch_size, self.cfg.classes).to(self.cfg.device)
        log_priors = torch.zeros(samples).to(self.cfg.device)
        log_variational_posteriors = torch.zeros(samples).to(self.cfg.device)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior)/self.cfg.num_batches + negative_log_likelihood
        
        return loss, log_prior, log_variational_posterior, negative_log_likelihood