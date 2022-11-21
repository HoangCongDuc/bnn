import os
import logging
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch
import numpy as np
import random
from datetime import datetime
# from torchmetrics import MeanSquaredError, Accuracy
from sklearn.metrics import mean_squared_error, accuracy_score

import sys
import importlib
from types import SimpleNamespace
import argparse

def seed_all(seed=None):
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        print("Using a generated random seed {}".format(seed))
    # torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all(initial_seed + worker_id)


def get_scheduler(optimizer, args):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    logger = logging.getLogger('base')
    logger.info(f'Decay type: {args.decay_type}')

    return scheduler
    

def get_optimizer(model, args):
    for k, v in model.named_parameters():
        if not v.requires_grad:
            print(f'Warning: {k} will not be optimized')

    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    logger = logging.getLogger('base')
    logger.info(f'Optimizer function: {args.optimizer}')
    logger.info(f'Learning rate: {args.lr}')
    logger.info(f'Weight decay: {args.weight_decay}')

    return optimizer_function(trainable, **kwargs)

# def parse_args():
#     parser = argparse.ArgumentParser(description='Compression-Driven Frame Interpolation Training')

#     # parameters
#     # Model Selection
#     # parser.add_argument('--fe_name', type=str, required=True, help='choose between regression and classification')
#     # parser.add_argument('--task', type=str, required=True, help='Each task has its corresponding head architecture')

#     # Directory Setting
#     parser.add_argument('--dataset', type=str, required=True, help="dataset name: uci, mnist")
#     parser.add_argument('--exp_name', type=str, default="regression")
#     parser.add_argument('--task', type=str, default="regression")

#     # Learning Options
#     parser.add_argument('--num_epochs', type=int, default=100, help='Max Epochs')
#     parser.add_argument('--num_samples', type=int, default=100, help='Model Samples')
#     parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
#     parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
#     parser.add_argument('--test_size', type=float, default=0.2, help='Batch size')
#     parser.add_argument('--seed', type=int, help='seed for exp')
#     parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
#     # parser.add_argument('--img_log_freq', type=int, help='saving image frequency')

#     # Optimization specifications
#     parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
#     parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
#     parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
#     parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
#     parser.add_argument('--optimizer', default='SGD', choices=('SGD', 'ADAM', "RMSprop"), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
#     parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
#     parser.add_argument('--kl_reweight', action='store_true', help='reweight KL divergence')
#     parser.add_argument('--vars', nargs='+', help='create variances for prior')
#     # head tuning options
#     args = parser.parse_args()

#     return args

# setup logger

def read_config():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-C", "--config", help="config filename")
    parser_args = parser.parse_args()

    print("Using config file", parser_args.config)

    spec = importlib.util.spec_from_file_location('config', parser_args.config)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)

    cfg =  SimpleNamespace(**cfg_module.cfg)

    return cfg


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")

def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

# get metric


def get_metric(task_name):
    if task_name == 'regression':
        return mse
    else:
        return accuracy


def mse(targets, preds):
    return mean_squared_error(targets, preds)

def accuracy(targets, preds):
    return accuracy_score(targets, preds)

