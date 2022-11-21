from types import SimpleNamespace
from copy import deepcopy
import numpy as np

import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
abs_path = os.path.dirname(__file__)
import cv2

cfg = {
    'model_path':'./weights/',
    'data_path':'./data/',

    'exp_name': 'deep_ensemble_toys',
    'device': 'cpu', 
    'filter_warnings': True,

    'seed':101,
    'num_workers':4,
    'save_weights_only':False,

    'dataset': 'toy',

    'model': {
        'name': 'MLP_de',
        'loss': 'mse',
        'n_models': 5,
        'act': 'relu',
        'flatten': True,
        'in_channels': 1,
        'layers': [50, 50,1],
    },

    'num_epochs': 10,
    'task': 'regression',
    'nll_loss': "mse", # cross entropy

    'kl_weight': 0.01,
    
    'optimizer': "ADAM",
    'decay_type': 'step',
    'lr_decay': 0.1,
    'gamma': 5, 
    'weight_decay': 0.01,
    'weight_decay_bias': 0.0,
    'eps': 1e-6,
    'lr': 5e-3,
    'encoder_lr_rate': 1.0,
    'batch_size': 32
}
