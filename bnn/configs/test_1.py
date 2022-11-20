import os
abs_path = os.path.dirname(__file__)

cfg = {
    'model_path':'./weights/',
    'data_path':'./data/',

    'experiment_name': 'bnn_toys',
    'filter_warnings':True,

    'seed':101,
    'num_workers':4,
    'save_weights_only':False,

    'dataset': 'toy',

    'model': {
        'name': 'MLP',
        'flatten': True,
        'in_channels': 768,
        'layers': [100, 100, 10],
        'log_std': (0, -6),
        'mixture_weight': (1, 3),
    },


    'nll_loss': "mse", # cross entropy

    'kl_weight': 0.01,
    
    'optimizer': "ADAM",
    'weight_decay': 0.01,
    'weight_decay_bias': 0.0,
    'eps': 1e-6,
    'lr': 2.5e-4,
    'encoder_lr_rate': 1.0,
    'batch_size': 64,
    'max_epochs': 30,
}
