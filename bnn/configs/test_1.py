cfg = {
    'exp_name': 'bnn_toys',
    'task': 'regression',

    'dataset': 'toy',
    'test_size': 0.2,

    'device': 'cpu',
    'model': {
        'type': 'bnn',
        'name': 'MLP',
        'layers': [1, 100, 1],
        'log_std': (0,),
        'mixture_weight': (1,),
        'nll_loss': "mse", # 'cross_entropy'
    },

    'num_epochs': 200,
    'num_warmup_epochs': 50,
    'num_workers': 8,
    'batch_size': 64,
    'val_interval': 1,
    
    'num_samples_train': 2,
    'num_samples_val': 20,
    'kl_weight': 10,

    'lr': 1e-3,
    'lr_decay': 150,
    'decay_type': 'step',
    'gamma': 0.1,
    'optimizer': "ADAM",
    'weight_decay': 0,
}
