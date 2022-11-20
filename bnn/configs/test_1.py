cfg = {
    'exp_name': 'bnn_toys',
    'task': 'regression',

    'dataset': 'toy',
    'test_size': 0.2,

    'num_epochs': 200,
    'num_workers': 8,
    'batch_size': 64,
    
    'num_samples_train': 2,
    'num_samples_val': 20,
    'kl_weight': 10,

    'device': 'cpu',
    'model': {
        'name': 'MLP',
        'layers': [1, 100, 1],
        'log_std': (0,),
        'mixture_weight': (1,),
        'nll_loss': "mse", # 'cross_entropy'
    },

    'lr': 1e-3,
    'lr_decay': 50,
    'decay_type': 'step',
    'gamma': 0.5,
    'optimizer': "ADAM",
    'weight_decay': 0,
}
