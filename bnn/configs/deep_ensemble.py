cfg = {
    'model_path':'./weights/',
    'data_path':'./data/',

    'exp_name': 'deep_ensemble_toys_3_models',
    'device': 'cpu', 
    'filter_warnings': True,

    'seed':101,
    'num_workers':4,
    'save_weights_only':False,

    'dataset': 'toy',

    'model': {
        'type': 'de',
        'name': 'MLP',
        'loss': 'mse',
        'n_models': 5,
        'act': 'relu',
        'flatten': True,
        'layers': [1, 50, 50,1],
    },

    'num_epochs': 100,
    'task': 'regression',
    'nll_loss': "mse", # cross entropy

    'kl_weight': 0.01,
    
    'optimizer': "ADAM",
    'decay_type': 'step',
    'lr_decay': 50,
    'gamma': 0.5, 
    'weight_decay': 0.01,
    'weight_decay_bias': 0.0,
    'eps': 1e-6,
    'lr': 1e-4,
    'encoder_lr_rate': 1.0,
    'batch_size': 64
}
