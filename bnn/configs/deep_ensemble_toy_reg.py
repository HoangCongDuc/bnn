cfg = {
    'model_path':'./weights/',
    'data_path':'./data/',

    'exp_name': 'deep_ensemble_toys_1_models',
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
        'n_models': 10,
        'act': 'relu',
        'flatten': True,
        'layers': [1, 100, 1],
    },

    'num_epochs': 200,
    'task': 'regression',
    'nll_loss': "mse", # cross entropy
    
    'optimizer': "ADAM",
    'decay_type': 'step',
    'lr_decay': 150,
    'gamma': 0.1, 
    'weight_decay': 0.01,
    'weight_decay_bias': 0.0,
    'eps': 1e-6,
    'lr': 1e-3,
    'batch_size': 64
}
