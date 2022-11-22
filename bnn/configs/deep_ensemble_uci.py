cfg = {
    'model_path':'./weights/',
    'data_path':'./data/',

    'exp_name': 'deep_ensemble_uci',
    'device': 'cuda', 
    'filter_warnings': True,

    'seed':101,
    
    'save_weights_only':False,

    'dataset': 'uci',
    'batch_size': 64,
    'num_workers':4,
    'test_size': 0.2,

    'model': {
        'type': 'de',
        'name': 'MLP',
        'loss': 'mse',
        'n_models': 3,
        'act': 'relu',
        'use_bn': False,
        'flatten': True,
        'layers': [11, 50, 1],
    },

    'num_epochs': 200,
    'task': 'regression',
    'nll_loss': "mse", # cross entropy
    
    'optimizer': "ADAM",
    'decay_type': 'step',
    'lr_decay': 50,
    'gamma': 0.1, 
    'weight_decay': 0.01,
    'weight_decay_bias': 0.0,
    'eps': 1e-6,
    'lr': 1e-3,
    
}
