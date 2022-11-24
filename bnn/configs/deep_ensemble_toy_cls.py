cfg = {
    'model_path':'./weights/',
    'data_path':'./data/',

    'exp_name': 'deep_ensemble_toys_cls',
    'device': 'cpu', 
    'filter_warnings': True,

    # 'seed':101,
    'num_workers':4,
    'save_weights_only':False,

    'dataset': 'toy',
    'data_info':{
        'total_size': 300,
        'train_size': 200,
        'valid_size': 100,
    },

    'model': {
        'type': 'de',
        'name': 'MLP',
        'loss': 'bce',
        'n_models': 5,
        'use_bn': False,
        'act': 'relu',
        'flatten': True,
        'layers': [2, 150, 150, 1],
    },

    'num_epochs': 200,
    'task': 'classification',
    # 'nll_loss': "bce", # cross entropy
    
    'optimizer': "ADAM",
    'decay_type': 'step',
    'lr_decay': 100,
    'gamma': 0.1, 
    'weight_decay': 0.3,
    'weight_decay_bias': 0.0,
    'eps': 1e-8,
    'lr': 0.001,
    'batch_size': 200
}
