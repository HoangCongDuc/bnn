from torchvision import transforms

cfg = {
    'model_path':'./weights/',
    'data_path':'/lclhome/cnguy049/projects/bnn/bnn/datasets',

    'exp_name': 'deep_ensemble_toys_1_models',
    'device': 'cuda', 
    'filter_warnings': True,

    'seed':101,
    'num_workers':4,
    'save_weights_only':False,

    'task': 'classification',
    'dataset': 'mnist',
    'download': True,
    'transform': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]),

    'model': {
        'type': 'de',
        'name': 'MLP',
        'loss': 'softmax',
        'n_models': 2,
        'act': 'relu',
        'use_bn': 'True',
        'flatten': True,
        'layers': [784, 200, 200, 200, 10],
    },

    'num_epochs': 2,
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
