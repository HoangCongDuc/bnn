from .model import MLP
from .model_de import MLP_de

def build_model(cfg):
    if cfg['name'] == 'MLP':
        model = MLP(cfg)
    elif cfg['name'] == 'MLP_de':
        model = MLP_de(cfg)
    else:
        raise NotImplementedError
    return model
    