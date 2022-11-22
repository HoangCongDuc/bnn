from .model import MLP
from .model_de import SingleModel

def build_model(cfg):
    if cfg['type'] == 'bnn':
        if cfg['name'] == 'MLP':
            model = MLP(cfg)
    elif cfg['type'] == 'de':
        model = SingleModel(cfg)
    else:
        raise NotImplementedError
    return model
    