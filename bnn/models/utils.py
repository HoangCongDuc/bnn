from .model import MLP


def build_model(cfg):
    if cfg['name'] == 'MLP':
        model = MLP(cfg)
    else:
        raise NotImplementedError
    return model
    