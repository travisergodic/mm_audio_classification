import torch.optim as optim

from audio_cls.utils.registry import OPTIMIZERS
from .sam import SAM


@OPTIMIZERS.register('Adam')
def build_Adam(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return optim.Adam(params, **kwargs)

@OPTIMIZERS.register('AdamW')
def build_AdamW(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return optim.AdamW(params, **kwargs)

@OPTIMIZERS.register('SGD')
def build_SGD(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return optim.SGD(params, **kwargs)

@OPTIMIZERS.register('SAM_Adam')
def build_SAM_Adam(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return SAM(params, optim.Adam, **kwargs)

@OPTIMIZERS.register('SAM_AdamW')
def build_SAM_AdamW(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return SAM(params, optim.AdamW, **kwargs)

@OPTIMIZERS.register('SAM_SGD')
def build_SAM_SGD(model, **kwargs):
    get_params = kwargs.pop('get_params')
    params = get_params(model)
    return SAM(params, optim.SGD, **kwargs)