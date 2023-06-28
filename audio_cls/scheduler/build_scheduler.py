import timm.scheduler
import torch.optim.lr_scheduler as lr_scheduler

from audio_cls.utils.registry import SCHEDULERS


@SCHEDULERS.register('LambdaLR')
def build_LambdaLRScheduler(optimizer, **kwargs):
    return lr_scheduler.LambdaLR(optimizer, **kwargs)

@SCHEDULERS.register('CosineLR')
def build_CosineLRScheduler(optimizer, **kwargs):
    return timm.scheduler.CosineLRScheduler(optimizer, **kwargs)