import logging

import timm.scheduler

from audio_cls.hooks.base_hook import BaseHook
from audio_cls.utils.registry import HOOKS
    

logger = logging.getLogger(__name__)


@HOOKS.register
class WarmupHook(BaseHook):
    def __init__(self, lr, warmup_steps=100, period=50):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.period = period
        self.global_step = 0

    def on_train_batch_end(self, trainer):
        # first several steps for warm-up
        if (self.global_step <= self.warmup_steps) and (self.global_step % self.period == 0):
            warm_lr = (self.global_step / self.warmup_steps) * self.lr
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = warm_lr
            logger.info("warm-up learning rate is {:f}".format(trainer.optimizer.param_groups[0]["lr"]))
        self.global_step += 1


@HOOKS.register
class EpochSchedulerHook(BaseHook):
    def on_train_epoch_end(self, trainer):
        if trainer.epoch_scheduler is not None:
            trainer.epoch_scheduler.step()


@HOOKS.register
class IterSchedulerHook(BaseHook):
    def on_train_batch_end(self, trainer):
        if trainer.iter_scheduler is not None:
            trainer.iter_scheduler.step()


@HOOKS.register
class CosLRSchedulerHook(BaseHook):
    def set_up(self, trainer):
        if not isinstance(trainer.iter_scheduler, timm.scheduler.CosineLRScheduler):
            logger.warning(f'iter_scheduler is not {str(timm.scheduler.CosineLRScheduler)}!')

    def on_train_batch_end(self, trainer):
        if trainer.iter_scheduler is not None:
            trainer.iter_scheduler.step(trainer.epoch + trainer.iter / len(trainer.train_loader))
