import logging

import torch
import numpy as np

from audio_cls.hooks.base_hook import BaseHook
from audio_cls.utils.registry import HOOKS
from audio_cls.eval.metrics import METRICS
              

logger = logging.getLogger(__name__)


class BaseEvalHook(BaseHook):
    def reset_vars(self):
        self.test_losses = []
        self.preds = []
        self.gts = []

    def on_test_batch_end(self, trainer):
        self.test_losses.append(trainer.test_loss)
        self.preds.append(trainer.pred.argmax(dim=1))
        self.gts.append(trainer.y.argmax(dim=1))

    def on_test_epoch_end(self, trainer):
        loss = np.mean(self.test_losses)
        preds = torch.cat(self.preds).cpu().numpy()
        gts = torch.cat(self.gts).cpu().numpy()

        trainer.metric_dict = {
            metric_name: METRICS[metric_name](gts, preds)
            for metric_name in trainer.metric_names
        }
        trainer.metric_dict.update({"test_loss": loss})
        self.reset_vars()


@HOOKS.register
class EpochBasedLogTrainLossHook(BaseHook):
    def __init__(self, epoch_period=1):
        self.train_losses = []
        self.epoch_period = epoch_period

    def on_train_batch_end(self, trainer):
        self.train_losses.append(trainer.train_loss)

    def on_train_epoch_end(self, trainer):
        if trainer.epoch % self.epoch_period == 0: 
            logger.info(f"Epoch {trainer.epoch} train_loss: {np.mean(self.train_losses)}")    
            self.train_losses = []


@HOOKS.register
class IterBasedLogTrainLossHook(BaseHook):
    def __init__(self, iter_period=50):
        self.train_losses = []
        self.iter_period = iter_period

    def on_train_batch_end(self, trainer): 
        self.train_losses.append(trainer.train_loss)
        if trainer.current_iter % self.iter_period == 0:
            logger.info(f"Period {trainer.current_iter // self.iter_period} train_loss: {np.mean(self.train_losses)}")
            self.train_losses = []


@HOOKS.register
class EpochBasedEvalHook(BaseEvalHook):
    def __init__(self, epoch_period=1):
        self.reset_vars()
        self.epoch_period = epoch_period

    def on_train_epoch_end(self, trainer):
        if (trainer.epoch % self.epoch_period == 0) and (trainer.test_loader is not None):
            trainer.test(trainer.test_loader) 

    def on_test_epoch_end(self, trainer):
        super().on_test_epoch_end(trainer)
        logger.info(f"Epoch {trainer.epoch} metric_dict: {trainer.metric_dict}")



@HOOKS.register
class IterBasedEvalHook(BaseEvalHook):
    def __init__(self, iter_period):
        self.reset_vars()
        self.iter_period = iter_period

    def on_train_batch_end(self, trainer):
        if (trainer.current_iter % self.iter_period == 0) and (trainer.test_loader is not None):
            trainer.test(trainer.test_loader)

    def on_test_epoch_end(self, trainer):
        super().on_test_epoch_end(trainer)
        logger.info(f"Period {trainer.current_iter // self.iter_period} metric_dict: {trainer.metric_dict}")
    