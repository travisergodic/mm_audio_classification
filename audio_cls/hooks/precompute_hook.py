import logging

from audio_cls.hooks.base_hook import BaseHook
from audio_cls.utils.registry import HOOKS


logger = logging.getLogger(__name__)


@HOOKS.register
class PrecomputeStatsHook(BaseHook):
    def set_up(self, trainer):
        logger.info('Precompute train_loader statistics.')
        trainer.audio_model.ar.precompute(trainer.device, trainer.train_loader)