import torch.nn.functional as F

from audio_cls.utils.registry import TRAINERS
from audio_cls.trainer.base_trainer import BaseTrainer


@TRAINERS.register
class EfficientatTrainer(BaseTrainer):
    def forward(self, X):
        return self.audio_model(X)[0]


@TRAINERS.register
class HtsatTrainer(BaseTrainer):
    def forward(self, X):
        return self.audio_model.sed_model(X)['clipwise_output']


@TRAINERS.register
class NormalTrainer(BaseTrainer):
    def forward(self, X):
        return self.audio_model(X)


@TRAINERS.register
class PannTrainer(BaseTrainer):
    def forward(self, X, mix_lambda=None):
        return self.audio_model(X, mix_lambda)["clipwise_output"]


@TRAINERS.register
class SsastTrainer(BaseTrainer):
    def forward(self, X):
        return self.audio_model(X, task='ft_avgtok')