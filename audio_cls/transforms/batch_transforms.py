import random
import logging

import torch
import numpy as np

from audio_cls.utils.registry import BATCH_TRANSFORMS
from audio_cls.transforms.preprocess import _mel_forward
from audio_cls.transforms.preprocess import AugmentMelSTFT


logger = logging.getLogger(__name__)


@BATCH_TRANSFORMS.register
class EfficientatMelTransform:
    def __init__(self, device, *args, **kwargs):
        self.device = device 
        self.mel = self.build_mel(*args, **kwargs).to(self.device)

    def __call__(self, X, y):
        """
        do mel transform
        """
        bs = X.size(0)
        X = X.reshape(bs, 1, -1)
        with torch.no_grad():
            X = _mel_forward(X, self.mel)
        return X, y    
    
    def build_mel(self, *args, **kwargs):
        return AugmentMelSTFT(*args, **kwargs)


@BATCH_TRANSFORMS.register
class MixupTransform(object):
    def __init__(self, mixup=0.5, mixup_alpha=0.5):
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        logger.info(f'Using mixup with alpha={mixup_alpha}')

    def __call__(self, X, y):
        if random.random() < self.mixup:
            self.get_lambda(X.size(0), X.device)
            return self.do_mixup(X, self.lambdas), self.do_mixup(y, self.lambdas)
        return X, y

    def get_lambda(self, batch_size, device):
        lambdas = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
        self.lambdas = torch.tensor(lambdas).to(torch.float).to(device)
        self.counter_indexes = np.random.permutation(batch_size)
    
    def do_mixup(self, x, mixup_lambda):
        x = x.transpose(0, -1)
        out = x * mixup_lambda + x[..., self.counter_indexes] * (1.0 - mixup_lambda)
        return out.transpose(0, -1)
