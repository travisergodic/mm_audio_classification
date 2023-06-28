
from typing import Any
import torch
import numpy as np


class PydueAugment:
    def __init__(self, gain_augment):
        self.gain_augment = gain_augment

    def __call__(self, waveform):
        if self.gain_augment:
            gain = torch.randint(self.gain_augment * 2, (1,)).item() - self.gain_augment
            amp = 10 ** (gain / 20)
            waveform = waveform * amp
        return waveform


class PadOrTruncate:
    def __init__(self, audio_length):
        self.audio_length = audio_length

    def __call__(self, waveform):
        """Pad all audio to specific length."""
        if len(waveform) <= self.audio_length:
            return np.concatenate(
                (waveform, np.zeros(self.audio_length - len(waveform), dtype=np.float32)), axis=0
            )
        return waveform[0: self.audio_length]
    

class NormalizeAudioDuration:
    def __init__(self, normailze_duration):
        self.normalize_duration = normailze_duration

    def __call__(self, waveform, sr):
        copy_num = int(self.normalize_duration * sr / waveform.shape[0])
        left_sample = self.normalize_duration * sr - copy_num * waveform.shape[0]
        return np.concatenate(
            (np.tile(waveform, copy_num), waveform[:left_sample]), axis=None
        )