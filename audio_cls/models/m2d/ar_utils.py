# ref: https://github.com/nttcslab/eval-audio-repr/blob/main/evar/ar_base.py
# ref: https://github.com/nttcslab/eval-audio-repr/blob/main/evar/utils/calculations.py

import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RunningMean:
    """Running mean calculator for arbitrary axis configuration.
    Thanks to https://math.stackexchange.com/questions/106700/incremental-averageing
    """

    def __init__(self, axis):
        self.n = 0
        self.axis = axis

    def put(self, x):
        if self.n == 0:
            self.mu = x.mean(self.axis, keepdims=True)
        else:
            self.mu += (x.mean(self.axis, keepdims=True) - self.mu) / self.n
        self.n += 1

    def __call__(self):
        return self.mu

    def __len__(self):
        return self.n


class RunningVariance:
    """Calculate mean/variance of tensors online.
    Thanks to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, axis, mean):
        self.update_mean(mean)
        self.s2 = RunningMean(axis)

    def update_mean(self, mean):
        self.mean = mean

    def put(self, x):
        self.s2.put((x - self.mean) **2)

    def __call__(self):
        return self.s2()

    def std(self):
        return np.sqrt(self())


class RunningStats:
    def __init__(self, axis=None):
        self.axis = axis
        self.mean = self.var = None

    def put(self, x):
        assert type(x)
        if self.mean is None:
            if self.axis is None:
                self.axis = list(range(len(x.shape)))
            self.mean = RunningMean(self.axis)
            self.var = RunningVariance(self.axis, 0)
        self.mean.put(x)
        self.var.update_mean(self.mean())
        self.var.put(x)

    def __call__(self):
        return self.mean(), self.var.std()

def _calculate_stats(device, data_loader, converter, max_samples):
    running_stats = RunningStats()
    sample_count = 0
    for batch in data_loader:
        with torch.no_grad():
            converteds = converter(batch['waveform'].to(device)).detach().cpu()
        running_stats.put(converteds)
        sample_count += len(batch)
        if sample_count >= max_samples:
            break
    return torch.tensor(running_stats())


def calculate_norm_stats(device, data_loader, converter, max_samples=5000):
    norm_stats = _calculate_stats(device, data_loader, converter, max_samples)
    logging.info(f' using spectrogram norimalization stats: {norm_stats.numpy()}')
    return norm_stats


def normalize_spectrogram(norm_stats, spectrograms):
    mu, sigma = norm_stats
    spectrograms = (spectrograms - mu) / sigma
    return spectrograms