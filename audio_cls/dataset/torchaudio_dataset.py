import os
import math

import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset

from audio_cls.utils.registry import DATASETS

    
@DATASETS.register
class TorchAudioDataset(Dataset):
    def __init__(self, audio_path_list, label_list, num_classes, unit_samples, random_crop, padding_mode=0, sample_rate=None, aug=None, **kwargs):
        self.audio_path_list = audio_path_list
        self.label_list = label_list
        self.num_classes = num_classes
        self.unit_samples = unit_samples
        self.random_crop = random_crop
        self.padding_mode = padding_mode
        self.sample_rate = sample_rate
        self.aug = aug

    def __getitem__(self, index):
        wav = self.get_audio(self.audio_path_list[index])

        # Trim or stuff padding
        l = wav.size(0)
        if l > self.unit_samples:
            start = np.random.randint(l - self.unit_samples) if self.random_crop else 0
            wav = wav[start:start + self.unit_samples]
 
        elif l < self.unit_samples:
            if isinstance(self.padding_mode, (float, int)):
                wav = F.pad(wav, (0, self.unit_samples - l), mode='constant', value=self.padding_mode)
            
            elif self.padding_mode == 'copy':
                copy_times = math.ceil(self.unit_samples / l)
                wav = wav.tile((copy_times,))[:self.unit_samples]
                
        wav = wav.to(torch.float)
        
        # augmentation
        if self.aug is not None:
            wav = self.aug(wav)

        return {
            'waveform': wav, 
            'target': self.get_label(self.label_list[index]),
            'name': os.path.basename(self.audio_path_list[index])
        }

    def get_audio(self, filename):
        wav, sr = torchaudio.load(filename)
        if sr != self.sample_rate:
            wav = AF.resample(wav, sr, self.sample_rate)
        return wav[0]
    
    def get_label(self, k):
        return F.one_hot(torch.tensor(k), self.num_classes).type(torch.FloatTensor)
    
    def __len__(self):
        return len(self.audio_path_list)