import os
import math
import random

import numpy as np
import torch
import torchaudio
import torch.nn.functional
from torch.utils.data import Dataset

from audio_cls.utils.registry import DATASETS


@DATASETS.register
class AudiosetDataset(Dataset):
    def __init__(
            self, audio_path_list, label_list, label_num, num_mel_bins, freqm, timem, 
            mixup, mean, std, skip_norm, noise, target_length, padding_mode='zero', **kwargs
        ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        """
        self.audio_path_list = audio_path_list
        self.label_list = label_list
        self.melbins = num_mel_bins
        self.freqm = freqm
        self.timem = timem 
        print('now using following mask: {:d} freq, {:d} time'.format(self.freqm, self.timem))
        self.mixup = mixup
        print('now using mix-up with rate {:f}'.format(self.mixup))
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = mean
        self.norm_std = std
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = skip_norm
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = noise
        if self.noise == True:
            print('now use noise augmentation')
        self.label_num = label_num
        print('number of classes is {:d}'.format(self.label_num))
        self.target_length = target_length
        self.padding_mode = padding_mode

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10
        )

        n_frames = fbank.shape[0]

        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            if self.padding_mode == 'zero':
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
                
            elif self.padding_mode == 'copy':
                copy_times = math.ceil(self.target_length / n_frames)
                fbank = fbank.tile((copy_times, 1))[:self.target_length, :]

        elif p < 0:
            fbank = fbank[0:self.target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.audio_path_list)-1)
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(self.audio_path_list[index], self.audio_path_list[mix_sample_idx] )
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            label_indices[self.label_list[index]] += mix_lambda
            # add sample 2 labels
            label_indices[self.label_list[mix_sample_idx]] += 1.0 - mix_lambda
        # if not do mixup
        else:
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(self.audio_path_list[index])
            label_indices[self.label_list[index]] = 1.0

        label_indices = torch.FloatTensor(label_indices)
        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze it back, it is just a trick to satisfy new torchaudio version
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return {"target": label_indices, "waveform": fbank, "name": os.path.basename(self.audio_path_list[index])}
        

    def __len__(self):
        return len(self.audio_path_list)