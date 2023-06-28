"""MSM-MAE wrapper for EVAR.
"""
import torch
import random
import logging
import torchaudio
import torch.nn as nn
import numpy as np
from easydict import EasyDict
from copy import deepcopy

from .runtime_audio import RuntimeM2D
from .ar_utils import calculate_norm_stats, normalize_spectrogram


logger = logging.getLogger(__name__)


class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.
    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward_one(self, lms):
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
                             .to(torch.float).to(lms.device))
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y+h, x:x+w] = lms
        # get random area
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i+h, j:j+w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = nn.functional.interpolate(crop.unsqueeze(0), size=lms.shape[-2:],
            mode=self.interpolation, align_corners=True).squeeze(0)
        return lms.to(torch.float)

    def forward(self, lms):
        if len(lms.shape) == 3:
            return self.forward_one(lms)
        for i in range(len(lms)):
            lms[i] = self.forward_one(lms[i])
        return lms

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
        format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))
        return format_string


class SpecAugment:
    @staticmethod
    def is_required(freqm, timem):
        if freqm > 0:
            return True
        if timem > 0:
            return True
        return False

    def __init__(self, freqm, timem):
        self.freqmask = torchaudio.transforms.FrequencyMasking(freqm) if freqm > 0 else None
        self.timemask = torchaudio.transforms.TimeMasking(timem) if timem > 0 else None

    def __call__(self, lms):
        if self.freqmask is not None:
            lms = self.freqmask(lms)
        if self.timemask is not None:
            lms = self.timemask(lms)
        return lms


class AudioFineuneAug:
    def __init__(self, freqm, timem, rrc=False):
        self.spec_aug = SpecAugment(freqm, timem) if SpecAugment.is_required(freqm, timem) else None
        self.rrc = RandomResizeCrop() if rrc else None
        if self.spec_aug is not None:
            logger.info(f' using SpecAugmentation with {freqm}, {timem}.')
        if self.rrc is not None:
            logger.info(f' using {self.rrc}')

    def __call__(self, lms):
        lms = lms if self.spec_aug is None else self.spec_aug(lms)
        lms = lms if self.rrc is None else self.rrc(lms)
        return lms


class AR_M2D_BatchNormStats(nn.Module):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.backbone = RuntimeM2D(cfg=cfg, weight_file=cfg.weight_file)
        self.backbone.eval()

    def encode_frames(self, batch_audio):
        with torch.no_grad():
            x = self.backbone.get_timestamp_embeddings(batch_audio)
        return x.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        with torch.no_grad():
            x = self.backbone.get_scene_embeddings(batch_audio)
        return x


class AR_M2D(nn.Module):
    def __init__(self, cfg, do_aug=True):
        super().__init__()
        self.runtime = RuntimeM2D(cfg=cfg, weight_file=cfg.weight_file)
        self.aug = AudioFineuneAug(cfg.ft_freq_mask, cfg.ft_time_mask, rrc=cfg.ft_rrc) if do_aug else None 
        self.register_buffer('norm_stats', torch.tensor([0., 0.]))

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.runtime.to_feature)

    def encode_frames(self, batch_audio):
        x = self.runtime.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x)

        if self.aug is not None and self.training:
            x = self.aug(x)
    
        # hidden_states = self.runtime.encode_lms(x, return_layers=True)
        # # stack layer outputs
        # states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in hidden_states]
        # features = torch.cat(states_to_stack, axis=-1)
        features = self.runtime.encode_lms(x, return_layers=False)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, hidden_dropout=0.5, mean=0.0, std=0.01, bias=0.):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        fcs = []
        for l, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            if l > 0:
                fcs.append(nn.Dropout(hidden_dropout))
            linear = nn.Linear(in_size, out_size)
            nn.init.normal_(linear.weight, mean=mean, std=std)
            nn.init.constant_(linear.bias, bias)
            fcs.append(linear)
            fcs.append(nn.ReLU())
        self.mlp = nn.Sequential(*fcs[:-1])

    def forward(self, x):
        out = self.mlp(x)
        return out


class TaskHead(nn.Module):
    def __init__(self, dim, n_class=1000, hidden=()):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, affine=False)
        self.mlp = MLP(input_size=dim, hidden_sizes=hidden, output_size=n_class, mean=0.0, std=0.01, bias=0.)

    def forward(self, x):
        x = self.norm(x.unsqueeze(-1)).squeeze(-1)
        return self.mlp(x)


class TaskNetwork(nn.Module):
    def __init__(self, cfg, ar, n_class):
        super().__init__()
        # self.cfg = EasyDict(cfg.copy())
        self.cfg = deepcopy(cfg)
        self.ar = ar
        # print(cfg.feature_d, cfg.runtime_cfg.hidden, cfg.runtime_cfg.n_class)
        # self.head = TaskHead(dim=cfg.feature_d, n_class=cfg.runtime_cfg.n_class, hidden=cfg.runtime_cfg.hidden)
        self.head = TaskHead(dim=cfg.feature_d, n_class=n_class, hidden=[])

        # print('Backbone representation:')
        # show_layers_trainable(self.ar, show_all_trainable=False)
        # print('Head:')
        # show_layers_trainable(self.head)

    def forward(self, batch_audio):
        x = self.ar(batch_audio)
        x = self.head(x)
        return x # returning logits, not probs