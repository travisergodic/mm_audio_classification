import torch 
import bisect
from ml_collections import ConfigDict

from configs.base import *

sample_rate = 32000

# pre-transfrom & aug
pre_transform_cfg = None
aug_cfg = None

# dataset
train_dataset_cfg = {
    'type': 'TorchAudioDataset',
    'unit_samples': sample_rate * 3,
    'num_classes': num_classes,
    'random_crop': True,
    'padding_mode': 'copy', 
    'sample_rate': sample_rate
}

test_dataset_cfg = {
    'type': 'TorchAudioDataset',
    'unit_samples': sample_rate * 3,
    'num_classes': num_classes,
    'random_crop': False,
    'padding_mode': 'copy', 
    'sample_rate': sample_rate
}

# sampler
sampler_cfg = {'type': 'BalancedBatchSampler', 'batch_size': 20, 'num_iter': 40}

# dataloader
train_batch_size = 20
test_batch_size = 40
num_workers = 2

# hooks
hook_cfg_list = [
    dict(type='SamIterHook'), 
    dict(type='TestIterHook'), 
    dict(type='IterBasedEvalHook', iter_period=40), 
    dict(type='EpochSchedulerHook'),
    dict(type='IterBasedSaveTopkHook', top_k=2, monitor='recall_average', iter_period=40, checkpoint_dir='checkpoints/')
]

# model
config = ConfigDict()
config.window_size = 1024
config.hop_size = 320
config.resample_rate = 32000
config.mel_bins = 64
config.fmin = 50
config.fmax = 14000
config.enable_tscam = True
config.htsat_attn_heatmap = False
config.loss_type = "clip_ce"
config.enable_repeat_mode = False
model_cfg = {
    'type': 'HtsatAudioModel', 'spec_size': 256, 'patch_size': 4, 'in_chans': 1, 'num_classes': num_classes, 'window_size': 8, 
    'depths': [2,2,6,2], 'embed_dim': 96, 'patch_stride': (4, 4), 'num_heads': [4, 8, 16, 32], 'config': config, 
    'checkpoint_path': '/content/drive/MyDrive/聲音 & 語音辨識/pretrain_model/HTS-AT-Model-Backup/AudioSet/HTSAT_AudioSet_Saved_1.ckpt'
}

checkpoint_path = None

# batch transform
train_transform_cfg = None
test_transform_cfg = None

# optimizer
optimizer_cfg = {
    'type': 'SAM_AdamW',
    'get_params': lambda model: filter(lambda p: p.requires_grad, model.parameters()),
    'lr': 1e-4, 
    'betas': (0.9, 0.999), 
    'eps': 1e-08, 
    'weight_decay': 0.05
}

# scheduler
lr_scheduler_epoch = [10,20,30]
lr_rate = [0.02, 0.05, 0.1]

def lr_foo(epoch):       
    if epoch < 3:
        # warm up lr
        lr_scale = lr_rate[epoch]
    else:
        # warmup schedule
        lr_pos = int(- 1 - bisect.bisect_left(lr_scheduler_epoch, epoch))
        if lr_pos < -3:
            lr_scale = max(lr_rate[0] * (0.98 ** epoch), 0.03)
        else:
            lr_scale = lr_rate[lr_pos]
    return lr_scale


epoch_scheduler_cfg = {
    'type': 'LambdaLR',
    'lr_lambda': lr_foo
}

iter_scheduler_cfg = {'type': None}

# trainer
trainer_cfg = {
    'type': 'HtsatTrainer',
    'max_iter': 4000,
    'loss_fn': torch.nn.CrossEntropyLoss(),
    'device': 'cuda',
    'metric_names': ['accuracy', 'recall_average']
}