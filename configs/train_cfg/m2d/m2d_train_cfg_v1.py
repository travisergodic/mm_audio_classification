import torch
from ml_collections import ConfigDict

from configs.base import *

# pre-transfrom & aug
pre_transform_cfg = None
aug_cfg = None 

# dataset
train_dataset_cfg = {
    'type': 'TorchAudioDataset',
    'sample_rate': 22050,
    'num_classes': num_classes,
    'random_crop': True,
    'unit_samples': 22050 * 3
}

test_dataset_cfg = {
    'type': 'TorchAudioDataset', 
    'sample_rate': 22050,
    'num_classes': num_classes,
    'random_crop': False,
    'unit_samples': 22050 * 3
}

# sampler
sampler_cfg = {'type': 'BalancedBatchSampler', 'batch_size': 20, 'num_iter': 40}

# dataloader
train_batch_size = 20
test_batch_size = 40
num_workers = 2

# hooks
hook_cfg_list = [
    dict(type='PrecomputeStatsHook'),
    dict(type='SamIterHook'), 
    dict(type='TestIterHook'), 
    dict(type='IterBasedEvalHook', iter_period=40), 
    dict(type='CosLRSchedulerHook'),
    dict(type='IterBasedSaveTopkHook', top_k=2, monitor='recall_average', save_after=20, iter_period=40, checkpoint_dir='checkpoints/')
]

# model
cfg = ConfigDict()
cfg.weight_file = '/content/m2d_vit_base-80x608p16x16-221006-mr6/checkpoint-300.pth'
cfg.feature_d = 3840
cfg.sample_rate = 22050
cfg.n_fft = 400
cfg.window_size = 400
cfg.hop_size = 160
cfg.n_mels = 80
cfg.f_min = 50
cfg.f_max = 8000
cfg.window = 'hanning'

# Model specific parameters.
cfg.cls_token = False # Use CLS token
cfg.output_layers = [-1]  # list of layers to stack

# Linear evaluation/Fine-tuning common parameters.
cfg.training_mask = 0.5

# Fine-tuning parameters.
cfg.ft_freq_mask =30
cfg.ft_time_mask =100
cfg.ft_noise = 0.0
cfg.ft_rrc = True

model_cfg = {
    'type': 'M2DModel',
    'cfg': cfg,
    'n_class': num_classes
}

checkpoint_path = None

# batch transform
train_transform_cfg = (dict(type='MixupTransform', mixup_alpha=5),)
test_transform_cfg = None

# optimizer
optimizer_cfg = {
    'type': 'SAM_AdamW',
    'get_params': lambda model: filter(lambda p: p.requires_grad, model.parameters()),
    'lr': 1e-5, 
    'betas': (0.9, 0.95), 
    'eps': 1e-08, 
    'weight_decay': 0.0001,
    'amsgrad': True
}

epoch_scheduler_cfg = {'type': None}
iter_scheduler_cfg = {
    'type': 'CosineLR',
    't_initial': 101, 
    'decay_rate': 0.5,
    'lr_min': 1e-6, 
    't_in_epochs': True, 
    'warmup_t': 10,
    'warmup_lr_init': 1e-6, 
    'cycle_limit': 1
}

# trainer
trainer_cfg = {
    'type': 'NormalTrainer',
    'max_iter': 4000,
    'loss_fn': torch.nn.CrossEntropyLoss(),
    'device': 'cuda',
    'metric_names': ['accuracy', 'recall_average']
}