import torch

from configs.base import *


device = 'cuda'
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
    dict(type='CosLRSchedulerHook'),
    dict(type='IterBasedSaveTopkHook', top_k=2, monitor='recall_average', iter_period=40, checkpoint_dir='checkpoints/')
]

# transform
mel_cfg = {
    'n_mels': 128,
    'sr': sample_rate,
    'hopsize': 320,
    'n_fft': 1024,
    'freqm': 0,
    'timem': 0,
    'fmin': 0,
    'fmax': None,
    'fmin_aug_range': 1,
    'fmax_aug_range': 1000
}

train_transform_cfg = (
    # dict(type='MixupTransform', mixup=0.5, mixup_alpha=10), 
    dict(type='EfficientatMelTransform', device=device, **mel_cfg),
)

test_transform_cfg = (dict(type='EfficientatMelTransform', device=device, **mel_cfg),)


# model
model_cfg = {
    "type": "EfficientatModel",
    "pretrained_name": "mn30_im", 
    "head_type": "mlp", 
    "se_dims": "c", 
    "num_classes": num_classes
}

checkpoint_path = None

# optimizer
optimizer_cfg = {
    "type": "SAM_AdamW", 
    "get_params": lambda model: model.parameters(),
    "lr": 1e-5,
    "weight_decay": 0.001
}

# scheduler
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
    'type': 'EfficientatTrainer',
    'max_iter': 4000,
    'loss_fn': torch.nn.CrossEntropyLoss(),
    'device': device,
    'metric_names': ['accuracy', 'recall_average']
}