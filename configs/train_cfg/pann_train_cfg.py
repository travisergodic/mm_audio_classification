import torch

from configs.base import *


resample_rate = 32000

# pre-transfrom & aug
pre_transform_cfg_list = [dict(type='NormalizeAudioDuration', normailze_duration=3)]
aug_cfg_list = None

# dataset
train_dataset_cfg = {
    'type': 'LibrosaDataset',
    'sample_rate': resample_rate,
    'num_classes': num_classes
}

test_dataset_cfg = {
    'type': 'LibrosaDataset', 
    'sample_rate': resample_rate,
    'num_classes': num_classes
}

# dataloader
train_batch_size = 8
test_batch_size = 16
num_workers = 2

# hooks
hook_cfg_list = [
    dict(type='NormalIterHook'), 
    dict(type='TestIterHook'), 
    dict(type='EpochSchedulerHook'),
    dict(type='LogTestMetric'),
    dict(type='SaveTopkCheckpointHook', top_k=1, monitor='accuracy', checkpoint_dir='checkpoints/')
]

# model
model_cfg = {
    "type": "Cnn14",
    "sample_rate": resample_rate,
    "window_size": 1024,
    "hop_size": 320, 
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "checkpoint_path": "Cnn14_mAP=0.431.pth",
    "classes_num": num_classes
}

checkpoint_path = None

# batch transform
train_transform_cfg_list = None
test_transform_cfg_list = None


# optimizer
optimizer_cfg = {
    'type': 'Adam',
    'get_params': lambda model: model.parameters(),
    'lr': 1e-4, 
    'betas': (0.9, 0.999), 
    'eps': 1e-08, 
    'amsgrad': True
}

# scheduler
epoch_scheduler_cfg = {'type': None}
iter_scheduler_cfg = {'type': None}


# trainer
trainer_cfg = {
    'type': 'PannTrainer',
    'n_epochs': 30,
    'loss_fn': torch.nn.CrossEntropyLoss(),
    'device': 'cpu',
    'metric_names': ['accuracy']
}