import torch 

from configs.base import *

lr = 3e-5

# pre-transfrom & aug
pre_transform_cfg = None 
aug_cfg = None

# dataset
train_dataset_cfg = {
    'type': 'AudiosetDataset',
    'label_num': num_classes, 
    'num_mel_bins': 128, 
    'freqm': 24, 
    'timem': 24, 
    'mixup': 0, 
    'mean': -6.6268077, 
    'std': 5.358466,
    'skip_norm': False,
    'noise': True,
    'target_length': 512
}

test_dataset_cfg = {
    'type': 'AudiosetDataset', 
    'label_num': num_classes, 
    'num_mel_bins': 128, 
    'freqm': 0, 
    'timem': 0, 
    'mixup': 0, 
    'mean': -6.6268077, 
    'std': 5.358466,
    'skip_norm': False,
    'noise': False,
    'target_length': 512
}

sampler_cfg = {'type': 'BalancedBatchSampler', 'batch_size': 20, 'num_iter': 40}

# dataloader
train_batch_size = 20
test_batch_size = 40
num_workers = 2

# hooks
hook_cfg_list = [
    dict(type='NormalIterHook'), 
    dict(type='TestIterHook'), 
    dict(type='IterBasedEvalHook', iter_period=40), 
    dict(type='WarmupHook', warmup_steps=500, period=40, lr=lr), 
    dict(type='IterBasedSaveTopkHook', top_k=2, monitor='recall_average', save_after=20, iter_period=40, checkpoint_dir='checkpoints/')
]

# model
model_cfg = {
    'type': 'SSASTModel', 'label_dim': num_classes, 'fstride': 10, 'tstride': 10, 'fshape': 16, 'tshape': 16, 
    'input_fdim': 128, 'input_tdim': 512, 'model_size': 'base', 'pretrain_stage': False, 
    'load_pretrained_mdl_path': '/content/t-brain_sound_classification/SSAST-Base-Patch-400.pth?dl=1'
}

checkpoint_path = None

# batch transform
train_transform_cfg = None
test_transform_cfg = None

# optimizer
optimizer_cfg = {
    'type': "Adam",
    'get_params': lambda model: [p for p in model.parameters() if p.requires_grad],
    'lr': lr, 
    'weight_decay': 5e-7, 
    'betas': (0.95, 0.999)
}

# scheduler
epoch_scheduler_cfg = {'type': None}
iter_scheduler_cfg = {'type': None}

# trainer
trainer_cfg = {
    'type': 'SsastTrainer',
    'max_iter': 4000, 
    'loss_fn': torch.nn.CrossEntropyLoss(),
    'device': 'cuda',
    'metric_names': ['accuracy', 'recall_average']
}