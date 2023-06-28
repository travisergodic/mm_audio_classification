import torch 

from configs.base import *

lr = 1e-5

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
    'mixup': 0.5, 
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

sampler_cfg = {'type': None}

# dataloader
train_batch_size = 20
test_batch_size = 40
num_workers = 2

# hooks
hook_cfg_list = [
    dict(type='NormalIterHook'), 
    dict(type='TestIterHook'), 
    dict(type='EpochBasedEvalHook'), 
    dict(type='WarmupHook', warmup_steps=240, period=40, lr=lr), 
    dict(type='EpochBasedSaveTopkHook', top_k=2, monitor='recall_average', checkpoint_dir='checkpoints/')
]

# model
model_cfg = {
    'type': 'ASTModel', 'label_dim': num_classes, 'fstride': 10, 'tstride': 10, 'input_fdim': 128, 'input_tdim': 512, 
    'imagenet_pretrain': True, 'audioset_pretrain': True, 'model_size': 'base384', 'save_dir': '/content/checkpoints'
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
    'type': 'NormalTrainer',
    'n_epochs': 100,
    'loss_fn': torch.nn.CrossEntropyLoss(weight=torch.tensor([1/0.536, 1/0.220, 1/0.168, 1/0.044, 1/0.032])),
    'device': 'cuda',
    'metric_names': ['accuracy', 'recall_average']
}