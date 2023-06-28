import torch.nn as nn 

from configs.base import *
from configs.train_cfg.m2d.m2d_train_cfg_v1 import cfg 

# pre-transfrom & aug
pre_transform_cfg = None 
aug_cfg = None

test_dataset_cfg = {
    'type': 'TorchAudioDataset', 
    'sample_rate': 22050,
    'num_classes': num_classes,
    'random_crop': True,
    'unit_samples': 22050 * 3
}

# dataloader
test_batch_size = 40
num_workers = 2

# model
model_cfg_list = [
    {
        'type': 'M2DModel',
        'cfg': cfg,
        'n_class': num_classes
    }
]

checkpoint_path_list = ['/content/t-brain_sound_classification/checkpoints/e48_0.6780979114953827.pt']

# batch transform
test_transform_cfg_list = [None]

# trainer
trainer_cfg_list = [
    {'type': 'NormalTrainer', 'device': 'cuda', 'n_epochs': 1}
]

# predictor 
predictor_cfg = {'classes_num': 5, 'activation_list': [nn.Softmax(dim=1)], 'shift_unit': 0, 'shift_num': 0}

# metric_list
metric_list = ["recall_average"]