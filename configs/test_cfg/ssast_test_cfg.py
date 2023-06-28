import torch.nn as nn 

from configs.base import *

# pre-transfrom & aug
pre_transform_cfg = None 
aug_cfg = None

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

# dataloader
test_batch_size = 40
num_workers = 2

# model
model_cfg_list = [
    {
        'type': 'SSASTModel', 'label_dim': num_classes, 'fstride': 10, 'tstride': 10, 'fshape': 16, 'tshape': 16, 
        'input_fdim': 128, 'input_tdim': 512, 'model_size': 'base', 'pretrain_stage': False, 
        'load_pretrained_mdl_path': '/content/t-brain_sound_classification/SSAST-Base-Patch-400.pth?dl=1'
    }
]

checkpoint_path_list = ['/content/t-brain_sound_classification/checkpoints/e27_0.6500474786346144.pt']

# batch transform
test_transform_cfg_list = [None]

# trainer
trainer_cfg_list = [
    {'type': 'SsastTrainer', 'device': 'cuda', 'n_epochs': 1}
]

# predictor 
predictor_cfg = {'classes_num': 5, 'activation_list': [nn.Softmax(dim=1)], 'shift_unit': 0, 'shift_num': 0}

# metric_list
metric_list = ["recall_average"]