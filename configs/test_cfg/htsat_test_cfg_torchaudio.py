import torch.nn as nn 

from configs.base import *
from configs.train_cfg.htsat.htsat_train_cfg_v1 import config as htsat_config

sample_rate = 32000

# pre-transfrom & aug
pre_transform_cfg = None
aug_cfg = None

test_dataset_cfg = {
    'type': 'TorchAudioDataset',
    'unit_samples': sample_rate * 3,
    'num_classes': num_classes,
    'random_crop': False,
    'padding_mode': 'copy', 
    'sample_rate': sample_rate
}


# dataloader
test_batch_size = 40
num_workers = 2

# model
model_cfg_list = [
    {
        'type': 'HtsatAudioModel', 'spec_size': 256, 'patch_size': 4, 'in_chans': 1, 'num_classes': num_classes, 'window_size': 8, 
        'depths': [2,2,6,2], 'embed_dim': 96, 'patch_stride': (4, 4), 'num_heads': [4, 8, 16, 32], 'config': htsat_config, 
        'checkpoint_path': '/content/drive/MyDrive/聲音 & 語音辨識/pretrain_model/HTS-AT-Model-Backup/AudioSet/HTSAT_AudioSet_Saved_1.ckpt'
    }
]

checkpoint_path_list = ['/content/drive/MyDrive/2-專案/多模態嗓音/模型權重檔/htsat_v4_e36_0.6834435615084324.pt']

# batch transform
test_transform_cfg_list = [None]

# trainer
trainer_cfg_list = [
    {'type': 'HtsatTrainer', 'device': 'cuda', 'n_epochs': 1}
]

# predictor 
predictor_cfg = {'classes_num': 5, 'activation_list': [nn.Softmax(dim=1)], 'shift_unit': 0, 'shift_num': 0}

# metric_list
metric_list = ["recall_average"]