import torch.nn as nn 

from configs.base import *

sample_rate = 32000
device = 'cuda'

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
        "type": "EfficientatModel",
        "pretrained_name": "mn30_im", 
        "head_type": "mlp", 
        "se_dims": "c", 
        "num_classes": num_classes
    }
]

checkpoint_path_list = ['/content/t-brain_sound_classification/checkpoints/e48_0.6780979114953827.pt']

# batch transform
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

test_transform_cfg_list = [
    (dict(type='EfficientatMelTransform', device=device, **mel_cfg),)
]

# trainer
trainer_cfg_list = [
    {'type': 'EfficientatTrainer', 'device': device, 'n_epochs': 1}
]

# predictor 
predictor_cfg = {'classes_num': 5, 'activation_list': [nn.Softmax(dim=1)], 'shift_unit': 0, 'shift_num': 0}

# metric_list
metric_list = ["recall_average"]