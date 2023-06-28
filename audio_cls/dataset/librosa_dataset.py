import os
import logging

import librosa
import numpy as np
from tqdm import tqdm 
from torch.utils.data import Dataset
from audio_cls.utils.registry import DATASETS


logger = logging.getLogger(__name__)


def load_full_dataset(path_list, label_list, num_classes, sample_rate=None, pre_transform=None):
    assert len(path_list) == len(label_list)
    output_list = []
    for audio_path, label in tqdm(zip(path_list, label_list)):
        name = os.path.basename(audio_path)
        target = np.zeros(num_classes)
        target[int(label)] = 1
        waveform, sr = librosa.load(audio_path, sr=sample_rate)
        if pre_transform is not None:
            waveform = pre_transform(waveform)
        output_list.append({"name": name, "target": target, "waveform": waveform})
    return output_list


@DATASETS.register
class LibrosaDataset(Dataset):
    def __init__(self, audio_path_list, label_list, num_classes, sample_rate=None, pre_transform=None, aug=None, **kwargs):
        logger.info("Creating audio full dataset ...")
        self.full_dataset = load_full_dataset(
            audio_path_list, label_list, num_classes, 
            sample_rate=sample_rate, pre_transform=pre_transform
        ) 
        self.aug= aug

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        X, y = self.full_dataset[index]['waveform'], self.full_dataset[index]['target']
        if self.aug is not None:
            X, y = self.aug(X, y)
        return {'waveform': X, 'target': y, 'name': self.full_dataset[index]['name']}

    def __len__(self):
        return len(self.full_dataset)