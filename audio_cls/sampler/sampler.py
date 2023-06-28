import logging

import numpy as np

from audio_cls.utils.registry import SAMPLERS


logger = logging.getLogger(__name__)


@SAMPLERS.register
class BalancedBatchSampler:
    def __init__(self, audio_path_list, label_list, batch_size, num_iter, random_seed=1234):
        """
        Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        """
        self.audio_paths = np.array(audio_path_list)
        self.labels = np.array(label_list)
        self.batch_size = batch_size
        self.classes, self.samples_num_per_class = np.unique(self.labels, return_counts=True)
        self.random_state = np.random.RandomState(random_seed)
        self.num_iter = num_iter
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(len(self.classes)):
            self.indexes_per_class.append(np.where(self.labels == k)[0])
            
        # Shuffle indexes
        for k in range(len(self.classes)):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * len(self.classes)

    def expand_queue(self, queue):
        classes_set = self.classes.tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        """
        while True:
            batch_indices = []
            i = 0
            while i < self.batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])
                batch_indices.append(index)
                i += 1
            yield batch_indices

    def __len__(self):
        return self.num_iter


@SAMPLERS.register
class TrainBatchSampler:
    def __init__(self, audio_path_list, label_list, batch_size, num_iter, random_seed=1234):
        """
        Balanced sampler. Generate batch meta for training.
        """        
        self.audios_num = len(audio_path_list)
        self.indices = np.arange(self.audios_num)
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.random_state.shuffle(self.indices)
        self.num_iter = num_iter
        self.pointer = 0

    def __iter__(self):
        """
        Generate batch meta for training. 
        """
        while True:
            batch_indices = []
            i = 0
            while i < self.batch_size:
                index = self.indices[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indices)
                
                batch_indices.append(index)
                i += 1
            yield batch_indices

    def __len__(self):
        return self.num_iter


@SAMPLERS.register
class AlternateBatchSampler:
    def __init__(self, audio_path_list, label_list, batch_size, num_iter, random_seed=1234):
        """
        AlternateSampler is a combination of Sampler and Balanced Sampler. 
        AlternateSampler alternately sample data from Sampler and Blanced Sampler.
        """
        self.sampler1 = TrainBatchSampler(audio_path_list, label_list, batch_size, random_seed)
        self.sampler2 = BalancedBatchSampler(audio_path_list, label_list, batch_size, random_seed)
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.count = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 'index_in_hdf5': int}, 
            ...]
        """
        while True:
            self.count += 1
            if self.count % 2 == 0:
                batch_indices = []
                i = 0
                while i < self.batch_size:
                    index = self.sampler1.indices[self.sampler1.pointer]
                    self.sampler1.pointer += 1

                    # Shuffle indexes and reset pointer
                    if self.sampler1.pointer >= self.sampler1.audios_num:
                        self.sampler1.pointer = 0
                        self.sampler1.random_state.shuffle(self.sampler1.indices)
                    
                    batch_indices.append(index)
                    i += 1

            elif self.count % 2 == 1:
                batch_indices = []
                i = 0
                while i < self.batch_size:
                    if len(self.sampler2.queue) == 0:
                        self.sampler2.queue = self.sampler2.expand_queue(self.sampler2.queue)

                    class_id = self.sampler2.queue.pop(0)
                    pointer = self.sampler2.pointers_of_classes[class_id]
                    self.sampler2.pointers_of_classes[class_id] += 1
                    index = self.sampler2.indexes_per_class[class_id][pointer]
                    
                    # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                    if self.sampler2.pointers_of_classes[class_id] >= self.sampler2.samples_num_per_class[class_id]:
                        self.sampler2.pointers_of_classes[class_id] = 0
                        self.sampler2.random_state.shuffle(self.sampler2.indexes_per_class[class_id])

                    batch_indices.append(index)
                    i += 1

            yield batch_indices

    def __len__(self):
        return self.num_iter