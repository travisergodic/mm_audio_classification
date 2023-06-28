from audio_cls.transforms.batch_transforms import *
from audio_cls.transforms.loader_transforms import *
from audio_cls.utils.registry import BATCH_TRANSFORMS, LOADER_TRANSFORMS


class ComposeBatchTransform:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, X, y):
        for transform in self.transform_list:
            X, y = transform(X, y)
        return X, y
    

class ComposeLoaderTransform:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, X):
        for transform in self.transform_list:
            X = transform(X)
        return X
    

def build_batch_processor(transform_cfg):
    return ComposeBatchTransform(
        [BATCH_TRANSFORMS.build(**ele) for ele in transform_cfg]
    ) if transform_cfg is not None else None


def build_loader_processor(transform_cfg):
    return ComposeLoaderTransform(
        [LOADER_TRANSFORMS.build(**ele) for ele in transform_cfg]
    ) if transform_cfg is not None else None