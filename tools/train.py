import os
import logging
import argparse
import sys
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from audio_cls.utils.logger_helper import setup_logger
from audio_cls.utils.get_cfg import get_cfg_by_file
from audio_cls.hooks import HOOKS
from audio_cls.models import MODELS
from audio_cls.trainer import TRAINERS
from audio_cls.dataset import DATASETS
from audio_cls.transforms import build_loader_processor, build_batch_processor
from audio_cls.optimizer import OPTIMIZERS
from audio_cls.scheduler import SCHEDULERS
from audio_cls.sampler import SAMPLERS 


logger = setup_logger(level=logging.INFO)


def main():
    # read csv
    df = pd.read_csv(config.csv_path)
    logger.info('Read csv file successuflly.')

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        df, df[config.target_column], 
        test_size=config.test_size, 
        random_state=config.seed, 
        stratify=df[config.target_column].values
    )

    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    train_path_list = X_train[config.audio_name_column].apply(lambda name: os.path.join(config.audio_path, name + '.wav')) 
    train_label_list = X_train[config.target_column].astype(int) - 1  
    test_path_list = X_test[config.audio_name_column].apply(lambda name: os.path.join(config.audio_path, name + '.wav')) 
    test_label_list = X_test[config.target_column].astype(int) - 1  

    # pre-transform & aug
    pre_transform = build_loader_processor(config.pre_transform_cfg)
    aug = build_loader_processor(config.aug_cfg)

    # train dataset & dataloader
    train_audio_dataset = DATASETS.build(
        audio_path_list=train_path_list, label_list=train_label_list, pre_transform=pre_transform,
        aug=aug, **config.train_dataset_cfg
    )

    batch_sampler = SAMPLERS.build(audio_path_list=train_path_list, label_list=train_label_list, **config.sampler_cfg)

    if batch_sampler is None:
        train_loader = torch.utils.data.DataLoader(
            train_audio_dataset, batch_size=config.train_batch_size, shuffle=True, 
            num_workers=config.num_workers, pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_audio_dataset, num_workers=config.num_workers, pin_memory=True, batch_sampler=batch_sampler
        )

    # test dataset & dataloader
    test_audio_dataset = DATASETS.build(
        audio_path_list=test_path_list, label_list=test_label_list, pre_transform=pre_transform, 
        **config.test_dataset_cfg
    )

    test_loader = torch.utils.data.DataLoader(
        test_audio_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    # build hooks
    hooks = [HOOKS.build(**hook_cfg) for hook_cfg in config.hook_cfg_list]

    # build model
    audio_model = MODELS.build(**config.model_cfg)

    # load checkpoint
    if config.checkpoint_path is not None:
        audio_model.load_state_dict(torch.load(config.checkpoint_path))

    # build batch processor
    train_batch_processor = build_batch_processor(config.train_transform_cfg)
    test_batch_processor = build_batch_processor(config.test_transform_cfg)

    # optimizer
    optimizer = OPTIMIZERS.build(model=audio_model, **config.optimizer_cfg)

    # scheduler
    epoch_scheduler = SCHEDULERS.build(optimizer=optimizer, **config.epoch_scheduler_cfg)
    iter_scheduler = SCHEDULERS.build(optimizer=optimizer, **config.iter_scheduler_cfg)

    # build trainer
    trainer = TRAINERS.build(
        type=config.trainer_cfg.pop('type'),
        audio_model=audio_model, 
        optimizer=optimizer,
        iter_scheduler=iter_scheduler,
        epoch_scheduler=epoch_scheduler, 
        train_batch_processor=train_batch_processor,
        test_batch_processor=test_batch_processor, 
        hooks=hooks,
        **config.trainer_cfg
    )

    trainer.fit(train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare sepsis data for training!")
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()
    config = get_cfg_by_file(args.config_file)
    main() 