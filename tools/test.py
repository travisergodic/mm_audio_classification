import os
import logging
import argparse
import sys
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd

from audio_cls.eval.predict import Predictor
from audio_cls.dataset import DATASETS
from audio_cls.models import MODELS
from audio_cls.trainer import TRAINERS
from audio_cls.utils.get_cfg import get_cfg_by_file
from audio_cls.transforms import build_loader_processor, build_batch_processor
from audio_cls.eval.metrics import METRICS
from audio_cls.utils.logger_helper import setup_logger


logger = setup_logger(level=logging.INFO)

def main():
    # import config file
    config = get_cfg_by_file(args.config_file)
    
    assert args.task in ("evaluate", "inference")

    if args.task == "evaluate":
        df = pd.read_csv(config.csv_path)
        logger.info('Read csv file successuflly.')
        test_path_list = df[config.audio_name_column].apply(lambda name: os.path.join(config.audio_path, name + '.wav')).tolist() 
        test_label_list = df[config.target_column].astype(int) - 1  
    
    elif args.task == "inference":
        test_path_list = [os.path.join(args.data_dir, name) for name in os.listdir(args.data_dir) if name.endswith('.wav')] 
        test_label_list = [0] * len(test_path_list)
        logger.info(f'Find {len(test_path_list)} number of data.')

    # pre-transform & aug
    pre_transform = build_loader_processor(config.pre_transform_cfg)

    # dataset
    test_audio_dataset = DATASETS.build(
        audio_path_list=test_path_list, label_list=test_label_list, pre_transform=pre_transform, 
        **config.test_dataset_cfg
    )

    # dataloader
    test_loader = torch.utils.data.DataLoader(
        test_audio_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    # build models & set eval mode
    model_list = []
    for model_cfg, checkpoint_path in zip(config.model_cfg_list, config.checkpoint_path_list):
        model = MODELS.build(**model_cfg).eval()
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Load checkpoint from {checkpoint_path} successfully.")
        model_list.append(model)

    # build batch processor
    
    batch_processor_list = [
        build_batch_processor(test_transform_cfg) for test_transform_cfg in config.test_transform_cfg_list
    ]

    # build trainer
    trainer_list = []
    for trainer_cfg, audio_model in zip(config.trainer_cfg_list, model_list):
        trainer_list.append(
            TRAINERS.build(
                type=trainer_cfg.pop('type'), 
                audio_model=audio_model, 
                **trainer_cfg
            )
        )

    # build predictor
    predictor = Predictor(trainer_list=trainer_list, batch_processor_list=batch_processor_list, **config.predictor_cfg)
    preds, names = predictor.predict(test_loader)

    names = [name.split('.')[0] for name in names]
    preds = preds.argmax(axis=1) + 1
    
    if args.task == "inference":
        if args.save_path is None:
            raise ValueError("save_path could not be None.")
        pd.DataFrame({"ID": names, "Category": preds}).to_csv(args.save_path, header=False, index=False)
        logging.info(f"Save inference result at {args.save_path}")
    
    elif args.task == "evaluate":
        df_res = pd.DataFrame({"ID": names, "pred": preds})
        series_gt = pd.read_csv(config.csv_path, index_col='ID')["Disease category"]
        df_res["gt"] = df_res['ID'].apply(lambda x: series_gt[x])
        for metric_name in config.metric_list: 
            print(metric_name, ':',METRICS[metric_name](df_res["gt"].to_list(), df_res["pred"].to_list()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare sepsis data for training!")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_path", type=str, required=False)
    args = parser.parse_args()
    main()