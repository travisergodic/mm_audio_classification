import logging
from pathlib import Path

import torch
from rich.table import Table
from rich.console import Console

from audio_cls.hooks.base_hook import BaseHook
from audio_cls.utils.registry import HOOKS

logger = logging.getLogger(__name__)


def save_ckpt(audio_model, ckpt_path):
    if not Path(ckpt_path).parent.is_dir():
        ckpt_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(audio_model.state_dict(), ckpt_path)
    logger.info(f'Save checkpoint at {ckpt_path}')


@HOOKS.register
class SaveAllEpochHook(BaseHook):
    def __init__(self, checkpoint_dir='checkpoints/'):
        self.checkpoint_dir = checkpoint_dir

    def on_train_epoch_end(self, trainer):
        # save checkpoint
        checkpoint_path = Path(self.checkpoint_dir) / f"e{self.trainer_epoch}_{trainer.metric_dict[self.monitor]}.pt"
        save_ckpt(trainer.audio_model, checkpoint_path)


@HOOKS.register
class EpochBasedSaveTopkHook(BaseHook):
    def __init__(self, top_k, monitor, save_after=0, checkpoint_dir='checkpoints/'):
        self.top_k = top_k
        self.monitor = monitor
        self.checkpoint_dir = checkpoint_dir
        self.save_after = save_after
        self.best_records = []

    def on_test_epoch_end(self, trainer):
        if trainer.epoch < self.save_after:
            return

        self.best_records.append(
            {'epoch': trainer.epoch, 'score': trainer.metric_dict[self.monitor]}
        )
        self.best_records = sorted(self.best_records, key=lambda s: s['score'], reverse=True)           

        if len(self.best_records) <= self.top_k:
            checkpoint_path = Path(self.checkpoint_dir) / f"e{trainer.epoch}_{trainer.metric_dict[self.monitor]}.pt"
            save_ckpt(trainer.audio_model, checkpoint_path)
        
        elif self.best_records[-1]['epoch'] != trainer.epoch:
            remove_ckpt = (Path(self.checkpoint_dir) / f'e{self.best_records[-1]["epoch"]}_{self.best_records[-1]["score"]}.pt')
            remove_ckpt.unlink(missing_ok=False)
            logger.info(f'Remove checkpoint {remove_ckpt}')

            checkpoint_path = Path(self.checkpoint_dir) / f"e{trainer.epoch}_{trainer.metric_dict[self.monitor]}.pt"
            save_ckpt(trainer.audio_model, checkpoint_path)
            
        self.best_records = self.best_records[:self.top_k]

    def tear_down(self, trainer):
        table = Table(title=f'Top {self.top_k} checkpoints')
        table.add_column('EPOCH', justify='left', style='cyan')
        table.add_column(f'{self.monitor}', justify='left', style='green')

        for record in self.best_records:
            table.add_row(str(record['epoch']), str(record['score']))
        console = Console()
        with console.capture() as capture:
            console.print(table, end='')
        print(capture.get())


@HOOKS.register
class IterBasedSaveTopkHook(BaseHook):
    def __init__(self, top_k, monitor, save_after, iter_period=50, checkpoint_dir='checkpoints/'):
        self.monitor = monitor
        self.top_k = top_k
        self.checkpoint_dir = checkpoint_dir
        self.best_records = []
        self.save_after = save_after
        self.iter_period = iter_period
        self.current_period = 0

    def on_test_epoch_end(self, trainer):
        self.current_period += 1
        if (trainer.current_iter % self.iter_period != 0) or (self.current_period < self.save_after):
            return 
        
        self.best_records.append(
            {'num_period': self.current_period, 'score': trainer.metric_dict[self.monitor]}
        )
        self.best_records = sorted(self.best_records, key=lambda s: s['score'], reverse=True)           

        if len(self.best_records) <= self.top_k:
            checkpoint_path = Path(self.checkpoint_dir) / f"e{self.current_period}_{trainer.metric_dict[self.monitor]}.pt"
            save_ckpt(trainer.audio_model, checkpoint_path)
        
        elif self.best_records[-1]['num_period'] != self.current_period:
            remove_ckpt = (Path(self.checkpoint_dir) / f'e{self.best_records[-1]["num_period"]}_{self.best_records[-1]["score"]}.pt')
            remove_ckpt.unlink(missing_ok=False)
            logger.info(f'Remove checkpoint {remove_ckpt}')

            checkpoint_path = Path(self.checkpoint_dir) / f"e{self.current_period}_{trainer.metric_dict[self.monitor]}.pt"
            save_ckpt(trainer.audio_model, checkpoint_path)
            
        self.best_records = self.best_records[:self.top_k]

    def tear_down(self, trainer):
        table = Table(title=f'Top {self.top_k} checkpoints')
        table.add_column('NUM_PERIOD', justify='left', style='cyan')
        table.add_column(f'{self.monitor}', justify='left', style='green')

        for record in self.best_records:
            table.add_row(str(record['num_period']), str(record['score']))
        console = Console()
        with console.capture() as capture:
            console.print(table, end='')
        print(capture.get())