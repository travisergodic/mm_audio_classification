import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from audio_cls.hooks.base_hook import BaseHook
from audio_cls.utils.registry import HOOKS
from audio_cls.transforms.aug import mixup1, Mixup2, do_mixup2


__all__ = [
    'NormalIterHook', 
    'NormalIterHookMixup1',
    'NormalIterHookMixup2', 
    'AutocastIterHook', 
    'SamIterHook', 
    'TestIterHook'
]


@HOOKS.register
class NormalIterHook(BaseHook):
    def run_train_iter(self, trainer):
        loss = trainer.loss_fn(trainer.forward(trainer.X), trainer.y)
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        trainer.train_loss = loss.item()


@HOOKS.register
class NormalIterHookMixup1(BaseHook):
    def __init__(self, mixup_alpha):
        self.mixup_alpha = mixup_alpha

    def run_train_iter(self, trainer):
        # generate mixup indices
        bs = trainer.X.size(0)
        rn_indices, lam = mixup1(bs, self.mixup_alpha)
        lam = lam.to(trainer.X.device)
        # mixup X
        trainer.X = trainer.X * lam.reshape(bs, 1, 1, 1) + trainer.X[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
        # forward
        y_hat = trainer.forward(trainer.X)
        # calculate loss
        samples_loss =  trainer.loss_fn(y_hat, trainer.y) * lam.reshape(bs) + \
            trainer.loss_fn(y_hat, trainer.y[rn_indices]) * (1. - lam.reshape(bs))
        loss = samples_loss.mean()
        # zero grad
        trainer.optimizer.zero_grad()
        # backward
        loss.backward()
        trainer.optimizer.step()
        trainer.train_loss = loss.item()


@HOOKS.register
class NormalIterHookMixup2(BaseHook):
    def __init__(self, mixup_alpha=1.):
        self.mixup_alpha = mixup_alpha
        self.mixup_augmenter = Mixup2(mixup_alpha=mixup_alpha)

    def run_train_iter(self, trainer):
        if self.mixup_alpha == 0:
            bs = trainer.X.size(0)
            mixup_lambda = self.mixup_augmenter.get_lambda(batch_size=bs).to(self.X.device)
            y_hat = trainer.forward(trainer.X, mixup_lambda)
            trainer.y = do_mixup2(trainer.y, mixup_lambda)
        else:
            y_hat = trainer.forward(trainer.X, None)
        loss = trainer.loss_fn(y_hat, trainer.y)
        # zero grad
        trainer.optimizer.zero_grad()
        # backward
        loss.backward()
        trainer.optimizer.step()
        trainer.train_loss = loss.item()


@HOOKS.register    
class AutocastIterHook(BaseHook):
    def set_up(self, trainer):
        self.scaler = GradScaler()

    def run_train_iter(self, trainer):
        with autocast(): 
            loss = trainer.loss_fn(trainer.forward(trainer.X), trainer.y)
        trainer.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(trainer.optimizer)
        self.scaler.update()
        trainer.train_loss = loss.item()


@HOOKS.register
class SamIterHook(BaseHook):
    def run_train_iter(self, trainer):
        loss = trainer.loss_fn(trainer.forward(trainer.X), trainer.y)  # use this loss for any training statistics
        loss.backward()
        trainer.optimizer.first_step(zero_grad=True)
        
        # second forward-backward pass
        trainer.loss_fn(trainer.forward(trainer.X), trainer.y).backward()  # make sure to do a full forward pass
        trainer.optimizer.second_step(zero_grad=True)
        trainer.train_loss = loss.item()
    

@HOOKS.register
class TestIterHook(BaseHook):
    def run_test_iter(self, trainer):
        trainer.pred = trainer.forward(trainer.X)
        test_loss = trainer.loss_fn(trainer.pred, trainer.y)
        if len(test_loss.size()) == 0:
            trainer.test_loss = test_loss.item()
        else:
            trainer.test_loss = test_loss.mean().item()