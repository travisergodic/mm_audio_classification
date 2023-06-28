import torch
from abc import ABC, abstractmethod

from tqdm import tqdm


class BaseTrainer(ABC):
    def __init__(
            self, audio_model, device, optimizer=None, n_epochs=None, max_iter=None, loss_fn=None, 
            metric_names=None, train_batch_processor=None, test_batch_processor=None, hooks=None, 
            epoch_scheduler=None, iter_scheduler=None, **kwargs
        ):
        assert (n_epochs or max_iter) is not None
        self.optimizer = optimizer
        self.n_epochs = 1 if n_epochs is None else n_epochs 
        self.max_iter = max_iter
        self.device = device
        self.metric_names = metric_names
        self.train_batch_processor = train_batch_processor
        self.test_batch_processor = test_batch_processor
        self.audio_model = audio_model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if loss_fn is not None else None
        self.hooks = hooks
        self.epoch_scheduler = epoch_scheduler
        self.iter_scheduler = iter_scheduler
        self.current_iter = 1 
        self.__dict__.update(kwargs)

    @abstractmethod
    def forward(self, X):
        pass    

    def fit(self, train_loader=None, test_loader=None):
        self.train_loader, self.test_loader = train_loader, test_loader
        break_out_flag = False
        # set up
        self.call_hook("set_up")
        for self.epoch in range(1, self.n_epochs + 1): 
            self.audio_model.train()
            pbar = tqdm(train_loader) 
            pbar.set_description(f"Epoch {self.epoch}/{self.n_epochs}")
            self.call_hook("on_train_epoch_start")
            for self.iter, batch in enumerate(pbar):
                self.X, self.y = batch['waveform'].to(self.device), batch['target'].to(self.device)
                # preprocess batch
                if self.train_batch_processor is not None:
                    self.X, self.y = self.train_batch_processor(self.X, self.y)
                self.call_hook("on_train_batch_start")
                self.call_hook("run_train_iter")
                self.call_hook("on_train_batch_end")
                # determine stop iteration
                self.current_iter += 1
                if (self.max_iter is not None) and self.current_iter >= self.max_iter:
                    break_out_flag = True
                    break
            self.call_hook("on_train_epoch_end")
            if break_out_flag:
                break
        self.call_hook("tear_down")

    @torch.no_grad()
    def test(self, test_loader):
        self.audio_model.eval()
        # test epoch start
        self.call_hook("on_test_epoch_start")
        for batch in test_loader:
            self.X, self.y = batch['waveform'].to(self.device), batch['target'].to(self.device)
            if self.test_batch_processor is not None:
                self.X, self.y = self.test_batch_processor(self.X, self.y)
            self.call_hook("on_test_batch_start")
            self.call_hook("run_test_iter")
            self.call_hook("on_test_batch_end")
        self.call_hook("on_test_epoch_end")

    def call_hook(self, name):
        for hook in self.hooks:
            getattr(hook, name)(self)