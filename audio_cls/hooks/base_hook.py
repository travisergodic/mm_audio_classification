class BaseHook:
    def set_up(self, trainer):
        pass

    def tear_down(self, trainer):
        pass

    def run_train_iter(self, trainer):
        pass

    def run_test_iter(self, trainer):
        pass

    def on_train_epoch_start(self, trainer):
        pass

    def on_train_epoch_end(self, trainer):
        pass

    def on_test_epoch_start(self, trainer):
        pass

    def on_test_epoch_end(self, trainer):
        pass

    def on_train_batch_start(self, trainer):
        pass

    def on_train_batch_end(self, trainer):
        pass

    def on_test_batch_start(self, trainer):
        pass

    def on_test_batch_end(self, trainer):
        pass