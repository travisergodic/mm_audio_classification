import torch
import numpy as np


class Predictor:
    def __init__(self, trainer_list, batch_processor_list, classes_num, activation_list, shift_unit=0, shift_num=0):
        self.trainer_list = trainer_list
        self.device = trainer_list[0].device
        self.batch_processor_list = batch_processor_list
        self.classes_num = classes_num
        self.activation_list = activation_list
        self.shift_unit = shift_unit
        self.shift_num = shift_num

    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        return torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)

    def forward(self, X, trainer_index):
        return self.trainer_list[trainer_index].forward(X)

    @torch.no_grad()
    def predict_step(self, batch):
        pred = torch.zeros(batch['waveform'].size(0), self.classes_num).float().to(self.device)
        for j in range(len(self.trainer_list)):
            X = batch["waveform"].to(self.device)
            if self.batch_processor_list[j] is not None:
                X, _ = self.batch_processor_list[j](X, None)
            temp_pred = self.forward(X, j)
            pred += self.activation_list[j](temp_pred)
        return (pred / len(self.trainer_list)).cpu().numpy(), batch["name"]

    def predict(self, test_loader):
        preds, names = [], [] 
        for batch in test_loader:
            res = self.predict_step(batch)
            preds.append(res[0])
            names.append(res[1])
        preds = np.concatenate(preds, axis = 0)
        names = np.concatenate(names, axis = 0)
        return preds, names
