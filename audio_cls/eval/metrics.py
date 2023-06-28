from sklearn.metrics import accuracy_score, recall_score, precision_score

from audio_cls.utils.registry import METRICS


@METRICS.register
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

@METRICS.register
def recall_binary(y_true, y_pred):
    return recall_score(y_true, y_pred, average='binary')

@METRICS.register
def recall_micro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='micro')

@METRICS.register
def recall_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

@METRICS.register
def recall_average(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None).mean()