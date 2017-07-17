from pt.recog.data.factory import MNIST, DEFAULT, NORMALIZE
from pt.recog.models.mini import MINI
from pt.common.optimizers import SGD


class CommonArgs():
    def __init__(self, cuda=False, seed=None):
        self.cuda = cuda
        self.seed = seed


class DatasetArgs():
    def __init__(self, dataset=MNIST, loader=DEFAULT, transforms=[NORMALIZE]):
        self.dataset = dataset
        self.loader = loader
        self.transforms = transforms


class ModelArgs():
    def __init__(self, model=MINI, input_shape=[1, 28, 28]):
        self.model = model
        self.input_shape = input_shape


class OptimizerArgs():
    def __init__(self, optimizer=SGD, lr=0.1, momentum=0.9):
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum


class TrainArgs():
    def __init__(self, optimizer=OptimizerArgs(),
                 batch_size=8, val_batch_size=8, epochs=2,
                 samples_per_epoch=16, val_samples_per_epoch=16,
                 log_interval=1):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.samples_per_epoch = samples_per_epoch
        self.val_samples_per_epoch = val_samples_per_epoch
        self.log_interval = log_interval
