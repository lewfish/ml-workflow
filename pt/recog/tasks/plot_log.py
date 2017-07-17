from os.path import join

import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from pt.common.task import Task

from pt.recog.tasks.train_model import load_log, TRAIN_MODEL
from pt.recog.tasks.args import CommonArgs

PLOT_LOG = 'plot_log'


class PlotLog(Task):
    task_name = PLOT_LOG

    class Args():
        def __init__(self, common=CommonArgs()):
            self.common = common

    def get_input_paths(self):
        return [join(self.namespace, TRAIN_MODEL)]

    def run(self):
        plot_path = self.get_local_path('plot.png')
        log = load_log(self.namespace)

        plt.figure()
        plt.grid()
        epochs = log[:, 0]
        test_loss = log[:, 1]
        test_acc = log[:, 2]

        plt.subplot(1, 2, 1)
        plt.xlabel('Epochs')
        plt.ylabel('Test Loss')
        plt.plot(epochs, test_loss, '-', label='Test Loss')

        plt.subplot(1, 2, 2)
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.plot(epochs, test_acc, '-', label='Test Accuracy')

        plt.savefig(plot_path)
