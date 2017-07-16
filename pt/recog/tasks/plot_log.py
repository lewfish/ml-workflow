from os.path import join

import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from pt.common.settings import results_path
from pt.common.utils import safe_makedirs
from pt.recog.tasks.train_model import load_log
from pt.recog.tasks.args import CommonArgs

PLOT_LOG = 'plot_log'


class PlotLogArgs():
    def __init__(self, common=CommonArgs()):
        self.common = common


def plot_log(args=PlotLogArgs()):
    task_path = join(results_path, args.common.namespace, PLOT_LOG)
    plot_path = join(task_path, 'plot.png')
    safe_makedirs(task_path)

    log = load_log(args.common.namespace)

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
