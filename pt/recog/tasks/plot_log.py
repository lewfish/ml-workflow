from os.path import join
import argparse

import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from pt.common.settings import results_path
from pt.common.utils import safe_makedirs
from pt.recog.tasks.train_model import load_log
from pt.recog.tasks.utils import add_common_args

PLOT_LOG = 'plot_log'


def plot_log(args):
    task_path = join(results_path, args.namespace, PLOT_LOG)
    plot_path = join(task_path, 'plot.png')
    safe_makedirs(task_path)

    log = load_log(args.namespace)

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


def parse_args():
    parser = argparse.ArgumentParser(description='Plot recognition data')
    add_common_args(parser)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    plot_log(parse_args())
