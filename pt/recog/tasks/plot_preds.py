from os.path import join
import argparse

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from pt.common.settings import results_path, VAL
from pt.common.utils import safe_makedirs
from pt.recog.data.factory import (
    get_data_loader, MNIST, DEFAULT)
from pt.recog.tasks.utils import add_common_args
from pt.recog.tasks.infer_preds import load_preds
from pt.recog.tasks.save_gt import load_gt

PLOT_PREDS = 'plot_preds'


def _plot_preds(y_preds, y_gt, dataset, task_path, max_plots):
    error_inds = np.where(y_preds != y_gt)[0]

    for error_count, error_ind in enumerate(error_inds, start=1):
        pred_label = dataset.get_label(y_preds[error_ind])
        gt_label = dataset.get_label(y_gt[error_ind])

        x = dataset[error_ind][0].numpy()
        x = np.transpose(x, [1, 2, 0])
        if x.ndim == 3 and x.shape[2] == 1:
            x = np.squeeze(x)

        min_val = x.ravel().min()
        max_val = x.ravel().max()
        x = (x - min_val) / (max_val - min_val)
        if x.ndim == 2:
            plt.imshow(x, cmap='gray')
        else:
            plt.imshow(x)

        title = 'GT: {}, Prediction: {}'.format(gt_label, pred_label)
        plt.title(title)

        plot_path = join(task_path, '{}.png'.format(error_ind))
        plt.savefig(plot_path)

        if error_count == max_plots:
            break


def plot_preds(args):
    task_path = join(results_path, args.namespace, PLOT_PREDS)
    safe_makedirs(task_path)

    split = VAL
    y_preds = load_preds(args.namespace, split)
    y_gt = load_gt(args.namespace, split)

    loader = get_data_loader(
        args.dataset, loader_name=args.loader,
        batch_size=1, shuffle=False, split=split,
        cuda=False)

    _plot_preds(y_preds, y_gt, loader.dataset, task_path, args.max_plots)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot predictions on validation set')
    add_common_args(parser)

    parser.add_argument('--dataset', type=str, default=MNIST,
                        help='name of the dataset')
    parser.add_argument('--loader', type=str, default=DEFAULT,
                        help='name of the dataset loader')
    parser.add_argument('--max-plots', type=int, default=10,
                        help='max number of plots to make')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    plot_preds(parse_args())
