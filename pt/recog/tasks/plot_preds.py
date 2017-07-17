from os.path import join

import numpy as np
import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt

from pt.common.settings import VAL
from pt.common.task import Task

from pt.recog.data.factory import get_data_loader
from pt.recog.tasks.infer_preds import load_preds, INFER_PREDS
from pt.recog.tasks.save_gt import load_gt, SAVE_GT
from pt.recog.tasks.args import CommonArgs, DatasetArgs

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


class PlotPreds(Task):
    task_name = PLOT_PREDS

    class Args():
        def __init__(self, common=CommonArgs(), dataset=DatasetArgs(),
                     max_plots=8):
            self.common = common
            self.dataset = dataset
            self.max_plots = max_plots

    def get_input_paths(self):
        return [
            join(self.namespace, INFER_PREDS),
            join(self.namespace, SAVE_GT)]

    def run(self):
        args = self.args

        split = VAL
        y_preds = load_preds(self.namespace, split)
        y_gt = load_gt(self.namespace, split)

        loader = get_data_loader(
            args.dataset.dataset, loader_name=args.dataset.loader,
            batch_size=1, shuffle=False, split=split,
            cuda=False)

        _plot_preds(
            y_preds, y_gt, loader.dataset, self.task_path, args.max_plots)
