from os.path import join

import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
import numpy as np
import luigi

from pt.common.settings import results_path, TRAIN
from pt.recog.data.factory import (
    get_data_loader, DEFAULT, NORMALIZE)
from pt.recog.tasks.utils import RecogTask


def plot_images(plot_path, images, labels, ncols=4, normalize=True):
    nimgs = images.shape[0]
    nrows = (nimgs // ncols) + 1

    if normalize:
        min_val = images.ravel().min()
        max_val = images.ravel().max()
        images = (images - min_val) / (max_val - min_val)

    plt.figure()
    for img_idx in range(nimgs):
        ax = plt.subplot(nrows, ncols, img_idx + 1)

        img = images[img_idx, :, :, :]
        if img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(labels[img_idx])

        if img_idx + 1 == nimgs:
            break

    plt.tight_layout()
    plt.savefig(plot_path, format='png')


class PlotDataTask(RecogTask):
    dataset = luigi.Parameter()
    loader = luigi.Parameter(default=DEFAULT)
    transforms = luigi.ListParameter(default=[NORMALIZE])
    split = luigi.Parameter(default=TRAIN)
    nimages = luigi.IntParameter(default=1)

    task_name = 'plot_data'

    def output(self):
        task_path = join(results_path, self.namespace, self.task_name)
        # safe_makedirs(task_path)
        plot_path = join(task_path, '{}_{}_{}.png'.format(
            self.dataset, self.loader, self.split))
        return luigi.LocalTarget(plot_path, format=luigi.format.Nop)

    def run(self):
        super(PlotDataTask).run()

        loader = get_data_loader(
            self.dataset, loader_name=self.loader,
            batch_size=self.nimages, shuffle=False, split=self.split,
            transform_names=self.transforms, cuda=self.cuda)

        x, y = next(iter(loader))
        images = np.transpose(x.numpy(), [0, 2, 3, 1])
        labels = [loader.dataset.get_label(label_idx)
                  for label_idx in y.numpy()]

        with self.output().open('wb') as plot_file:
            plot_images(plot_file, images, labels, ncols=4, normalize=True)
