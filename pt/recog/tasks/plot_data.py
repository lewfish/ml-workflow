from os.path import join

import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
import numpy as np

from pt.common.settings import TRAIN
from pt.common.task import Task

from pt.recog.data.factory import get_data_loader
from pt.recog.tasks.args import CommonArgs, DatasetArgs

PLOT_DATA = 'plot_data'


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
    plt.savefig(plot_path)


class PlotData(Task):
    task_name = PLOT_DATA

    class Args():
        def __init__(self, common=CommonArgs(), dataset=DatasetArgs(),
                     split=TRAIN, nimages=1):
            self.common = common
            self.dataset = dataset
            self.split = split
            self.nimages = nimages

    def run(self):
        args = self.args
        loader = get_data_loader(
            args.dataset.dataset, loader_name=args.dataset.loader,
            batch_size=args.nimages, shuffle=False, split=args.split,
            transform_names=args.dataset.transforms, cuda=args.common.cuda)

        x, y = next(iter(loader))
        images = np.transpose(x.numpy(), [0, 2, 3, 1])
        labels = [loader.dataset.get_label(label_idx)
                  for label_idx in y.numpy()]

        plot_path = self.get_local_path('{}_{}_{}.png'.format(
                args.dataset.dataset, args.dataset.loader, args.split))
        plot_images(plot_path, images, labels, ncols=4, normalize=True)
