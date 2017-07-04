from os.path import join
import argparse

import matplotlib as mpl
# For headless environments
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
import numpy as np
import torch

from pt.common.settings import results_path, TRAIN
from pt.common.utils import safe_makedirs
from pt.recog.data.factory import (
    get_data_loader, MNIST, DEFAULT, NORMALIZE)
from pt.recog.tasks.utils import add_common_args

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


def plot_data(args):
    task_path = join(results_path, args.namespace, PLOT_DATA)
    safe_makedirs(task_path)

    loader = get_data_loader(
        args.dataset, loader_name=args.loader,
        batch_size=args.nimages, shuffle=False, split=args.split,
        transform_names=args.transforms, cuda=args.cuda)

    x, y = next(iter(loader))
    images = np.transpose(x.numpy(), [0, 2, 3, 1])
    labels = [loader.dataset.get_label(label_idx)
              for label_idx in y.numpy()]

    plot_path = join(
        task_path, '{}_{}_{}.png'.format(
            args.dataset, args.loader, args.split))
    plot_images(plot_path, images, labels, ncols=4, normalize=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Plot recognition data')
    add_common_args(parser)

    parser.add_argument('--dataset', type=str, default=MNIST,
                        help='name of the dataset')
    parser.add_argument('--loader', type=str, default=DEFAULT,
                        help='name of the dataset loader')
    parser.add_argument('--transforms', type=str, nargs='*',
                        default=[NORMALIZE],
                        help='list of transform')

    parser.add_argument('--split', type=str, default=TRAIN,
                        help='name of the dataset split')
    parser.add_argument('--nimages', type=int, default=1, metavar='N',
                        help='how many images to plot')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    plot_data(parse_args())
