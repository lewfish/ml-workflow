from os.path import join
import argparse

import numpy as np
import torch

from pt.common.settings import results_path, TRAIN, TEST
from pt.common.utils import safe_makedirs
from pt.recog.data.factory import get_data_loader, MNIST, DEFAULT
from pt.recog.tasks.utils import add_common_args

SAVE_GT = 'save_gt'


def load_gt(namespace, split):
    save_gt_path = join(results_path, namespace, SAVE_GT)
    gt_path = join(save_gt_path, '{}.npy'.format(split))
    return np.load(gt_path)


def save_gt(args):
    task_path = join(results_path, args.namespace, SAVE_GT)
    safe_makedirs(task_path)
    loader = get_data_loader(
        args.dataset, loader_name=args.loader,
        batch_size=args.batch_size, shuffle=False, split=args.split,
        cuda=args.cuda)

    y_list = []
    for batch_idx, (_, y) in enumerate(loader):
        y_list.append(y.numpy())

    probs_path = join(task_path, '{}.npy'.format(args.split))
    y = np.concatenate(y_list)
    np.save(probs_path, y)


def parse_args():
    parser = argparse.ArgumentParser(description='Save ground truth labels')
    add_common_args(parser)

    parser.add_argument('--dataset', type=str, default=MNIST,
                        help='name of the dataset')
    parser.add_argument('--loader', type=str, default=DEFAULT,
                        help='name of the dataset loader')
    parser.add_argument('--split', type=str, default=TRAIN,
                        help='name of the dataset split')
    parser.add_argument('--batch-size', type=int, default=1000,
                        metavar='N',
                        help='batch size for testing (default: 1000)')

    args = parser.parse_args()
    if args.split == TEST:
        raise ValueError('split cannot be test')
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    save_gt(parse_args())
