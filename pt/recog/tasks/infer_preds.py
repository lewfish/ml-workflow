from os.path import join
import argparse

import numpy as np

from pt.common.settings import results_path, TRAIN
from pt.common.utils import safe_makedirs
from pt.recog.tasks.utils import add_common_args
from pt.recog.tasks.infer_probs import load_probs

INFER_PREDS = 'infer_preds'


def load_preds(namespace, split):
    infer_preds_path = join(results_path, namespace, INFER_PREDS)
    preds_path = join(infer_preds_path, '{}.npy'.format(split))
    return np.load(preds_path)


def infer_preds(args):
    task_path = join(results_path, args.namespace, INFER_PREDS)
    safe_makedirs(task_path)

    y_probs = load_probs(args.namespace, args.split)

    y_preds = np.argmax(y_probs, axis=1)
    preds_path = join(task_path, '{}.npy'.format(args.split))
    np.save(preds_path, y_preds)


def parse_args():
    parser = argparse.ArgumentParser(description='Infer output probs')
    add_common_args(parser)

    parser.add_argument('--split', type=str, default=TRAIN,
                        help='name of the dataset split')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    infer_preds(parse_args())
