from os.path import join

import numpy as np

from pt.common.settings import results_path, TRAIN
from pt.common.utils import safe_makedirs
from pt.recog.tasks.infer_probs import load_probs
from pt.recog.tasks.args import CommonArgs

INFER_PREDS = 'infer_preds'


def load_preds(namespace, split):
    infer_preds_path = join(results_path, namespace, INFER_PREDS)
    preds_path = join(infer_preds_path, '{}.npy'.format(split))
    return np.load(preds_path)


class InferPredsArgs():
    def __init__(self, common=CommonArgs(), split=TRAIN):
        self.common = common
        self.split = split


def infer_preds(args=InferPredsArgs()):
    task_path = join(results_path, args.common.namespace, INFER_PREDS)
    safe_makedirs(task_path)

    y_probs = load_probs(args.common.namespace, args.split)

    y_preds = np.argmax(y_probs, axis=1)
    preds_path = join(task_path, '{}.npy'.format(args.split))
    np.save(preds_path, y_preds)
