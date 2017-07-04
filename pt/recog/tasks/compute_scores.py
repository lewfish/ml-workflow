from os.path import join
import argparse

import numpy as np

from pt.common.settings import results_path, VAL
from pt.common.utils import safe_makedirs, save_json
from pt.recog.tasks.utils import add_common_args
from pt.recog.tasks.infer_preds import load_preds
from pt.recog.tasks.save_gt import load_gt

COMPUTE_SCORES = 'compute_scores'


def _compute_scores(y_preds, y_gt):
    ncorrect = np.sum(y_preds == y_gt)
    n = y_preds.shape[0]
    accuracy = ncorrect / n

    return {
        'accuracy': accuracy
    }


def compute_scores(args):
    task_path = join(results_path, args.namespace, COMPUTE_SCORES)
    safe_makedirs(task_path)

    split = VAL
    y_preds = load_preds(args.namespace, split)
    y_gt = load_gt(args.namespace, split)

    scores_path = join(task_path, 'scores.json')
    scores = _compute_scores(y_preds, y_gt)
    save_json(scores, scores_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute scores on validation set')
    add_common_args(parser)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    compute_scores(parse_args())
