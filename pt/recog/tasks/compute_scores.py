from os.path import join

import numpy as np

from pt.common.settings import results_path, VAL
from pt.common.utils import safe_makedirs, save_json
from pt.recog.tasks.infer_preds import load_preds
from pt.recog.tasks.save_gt import load_gt
from pt.recog.tasks.args import CommonArgs

COMPUTE_SCORES = 'compute_scores'


def _compute_scores(y_preds, y_gt):
    ncorrect = np.sum(y_preds == y_gt)
    n = y_preds.shape[0]
    accuracy = ncorrect / n

    return {
        'accuracy': accuracy
    }


class ComputeScoresArgs():
    def __init__(self, common=CommonArgs()):
        self.common = common


def compute_scores(args=ComputeScoresArgs()):
    task_path = join(results_path, args.common.namespace, COMPUTE_SCORES)
    safe_makedirs(task_path)

    split = VAL
    y_preds = load_preds(args.common.namespace, split)
    y_gt = load_gt(args.common.namespace, split)

    scores_path = join(task_path, 'scores.json')
    scores = _compute_scores(y_preds, y_gt)
    save_json(scores, scores_path)
