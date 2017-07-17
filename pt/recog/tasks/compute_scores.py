from os.path import join

import numpy as np

from pt.common.settings import VAL
from pt.common.utils import save_json
from pt.common.task import Task

from pt.recog.tasks.infer_preds import load_preds, INFER_PREDS
from pt.recog.tasks.save_gt import load_gt, SAVE_GT
from pt.recog.tasks.args import CommonArgs

COMPUTE_SCORES = 'compute_scores'


def _compute_scores(y_preds, y_gt):
    ncorrect = np.sum(y_preds == y_gt)
    n = y_preds.shape[0]
    accuracy = ncorrect / n

    return {
        'accuracy': accuracy
    }


class ComputeScores(Task):
    task_name = COMPUTE_SCORES

    class Args():
        def __init__(self, common=CommonArgs()):
            self.common = common

    def get_input_paths(self):
        return [
            join(self.namespace, INFER_PREDS),
            join(self.namespace, SAVE_GT)]

    def run(self):
        split = VAL
        y_preds = load_preds(self.namespace, split)
        y_gt = load_gt(self.namespace, split)

        scores_path = self.get_local_path('scores.json')
        scores = _compute_scores(y_preds, y_gt)
        save_json(scores, scores_path)
