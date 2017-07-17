from os.path import join

import numpy as np

from pt.common.settings import results_path, TRAIN
from pt.common.task import Task

from pt.recog.tasks.infer_probs import load_probs, INFER_PROBS
from pt.recog.tasks.args import CommonArgs

INFER_PREDS = 'infer_preds'


def load_preds(namespace, split):
    infer_preds_path = join(results_path, namespace, INFER_PREDS)
    preds_path = join(infer_preds_path, '{}.npy'.format(split))
    return np.load(preds_path)


class InferPreds(Task):
    task_name = INFER_PREDS

    class Args():
        def __init__(self, common=CommonArgs(), split=TRAIN):
            self.common = common
            self.split = split

    def get_input_paths(self):
        return [join(self.namespace, INFER_PROBS)]

    def run(self):
        args = self.args

        y_probs = load_probs(self.namespace, args.split)

        y_preds = np.argmax(y_probs, axis=1)
        preds_path = self.get_local_path('{}.npy'.format(args.split))
        np.save(preds_path, y_preds)
