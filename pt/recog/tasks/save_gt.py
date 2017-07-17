from os.path import join

import numpy as np

from pt.common.settings import results_path, TRAIN
from pt.common.task import Task

from pt.recog.data.factory import get_data_loader
from pt.recog.tasks.args import CommonArgs, DatasetArgs

SAVE_GT = 'save_gt'


def load_gt(namespace, split):
    save_gt_path = join(results_path, namespace, SAVE_GT)
    gt_path = join(save_gt_path, '{}.npy'.format(split))
    return np.load(gt_path)


class SaveGt(Task):
    task_name = SAVE_GT

    class Args():
        def __init__(self, common=CommonArgs(), dataset=DatasetArgs(),
                     split=TRAIN, nsamples=8):
            self.common = common
            self.dataset = dataset
            self.split = split
            self.nsamples = nsamples

    def run(self):
        args = self.args
        loader = get_data_loader(
            args.dataset.dataset, loader_name=args.dataset.loader,
            batch_size=100, shuffle=False, split=args.split,
            cuda=args.common.cuda)

        y_list = []
        sample_count = 0
        for batch_idx, (_, y) in enumerate(loader):
            if sample_count + len(y) > args.nsamples:
                extra_samples = sample_count + len(y) - args.nsamples
                samples_to_keep = len(y) - extra_samples
                y = y.narrow(0, 0, samples_to_keep)

            y_list.append(y.numpy())
            sample_count += len(y)

            if args.nsamples is not None and sample_count >= args.nsamples:
                break

        probs_path = self.get_local_path('{}.npy'.format(args.split))
        y = np.concatenate(y_list)
        np.save(probs_path, y)
