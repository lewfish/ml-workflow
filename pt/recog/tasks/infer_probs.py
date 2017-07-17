from os.path import join

import numpy as np
import torch
from torch.autograd import Variable

from pt.common.settings import results_path, TRAIN
from pt.common.task import Task

from pt.recog.data.factory import MNIST, DEFAULT, NORMALIZE
from pt.recog.data.factory import get_data_loader
from pt.recog.models.factory import make_model
from pt.recog.models.mini import MINI
from pt.recog.tasks.train_model import TRAIN_MODEL, load_weights
from pt.recog.tasks.args import (
    CommonArgs, DatasetArgs, ModelArgs)

INFER_PROBS = 'infer_probs'


def load_probs(namespace, split):
    infer_probs_path = join(results_path, namespace, INFER_PROBS)
    probs_path = join(infer_probs_path, '{}.npy'.format(split))
    return np.load(probs_path)


class InferProbs(Task):
    task_name = INFER_PROBS

    class Args():
        def __init__(self, common=CommonArgs(), dataset=DatasetArgs(),
                     model=ModelArgs(), split=TRAIN, batch_size=100,
                     nsamples=8):
            self.common = common
            self.dataset = dataset
            self.model = model
            self.split = split
            self.batch_size = batch_size
            self.nsamples = nsamples

    def get_input_paths(self):
        return [join(self.namespace, TRAIN_MODEL)]

    def run(self):
        args = self.args
        loader = get_data_loader(
            args.dataset.dataset, loader_name=args.dataset.loader,
            batch_size=args.batch_size, shuffle=False, split=args.split,
            cuda=args.common.cuda)

        model = make_model(args.model.model, args.model.input_shape)
        model.load_state_dict(load_weights(self.namespace))
        if args.common.cuda:
            model.cuda()
        model.eval()

        y_list = []
        sample_count = 0
        for batch_idx, (x, _) in enumerate(loader):
            print('.', end='') # noqa

            if sample_count + len(x) > args.nsamples:
                extra_samples = sample_count + len(x) - args.nsamples
                samples_to_keep = len(x) - extra_samples
                x = x.narrow(0, 0, samples_to_keep)

            if args.common.cuda:
                x = x.cuda()
            x = Variable(x, volatile=True)
            y = model(x)
            y_list.append(y.data.numpy())

            sample_count += len(x)
            if args.nsamples is not None and sample_count >= args.nsamples:
                break

        print()
        probs_path = self.get_local_path('{}.npy'.format(args.split))
        y = np.concatenate(y_list)
        np.save(probs_path, y)
