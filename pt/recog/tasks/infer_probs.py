from os.path import join
import argparse

import numpy as np
import torch
from torch.autograd import Variable

from pt.common.settings import results_path, TRAIN
from pt.common.utils import safe_makedirs
from pt.recog.data.factory import MNIST, DEFAULT, NORMALIZE
from pt.recog.data.factory import get_data_loader
from pt.recog.models.factory import make_model
from pt.recog.models.mini import MINI
from pt.recog.tasks.utils import add_common_args
from pt.recog.tasks.train_model import load_weights

INFER_PROBS = 'infer_probs'


def load_probs(namespace, split):
    infer_probs_path = join(results_path, namespace, INFER_PROBS)
    probs_path = join(infer_probs_path, '{}.npy'.format(split))
    return np.load(probs_path)


def infer_probs(args):
    task_path = join(results_path, args.namespace, INFER_PROBS)
    safe_makedirs(task_path)
    loader = get_data_loader(
        args.dataset, loader_name=args.loader,
        batch_size=args.batch_size, shuffle=False, split=args.split,
        cuda=args.cuda)

    model = make_model(args.model_name, args.input_shape)
    model.load_state_dict(load_weights(args.namespace))
    if args.cuda:
        model.cuda()
    model.eval()

    y_list = []
    for batch_idx, (x, _) in enumerate(loader):
        print('.', end='') # noqa

        samples_generated = (batch_idx + 1) * args.batch_size
        if args.nsamples is not None and samples_generated > args.nsamples:
            extra_samples = samples_generated - args.nsamples
            samples_to_keep = args.batch_size - extra_samples
            x = x.narrow(0, 0, samples_to_keep)

        if args.cuda:
            x = x.cuda()
        x = Variable(x, volatile=True)
        y = model(x)
        y_list.append(y.data.numpy())

        if args.nsamples is not None and samples_generated > args.nsamples:
            break

    print()
    probs_path = join(task_path, '{}.npy'.format(args.split))
    y = np.concatenate(y_list)
    np.save(probs_path, y)


def parse_args():
    parser = argparse.ArgumentParser(description='Infer output probs')
    add_common_args(parser)

    parser.add_argument('--model-name', type=str, default=MINI,
                        help='name of the model')
    parser.add_argument('--input-shape', type=int, nargs='*',
                        default=None,
                        help='shape of input data')

    parser.add_argument('--dataset', type=str, default=MNIST,
                        help='name of the dataset')
    parser.add_argument('--loader', type=str, default=DEFAULT,
                        help='name of the dataset loader')
    parser.add_argument('--transforms', type=str, nargs='*',
                        default=[NORMALIZE],
                        help='list of transform')
    parser.add_argument('--split', type=str, default=TRAIN,
                        help='name of the dataset split')
    parser.add_argument('--batch-size', type=int, default=1000,
                        metavar='N',
                        help='batch size for testing (default: 1000)')
    parser.add_argument('--nsamples', type=int, default=None,
                        help='number of samples to run this on')

    args = parser.parse_args()
    args.nsamples = None if args.nsamples == -1 else args.nsamples
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    infer_probs(parse_args())
