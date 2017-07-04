from os.path import join
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


from pt.common.settings import results_path, TRAIN, VAL
from pt.common.utils import (
    safe_makedirs, append_log, setup_log)
from pt.common.optimizers import get_optimizer, SGD
from pt.recog.data.factory import (
    get_data_loader, MNIST, DEFAULT, NORMALIZE)
from pt.recog.models.factory import make_model
from pt.recog.models.mini import MINI
from pt.recog.tasks.utils import add_common_args

TRAIN_MODEL = 'train_model'


def load_weights(namespace):
    train_model_path = join(results_path, namespace, TRAIN_MODEL)
    best_model_path = join(train_model_path, 'best_model')
    return torch.load(best_model_path)


def load_log(namespace):
    train_model_path = join(results_path, namespace, TRAIN_MODEL)
    log_path = join(train_model_path, 'log.csv')
    return np.genfromtxt(log_path, delimiter=',', skip_header=1)


def train_epoch(epoch, train_loader, model, optimizer, cuda, log_interval,
                nsamples):
    model.train()

    if nsamples is None:
        nsamples = len(train_loader.dataset)
    sample_count = 0
    for batch_idx, (x, y) in enumerate(train_loader, start=1):
        if cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        if sample_count + len(x) > nsamples:
            extra_samples = sample_count - nsamples
            samples_to_keep = len(x) - extra_samples
            x = x.narrow(0, 0, samples_to_keep)
            y = y.narrow(0, 0, samples_to_keep)

        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

        sample_count += len(x)
        percent_sample_count = 100. * sample_count / nsamples
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, sample_count, nsamples, percent_sample_count,
                loss.data[0]))

        if nsamples is not None and sample_count >= nsamples:
            break


def val_epoch(epoch, val_loader, model, optimizer, cuda, log_interval,
              nsamples):
    model.eval()

    if nsamples is None:
        nsamples = len(val_loader.dataset)
    val_loss = 0
    val_acc = 0
    correct = 0
    sample_count = 0
    for batch_idx, (x, y) in enumerate(val_loader, start=1):
        if cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x, volatile=True), Variable(y)

        if sample_count + len(x) > nsamples:
            extra_samples = sample_count - nsamples
            samples_to_keep = len(x) - extra_samples
            x = x.narrow(0, 0, samples_to_keep)
            y = y.narrow(0, 0, samples_to_keep)

        output = model(x)
        val_loss += F.nll_loss(output, y).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data).cpu().sum()
        sample_count += len(x)

        if (nsamples is not None and
                sample_count >= nsamples):
            break

    # loss function already averages over batch size
    val_loss /= nsamples
    val_acc = 100. * correct / nsamples

    display_str = '\nTest set: Average loss: ' + \
                  '{:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
    print(display_str.format(
        val_loss, correct, nsamples, val_acc))

    return val_loss, val_acc


def train_model(args):
    task_path = join(results_path, args.namespace, TRAIN_MODEL)
    safe_makedirs(task_path)
    model_path = join(task_path, 'model')
    best_model_path = join(task_path, 'best_model')

    train_loader = get_data_loader(
        args.dataset, loader_name=args.loader,
        batch_size=args.batch_size, shuffle=True, split=TRAIN,
        transform_names=args.transforms, cuda=args.cuda)

    val_loader = get_data_loader(
        args.dataset, loader_name=args.loader,
        batch_size=args.val_batch_size, shuffle=False, split=VAL,
        transform_names=args.transforms, cuda=args.cuda)

    model = make_model(args.model_name, args.input_shape)
    if args.cuda:
        model.cuda()

    optimizer = get_optimizer(
        args.optimizer_name, model.parameters(), lr=args.lr,
        momentum=args.momentum)

    log_path = join(task_path, 'log.csv')
    first_epoch = setup_log(log_path)
    if first_epoch > 1:
        model.load_state_dict(load_weights(args.namespace))

    min_val_loss = np.inf
    for epoch in range(first_epoch, args.epochs + 1):
        train_epoch(epoch, train_loader, model, optimizer, args.cuda,
                    args.log_interval, args.samples_per_epoch)

        val_loss, val_acc = val_epoch(
            epoch, val_loader, model, optimizer, args.cuda, args.log_interval,
            args.val_samples_per_epoch)

        append_log(log_path, (epoch, val_loss, val_acc))
        torch.save(model.state_dict(), model_path)
        if val_loss < min_val_loss:
            torch.save(model.state_dict(), best_model_path)
            min_val_loss = val_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train recognition model')
    add_common_args(parser)

    parser.add_argument('--dataset', type=str, default=MNIST,
                        help='name of the dataset')
    parser.add_argument('--loader', type=str, default=DEFAULT,
                        help='name of the dataset loader')
    parser.add_argument('--transforms', type=str, nargs='*',
                        default=[NORMALIZE],
                        help='list of transform')

    parser.add_argument('--model-name', type=str, default=MINI,
                        help='name of the model')
    parser.add_argument('--input-shape', type=int, nargs='*',
                        default=None,
                        help='shape of input data')

    parser.add_argument('--optimizer-name', type=str, default=SGD,
                        help='name of the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--samples-per-epoch', type=int, default=None)
    parser.add_argument('--val-samples-per-epoch', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before' +
                             'logging training status')

    args = parser.parse_args()
    args.samples_per_epoch = None if args.samples_per_epoch == -1 \
        else args.samples_per_epoch
    args.val_samples_per_epoch = None if args.val_samples_per_epoch == -1 \
        else args.val_samples_per_epoch
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    train_model(parse_args())
