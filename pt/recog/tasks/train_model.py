from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import luigi

from pt.common.settings import results_path, TRAIN, VAL
from pt.common.utils import (
    safe_makedirs, append_log, setup_log)
from pt.common.optimizers import get_optimizer, SGD
from pt.recog.data.factory import (
    get_data_loader, MNIST, DEFAULT, NORMALIZE)
from pt.recog.models.factory import make_model
from pt.recog.models.mini import MINI
from pt.recog.tasks.utils import RecogTask


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


class TrainModelTask(RecogTask):
    dataset = luigi.Parameter()
    loader = luigi.Parameter(default=DEFAULT)
    transforms = luigi.ListParameter(default=[NORMALIZE])
    model_name = luigi.Parameter(default=MINI)
    input_shape = luigi.ListParameter()

    optimizer_name = luigi.Parameter(default=SGD)
    lr = luigi.FloatParameter(default=0.01)
    momentum = luigi.FloatParameter(default=0.5)
    batch_size = luigi.IntParameter(default=64)
    val_batch_size = luigi.IntParameter(default=1000)
    epochs = luigi.IntParameter(default=10)
    samples_per_epoch = luigi.IntParameter(default=None)
    val_samples_per_epoch = luigi.IntParameter(default=None)
    log_interval = luigi.IntParameter(default=10)

    task_name = 'train_model'

    def output(self):
        task_path = join(results_path, self.namespace, self.task_name)
        model_path = join(task_path, 'model')
        best_model_path = join(task_path, 'best_model')
        log_path = join(task_path, 'log.csv')


    def run(self):
        self.samples_per_epoch = None if self.samples_per_epoch == -1 \
            else self.samples_per_epoch
        self.val_samples_per_epoch = None if self.val_samples_per_epoch == -1 \
            else self.val_samples_per_epoch

        train_loader = get_data_loader(
            self.dataset, loader_name=self.loader,
            batch_size=self.batch_size, shuffle=True, split=TRAIN,
            transform_names=self.transforms, cuda=self.cuda)

        val_loader = get_data_loader(
            self.dataset, loader_name=self.loader,
            batch_size=self.val_batch_size, shuffle=False, split=VAL,
            transform_names=self.transforms, cuda=self.cuda)

        model = make_model(self.model_name, self.input_shape)
        if self.cuda:
            model.cuda()

        optimizer = get_optimizer(
            self.optimizer_name, model.parameters(), lr=self.lr,
            momentum=self.momentum)

        first_epoch = setup_log(log_path)
        if first_epoch > 1:
            model.load_state_dict(load_weights(self.namespace))

        min_val_loss = np.inf
        for epoch in range(first_epoch, self.epochs + 1):
            train_epoch(epoch, train_loader, model, optimizer, self.cuda,
                        self.log_interval, self.samples_per_epoch)

            val_loss, val_acc = val_epoch(
                epoch, val_loader, model, optimizer, self.cuda, self.log_interval,
                self.val_samples_per_epoch)

            append_log(log_path, (epoch, val_loss, val_acc))
            torch.save(model.state_dict(), model_path)
            if val_loss < min_val_loss:
                torch.save(model.state_dict(), best_model_path)
                min_val_loss = val_loss
