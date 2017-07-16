from pt.common.optimizers import SGD
from pt.common.settings import TRAIN, VAL

from pt.recog.data.factory import MNIST, DEFAULT, NORMALIZE
from pt.recog.models.mini import MINI
from pt.recog.tasks.args import (
    CommonArgs, DatasetArgs, ModelArgs, OptimizerArgs, TrainArgs)
from pt.recog.tasks.plot_data import plot_data, PlotDataArgs
from pt.recog.tasks.train_model import train_model, TrainModelArgs
from pt.recog.tasks.save_gt import save_gt, SaveGtArgs
from pt.recog.tasks.plot_log import plot_log, PlotLogArgs
from pt.recog.tasks.infer_probs import infer_probs, InferProbsArgs
from pt.recog.tasks.infer_preds import infer_preds, InferPredsArgs
from pt.recog.tasks.compute_scores import compute_scores, ComputeScoresArgs
from pt.recog.tasks.plot_preds import plot_preds, PlotPredsArgs


def run():
    common = CommonArgs(
        namespace='recog/mnist_test',
        cuda=False,
        seed=None)
    dataset = DatasetArgs(
        dataset=MNIST,
        loader=DEFAULT,
        transforms=[NORMALIZE])
    model = ModelArgs(
        model=MINI,
        input_shape=[1, 28, 28])
    optimizer = OptimizerArgs(
        optimizer=SGD,
        lr=0.1,
        momentum=0.9)
    train = TrainArgs(
        optimizer=optimizer,
        batch_size=8,
        val_batch_size=8,
        epochs=2,
        samples_per_epoch=16,
        val_samples_per_epoch=16,
        log_interval=1)
    nsamples = 8

    plot_data(PlotDataArgs(
        common=common, dataset=dataset, split=TRAIN, nimages=8))
    train_model(TrainModelArgs(
        common=common, dataset=dataset, model=model, train=train))
    plot_log(PlotLogArgs(common=common))

    for split in [TRAIN, VAL]:
        save_gt(SaveGtArgs(
            common=common, dataset=dataset, split=split,
            nsamples=nsamples))
        infer_probs(InferProbsArgs(
            common=common, dataset=dataset, model=model, split=split,
            batch_size=100, nsamples=nsamples))
        infer_preds(InferPredsArgs(common=common, split=split))

    compute_scores(ComputeScoresArgs(common=common))
    plot_preds(PlotPredsArgs(
        common=common, dataset=dataset, max_plots=nsamples))


if __name__ == '__main__':
    run()
