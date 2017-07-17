from pt.common.optimizers import SGD
from pt.common.settings import TRAIN, VAL

from pt.recog.data.factory import MNIST, DEFAULT, NORMALIZE
from pt.recog.models.mini import MINI
from pt.recog.tasks.args import (
    CommonArgs, DatasetArgs, ModelArgs, OptimizerArgs, TrainArgs)
from pt.recog.tasks.plot_data import PlotData
from pt.recog.tasks.train_model import TrainModel
from pt.recog.tasks.save_gt import SaveGt
from pt.recog.tasks.plot_log import PlotLog
from pt.recog.tasks.infer_probs import InferProbs
from pt.recog.tasks.infer_preds import InferPreds
from pt.recog.tasks.compute_scores import ComputeScores
from pt.recog.tasks.plot_preds import PlotPreds


def run():
    namespace = 'recog/mnist_test'
    common = CommonArgs(
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

    PlotData(namespace, PlotData.Args(
        common=common, dataset=dataset, split=TRAIN, nimages=8))()
    TrainModel(namespace, TrainModel.Args(
        common=common, dataset=dataset, model=model, train=train))()
    PlotLog(namespace, PlotLog.Args(common=common))()

    for split in [TRAIN, VAL]:
        SaveGt(namespace, SaveGt.Args(
            common=common, dataset=dataset, split=split,
            nsamples=nsamples))()
        InferProbs(namespace, InferProbs.Args(
            common=common, dataset=dataset, model=model, split=split,
            batch_size=100, nsamples=nsamples))()
        InferPreds(namespace, InferPreds.Args(common=common, split=split))()

    ComputeScores(namespace, ComputeScores.Args(common=common))()
    PlotPreds(namespace, PlotPreds.Args(
        common=common, dataset=dataset, max_plots=nsamples))()


if __name__ == '__main__':
    run()
