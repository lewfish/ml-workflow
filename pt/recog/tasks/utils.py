import luigi
import torch


class RecogTask(luigi.Task):
    namespace = luigi.Parameter()
    no_cuda = luigi.BoolParameter(default=False)
    seed = luigi.IntParameter(default=1)

    def run(self):
        self.cuda = not self.no_cuda and torch.cuda.is_available()
