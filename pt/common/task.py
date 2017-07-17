from os.path import join, exists
from subprocess import call

from pt.common.utils import safe_makedirs
from pt.common.settings import (
    is_remote, results_path, s3_results_path)


def local_path_exists(path):
    return exists(join(results_path, path))


def s3_sync(path, is_download=True):
    # Using aws cli instead of boto because boto does not have
    # ability to upload/download recursively based on prefix.
    local_path = join(results_path, path)
    remote_path = join(s3_results_path, path)
    if is_download:
        call(['aws', 's3', 'sync', remote_path, local_path])
    else:
        call(['aws', 's3', 'sync', local_path, remote_path])


class Task():
    def __init__(self, args, namespace, task_name):
        self.args = args
        self.namespace = namespace
        self.task_name = task_name

        self.namespace_path = join(results_path, self.namespace)
        self.task_path = join(self.namespace_path, self.task_name)

    def _make_task_dir(self):
        safe_makedirs(self.task_path)

    def _download_inputs(self):
        input_paths = self.get_input_paths()
        for input_path in input_paths:
            s3_sync(input_path, is_download=True)

    def _upload_outputs(self):
        output_paths = self.get_output_paths()
        for output_path in output_paths:
            s3_sync(output_path, is_download=False)

    def _is_runnable(self):
        path_missing = False
        input_paths = self.get_input_paths()
        for input_path in input_paths:
            if not local_path_exists(input_path):
                print('{} does not exist!'.format(input_path))
                path_missing = True
        return not path_missing

    def get_local_path(self, path, namespace=None, task_name=None):
        namespace = self.namespace if namespace is None else namespace
        task_name = self.task_name if task_name is None else namespace
        return join(results_path, namespace, task_name, path)

    def __call__(self):
        self._make_task_dir()
        if is_remote:
            self._download_inputs()
        if self._is_runnable():
            self.run()
            if is_remote:
                self._upload_outputs()
        else:
            raise ValueError('Task {} is missing input files!'.format(
                self.task_name))

    def get_input_paths(self):
        return []

    def get_output_paths(self):
        return [join(self.namespace, self.task_name)]

    def run(self):
        raise NotImplementedError()


class TestTask(Task):
    def __init__(self, args, namespace):
        super().__init__(args, namespace, 'test_task')

    def get_input_paths(self):
        return ['test_namespace/other_task']

    def run(self):
        out_path = self.get_local_path('out')
        safe_makedirs(out_path)

        for i in range(2):
            number_path = join(out_path, 'number{}.txt'.format(i))
            with open(number_path, 'w') as number_file:
                number_file.write(str(i))


if __name__ == '__main__':
    args = 5
    namespace = 'test_namespace'
    TestTask(args, namespace)()
