from os.path import join, exists

import boto3
import botocore

from pt.common.utils import safe_makedirs
from pt.common.settings import (
    is_remote, results_path, s3_results_dir, s3_bucket)


def local_path_exists(path):
    local_path = join(results_path, path)
    return exists(local_path)


def s3_download(path):
    s3 = boto3.resource('s3')
    key = join(s3_results_dir, path)
    local_path = join(results_path, path)
    try:
        s3.meta.client.download_file(s3_bucket, key, local_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print('The object {} does not exist.'.format(key))
        else:
            raise


def s3_upload(path):
    s3 = boto3.resource('s3')
    key = join(s3_results_dir, path)
    local_path = join(results_path, path)
    try:
        s3.meta.client.upload_file(local_path, s3_bucket, key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print('The object {} does not exist.'.format(key))
        else:
            raise


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
            if not local_path_exists(input_path):
                s3_download(input_path)

    def _is_runnable(self):
        input_paths = self.get_input_paths()
        for input_path in input_paths:
            if not local_path_exists(input_path):
                return False
        return True

    def _upload_outputs(self):
        output_paths = self.get_output_paths()
        for output_path in output_paths:
            s3_upload(output_path)

    def get_input_paths(self):
        return []

    def get_output_paths(self):
        return []

    def run(self):
        raise NotImplementedError()

    def __call__(self):
        self._make_task_dir()
        if is_remote:
            self._download_inputs()
        if self._is_runnable():
            self.run()
            if is_remote:
                self._upload_outputs()
        else:
            raise ValueError('Task {} is not runnable'.format(self.task_name))


class TestTask(Task):
    def __init__(self, args, namespace):
        super().__init__(args, namespace, 'test_task')

    def get_input_paths(self):
        return ['test_namespace/other_task/out.txt']

    def get_output_paths(self):
        paths = ['out.txt']
        return map(lambda path: join(self.task_path, path), paths)

    def run(self):
        out_path = join(self.task_path, 'out.txt')
        with open(out_path, 'w') as out_file:
            out_file.write(str(self.args))


if __name__ == '__main__':
    args = 5
    namespace = 'test_namespace'
    TestTask(args, namespace)()
