from os import environ
from os.path import join

workspace_path = '/opt/data/'
datasets_path = join(workspace_path, 'datasets')
results_path = join(workspace_path, 'results')

is_remote = environ.get('IS_REMOTE')
s3_bucket = environ.get('S3_BUCKET')
s3_datasets_path = 's3://{}/datasets'.format(s3_bucket)
s3_results_path = 's3://{}/results'.format(s3_bucket)

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
split_names = [TRAIN, VAL, TEST]
