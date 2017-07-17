from os import environ
from os.path import join

workspace_path = '/opt/data/'
datasets_path = join(workspace_path, 'datasets')
results_path = join(workspace_path, 'results')

is_remote = environ.get('IS_REMOTE')
s3_bucket = environ.get('S3_BUCKET')
s3_datasets_dir = 'datasets'
s3_results_dir = 'results'

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
split_names = [TRAIN, VAL, TEST]
