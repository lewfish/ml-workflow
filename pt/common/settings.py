from os import environ
from os.path import join

workspace_path = '/opt/data/'
datasets_path = join(workspace_path, 'datasets')
results_path = join(workspace_path, 'results')

s3_bucket = environ.get('S3_BUCKET')
s3_datasets_path = join('s3://{}'.format(s3_bucket), 'datasets')
s3_results_path = join('s3://{}'.format(s3_bucket), 'results')

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
split_names = [TRAIN, VAL, TEST]
