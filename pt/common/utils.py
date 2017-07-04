from os import makedirs
import errno
import csv
from os.path import isfile
import json

json.encoder.FLOAT_REPR = lambda o: format(o, '.5f')


def safe_makedirs(path):
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def append_log(log_path, row):
    with open(log_path, 'a') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(row)


def get_last_epoch(log_path):
    with open(log_path, 'r') as log_file:
        log_reader = csv.reader(log_file)
        epoch = 0
        for row_idx, row in enumerate(log_reader):
            # skip header row
            if row_idx > 0:
                epoch = row[0]
        return int(epoch)


def setup_log(log_path):
    if isfile(log_path):
        return get_last_epoch(log_path) + 1

    header = ('epoch', 'test_loss', 'test_acc')
    append_log(log_path, header)
    return 1


def save_json(x, path):
    with open(path, 'w') as f:
        json.dump(x, f)
