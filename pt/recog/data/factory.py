import torch
import torchvision

from pt.common.settings import datasets_path, TRAIN, TEST

# dataset names
MNIST = 'mnist'
CIFAR10 = 'cifar10'

# loader types
DEFAULT = 'default'

# transform types
NORMALIZE = 'normalize'
HFLIP = 'hflip'


def get_transform(channel_stats, transform_names):
    transforms = []

    if HFLIP in transform_names:
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    transforms.append(torchvision.transforms.ToTensor())
    if NORMALIZE in transform_names:
        transforms.append(torchvision.transforms.Normalize(*channel_stats))

    return torchvision.transforms.Compose(transforms)


class MNISTDataset(torchvision.datasets.MNIST):
    def get_label(self, idx):
        return str(idx)

    def get_shape():
        return (1, 28, 28)


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def get_label(self, idx):
        labels = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
            'frog', 'horse', 'ship', 'truck']
        return labels[idx]

    def get_shape():
        return (3, 32, 32)


def get_data_loader(dataset_name, loader_name=DEFAULT,
                    batch_size=32, shuffle=False, split=TRAIN,
                    transform_names=[NORMALIZE], cuda=True):
    # TODO figure out what this means
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train = True if split == TRAIN else False

    if dataset_name == MNIST:
        if split == TEST:
            raise ValueError('No test set for MNIST')
        channel_stats = ((0.1307,), (0.3081,))
        transform = get_transform(channel_stats, transform_names)
        dataset = MNISTDataset(
            datasets_path, train=train, download=True, transform=transform)
    elif dataset_name == CIFAR10:
        if split == TEST:
            raise ValueError('No test set for CIFAR10')
        channel_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform = get_transform(channel_stats, transform_names)
        dataset = CIFAR10Dataset(
            datasets_path, train=train, download=True, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return loader
