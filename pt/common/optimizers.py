import torch.optim as optim

SGD = 'sgd'


def get_optimizer(optimizer_name, model_params, lr=1e-3, momentum=None):
    if optimizer_name == SGD:
        optimizer = optim.SGD(
            model_params, lr=lr, momentum=momentum)

    return optimizer
