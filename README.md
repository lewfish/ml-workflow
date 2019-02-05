# ML Workflow Playground (*Deprecated*)

![](https://img.shields.io/badge/license-MIT-blue.svg)
![](https://img.shields.io/badge/status-experimental-orange.svg)

This repo is a playground for implementing machine learning workflows which
contain DAGs of dependent tasks. It's inspired by ideas from the blog post
[Patterns for Research in Machine
Learning](http://arkitus.com/patterns-for-research-in-machine-learning/) and the
[FBLearner
Flow](https://code.facebook.com/posts/1072626246134461/introducing-fblearner-flow-facebook-s-ai-backbone/)
project, as well as personal experience working on the [Raster
Vision](https://github.com/azavea/raster-vision) project. To make things
concrete yet simple, this workflow implements object recognition using neural
networks via [PyTorch](http://pytorch.org/) on the MNIST and CIFAR10 datasets.
Hopefully, this repo will be a good starting point for new projects, even if
they use a different framework. The goals are to make it easy to:
* add and switch between problem types, datasets, model architectures, and hyperparameters
* run complex workflows and cache computations shared between tasks
* switch between running locally and in the cloud
* test workflows in a pragmatic fashion

## Getting started
Install Docker and run `./scripts/update` which will build a Docker container.
Create a directory to hold datasets and results and set it as an environment
variable called `DATA_DIR`. Then, run `./scripts/run` to run the container. To
run an example task, run the following which will generate
`$DATA_DIR/results/recog/plot_data/mnist_default_train.png`

```
python -m pt.recog.tasks.plot_data --namespace recog --dataset mnist \
  --split train --nimages 12
```

To run an entire Makefile-based workflow, run the following which will generate
a set of directories inside `$DATA_DIR/results/recog/mnist_tiny`. Each
subdirectory will contain the output of a task.

```
./workflows/recog/tests/mnist_tiny all
```

## TODO
- [x] Implement object recognition workflow using command-line arguments and Makefiles
- [ ] Testing with code coverage tool
- [ ] Run in cloud using AWS Batch
- [ ] Switch to using Luigi or Airflow
- [ ] Support hyperparameter sweeps, ensembling and multiple post-processing tasks
