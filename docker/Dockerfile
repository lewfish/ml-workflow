# Adapted from https://github.com/pytorch/pytorch/blob/6626881e7a9d430bd37bd67057ba74b6669a7ccf/Dockerfile
FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.20
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         unzip \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
         libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 && \
     rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

RUN conda install -y pytorch torchvision cuda80 -c soumith
RUN conda install -y scikit-learn scikit-image six h5py \
        matplotlib pillow jupyter && \
    conda clean -yt

RUN pip install flake8 boto3 networkx

RUN mkdir /opt/data && chmod -R a+w /opt/data
RUN mkdir /opt/src && chmod -R a+w /opt/src
WORKDIR /opt/src
ENV PYTHONPATH /opt/src:$PYTHONPATH

# Needed for running on GPU without nvidia-docker
COPY gpu_startup.sh /usr/local/bin/
