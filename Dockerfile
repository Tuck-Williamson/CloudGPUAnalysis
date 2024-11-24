FROM quay.io/jupyter/scipy-notebook
LABEL authors="tuckw"

USER ${NB_UID}

RUN conda install jupytext -c conda-forge && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN mamba install --yes \
    'quarto' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN python -m pip install jupyterlab-quarto && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN mamba run quarto install chromium --no-prompt && \
    mamba run quarto install tinytex --no-prompt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER root
RUN apt-get update --yes && \
    apt-get install --yes  \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libgtk-3-common \
    libatk-adaptor && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

ENV QPORT=10001
EXPOSE $QPORT

WORKDIR "${HOME}/work"

COPY Quarto.md "${HOME}"
# docker run -it --rm --volume c:/Users/tuckw/IdeaProjects/pubtest/:/home/jovyan/work -u root -p 8888:8888 -p 10001:10001 tuck-williamson/test-pub

#ENTRYPOINT ["top", "-b"]

## Copyright (c) Jupyter Development Team.
## Distributed under the terms of the Modified BSD License.
#ARG REGISTRY=quay.io
#ARG OWNER=jupyter
#ARG BASE_IMAGE=$REGISTRY/$OWNER/minimal-notebook
#FROM $BASE_IMAGE
#
#LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"
#
## Fix: https://github.com/hadolint/hadolint/wiki/DL4006
## Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
#SHELL ["/bin/bash", "-o", "pipefail", "-c"]
#
#USER root
#
#RUN apt-get update --yes && \
#    apt-get install --yes --no-install-recommends \
#    # for cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
#    build-essential \
#    # for latex labels
#    cm-super \
#    dvipng \
#    # for matplotlib anim
#    ffmpeg && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*
#
#USER ${NB_UID}
#
## Install Python 3 packages
#RUN mamba install --yes \
#    'altair' \
#    'beautifulsoup4' \
#    'bokeh' \
#    'bottleneck' \
#    'cloudpickle' \
#    'conda-forge::blas=*=openblas' \
#    'cython' \
#    'dask' \
#    'dill' \
#    'h5py' \
#    'ipympl' \
#    'ipywidgets' \
#    'jupyterlab-git' \
#    'matplotlib-base' \
#    'numba' \
#    'numexpr' \
#    'openpyxl' \
#    'pandas' \
#    'patsy' \
#    'protobuf' \
#    'pytables' \
#    'scikit-image' \
#    'scikit-learn' \
#    'scipy' \
#    'seaborn' \
#    'sqlalchemy' \
#    'statsmodels' \
#    'sympy' \
#    'widgetsnbextension' \
#    'xlrd' && \
#    mamba clean --all -f -y && \
#    fix-permissions "${CONDA_DIR}" && \
#    fix-permissions "/home/${NB_USER}"
#
## Install facets package which does not have a `pip` or `conda-forge` package at the moment
#WORKDIR /tmp
#RUN git clone https://github.com/PAIR-code/facets && \
#    jupyter nbclassic-extension install facets/facets-dist/ --sys-prefix && \
#    rm -rf /tmp/facets && \
#    fix-permissions "${CONDA_DIR}" && \
#    fix-permissions "/home/${NB_USER}"
#
## Import matplotlib the first time to build the font cache
#RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
#    fix-permissions "/home/${NB_USER}"
#
#USER ${NB_UID}
#
#WORKDIR "${HOME}"