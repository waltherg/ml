# Some machine learning snippets

## Setup virtual environment

Create a new virtual environment with conda:

```bash
$ conda create --name ml python=3.11
```

Activate the virtual environment:

```bash
$ conda activate ml
```

Install the required packages:

```bash
$ pip install -r requirements.txt
```

## Install PyTorch with MPS support on Mac

Select appropriate PIP command from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

```bash
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Install PyTorch Forecasting

```bash
$ pip install --upgrade git+https://github.com/waltherg/pytorch-forecasting
```

## Run MLflow server

To persist experiments and artifacts, let's first create a local directory to store them:

```bash
$ mkdir mlflow_data
$ mkdir mlflow_data/experiments
$ mkdir mlflow_data/artifacts
```

Then, spin up an MLflow server:

```bash
$ docker run \
    --platform linux/amd64 \
    -p 5555:5000 \
    -v $(pwd)/mlflow_data/experiments:/mlflow \
    -v $(pwd)/mlflow_data/artifacts:/mlartifacts \
    ghcr.io/mlflow/mlflow:latest mlflow server \
    --host 0.0.0.0 \
    --backend-store-uri /mlflow
```

Your MLflow server is now running at [http://localhost:5555](http://localhost:5555).
