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
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```
