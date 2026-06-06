# Installation

## Prerequisites

### 1. GPU drivers (AMD only)

PyTorch's AMD build runs on the system ROCm stack - the `amdgpu` kernel driver and the ROCm runtime. Install it with [AMD's ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for your distribution, then confirm the installed version:

```bash
cat /opt/rocm/.info/version
```

That determines the version need in the PyTorch installation step (ex: `7.2.4` -> `rocm7.2`)


### 2. uv (recommended)

Alphazoo uses Python 3.14. You will want to create a virtual env with this version of python independently of your system's Python.  
The [uv](https://docs.astral.sh/uv/) package manager makes this very easy. You can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

### 1. Create and activate the virtual environment


```bash
uv venv --seed --python 3.14
source .venv/bin/activate
```


### 2. Install PyTorch (AMD only)

Install PyTorch before alphazoo so that you get the correct version and not the default one.

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

The [PyTorch install selector](https://pytorch.org/get-started/locally/) lists which ROCm versions currently ship wheels.

### 3. Install alphazoo

```bash
# to use the package
pip install .

# for development - editable install with test tooling
pip install -e . --group dev
```

