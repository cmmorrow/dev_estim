# dev-estim

A small Python library for estimating task durations and delivery probabilities using working days and a simple Bayesian duration model.

## Requirements
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

## Installation
Install dependencies (including test extras) into a local virtual environment managed by `uv`:

```bash
uv sync --group tests
```

If you prefer to install the package into an existing environment:

```bash
uv pip install .
```

## Running tests
Execute the full pytest suite (uses the virtualenv created by `uv sync`):

```bash
uv run -m pytest -p no:cacheprovider
```

You can also invoke pytest directly from the environment:

```bash
source .venv/bin/activate
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider
```
