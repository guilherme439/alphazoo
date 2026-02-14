# Installation

## Setup

```bash

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install the package in editable mode
pip install -e .

# Install with dev dependencies (testing, linting)
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest test/ -v
```
