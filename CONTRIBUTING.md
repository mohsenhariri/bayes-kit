# Contributing to Scorio

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mohsenhariri/scorio.git
cd scorio
```

2. Create a virtual environment:

**Using uv:**
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

**Using venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

3. Install with dev dependencies:

**Using uv:**
```bash
uv pip install -e ".[dev]"
```

**Using pip:**
```bash
pip install -e ".[dev]"
```

### Dependencies

**Runtime:**
- numpy, scipy

**Development:**
- pytest, ruff, mypy, build, twine, sphinx

## Code Style

- Follow PEP 8
- Format and lint with Ruff
- Type check with mypy

```bash
ruff format scorio/
ruff check --fix scorio/
mypy scorio/
```

## Testing

```bash
pytest
```

## Docstrings

- Use Google-style docstrings
- Document all public APIs
- Include type hints

## Documentation

Build docs locally:
```bash
make docs
```

Full documentation: https://scorio.readthedocs.io/
