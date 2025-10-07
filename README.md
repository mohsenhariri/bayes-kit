# Bayes-Kit

[![arXiv](https://img.shields.io/badge/arXiv-2504.11651-b31b1b.svg)](https://arxiv.org/abs/2504.11651)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Julia 1.6+](https://img.shields.io/badge/julia-1.6+-9558B2.svg)](https://julialang.org/downloads/)

---

## 📦 Packages

This repository contains two packages:

1. **`scorio`** (Python) - Python implementation
2. **`Scorio.jl`** (Julia) - Julia implementation

---

## 🚀 Quick Start

### Python (scorio)

#### Installation

```bash
# Install from PyPI
pip install scorio

# Install from repository
pip install -e .

```

#### Basic Usage

```python
import numpy as np
from scorio import eval

# Outcomes R: shape (M, N) with integer categories in {0, ..., C}
R = np.array([[0, 1, 2, 2, 1],
              [1, 1, 0, 2, 2]])

# Rubric weights w: length C+1
# Here: 0=incorrect(0.0), 1=partial(0.5), 2=correct(1.0)
w = np.array([0.0, 0.5, 1.0])

# Optional prior outcomes R0: shape (M, D)
R0 = np.array([[0, 2],
               [1, 2]])

# Bayesian evaluation with prior
mu, sigma = eval.bayes(R, w, R0)
print(f"μ = {mu:.6f}, σ = {sigma:.6f}")
# Expected: μ ≈ 0.575, σ ≈ 0.084275

# Bayesian evaluation without prior
mu2, sigma2 = eval.bayes(R, w)
print(f"μ = {mu2:.6f}, σ = {sigma2:.6f}")
# Expected: μ ≈ 0.5625, σ ≈ 0.091998

# Simple average
accuracy = eval.avg(R)
print(f"Average: {accuracy:.6f}")
```

### Julia (Scorio.jl)

#### Installation

```julia
using Pkg

# From local development
Pkg.develop(path="./julia/Scorio.jl")

# Or from Julia General Registry (comming soon)
# Pkg.add("Scorio")
```

#### Basic Usage

```julia
using Scorio

# Outcomes R: shape (M, N) with integer categories in {0, ..., C}
R = [0 1 2 2 1;
     1 1 0 2 2]

# Rubric weights w: length C+1
# Here: 0=incorrect(0.0), 1=partial(0.5), 2=correct(1.0)
w = [0.0, 0.5, 1.0]

# Optional prior outcomes R0: shape (M, D)
R0 = [0 2;
      1 2]

# Bayesian evaluation with prior
mu, sigma = bayes(R, w, R0)
println("μ = $mu, σ = $sigma")
# Expected: μ ≈ 0.575, σ ≈ 0.084275

# Bayesian evaluation without prior
mu2, sigma2 = bayes(R, w)
println("μ = $mu2, σ = $sigma2")
# Expected: μ ≈ 0.5625, σ ≈ 0.091998

# Simple average
accuracy = avg(R)
println("Average: $accuracy")
```

---


### Evaluation Functions

#### `bayes(R, w, R0=None)`
Bayesian performance evaluation with uncertainty quantification using the Bayes@N framework.

- **`R`**: `M × N` integer matrix with entries in `{0, ..., C}` (outcomes for M systems over N trials)
- **`w`**: length `C+1` float vector of rubric weights mapping categories to scores
- **`R0`** (optional): `M × D` integer matrix of prior outcomes
- **Returns**: `(mu, sigma)` - posterior estimate and uncertainty


## Data and Shape Conventions

- **Categories**: Encode outcomes per trial as integers in `{0, ..., C}`
- **Weights**: Choose rubric weights `w` of length `C+1` (e.g., `[0, 1]` for binary outcomes)
- **Shapes**: 
  - `R` is `M × N` (M systems, N trials)
  - `R0` is `M × D` (M systems, D prior trials)
  - Both must share the same `M` and category set

---

## 📝 Requirements

### Python
- Python 3.9 - 3.13
- NumPy 2.0+

### Julia
- Julia 1.6 or higher

---

## 📚 Documentation

Full documentation is available at: [https://mohsenhariri.github.io/bayes-kit/](https://mohsenhariri.github.io/bayes-kit/)

---

## 📄 Citation

If you use Bayes-Kit in your research, please cite:

```bibtex
@article{bayeskit2025,
  title={Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author={Hariri, Mohsen and Samandar, Amirhossein},
  journal={arXiv preprint arXiv:2504.11651},
  year={2025}
}
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Homepage**: [https://mohsenhariri.github.io/bayes-kit/](https://mohsenhariri.github.io/bayes-kit/)
- **Repository**: [https://github.com/mohsenhariri/bayes-kit](https://github.com/mohsenhariri/bayes-kit)
- **Issues**: [https://github.com/mohsenhariri/bayes-kit/issues](https://github.com/mohsenhariri/bayes-kit/issues)
- **Paper**: [https://arxiv.org/abs/2504.11651](https://arxiv.org/abs/2504.11651)


