<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mohsenhariri/scorio/main/assets/scorio-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mohsenhariri/scorio/main/assets/scorio.svg">
    <img src="https://raw.githubusercontent.com/mohsenhariri/scorio/main/assets/scorio.svg" alt="Scorio" width="240">
  </picture>
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2510.04265"><img alt="ICLR 2026" src="https://img.shields.io/badge/ICLR-2026-blue.svg"></a>
  <a href="https://arxiv.org/abs/2603.10960"><img alt="ACL 2026" src="https://img.shields.io/badge/ACL-2026-blue.svg"></a>
  <a href="https://arxiv.org/abs/2608.04001"><img alt="arXiv: Test-Time Scaling" src="https://img.shields.io/badge/arXiv-2608.04001-b31b1b.svg"></a>
  <a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://pypi.org/project/scorio/"><img alt="PyPI package" src="https://img.shields.io/pypi/v/scorio.svg"></a>
  <a href="https://julialang.org/downloads/"><img alt="Julia 1.6+" src="https://img.shields.io/badge/julia-1.6+-9558B2.svg"></a>
  <a href="https://www.npmjs.com/package/scorio"><img alt="npm package" src="https://img.shields.io/npm/v/scorio.svg"></a>
  <a href="https://scorio.readthedocs.io/en/latest/"><img alt="Python Docs" src="https://readthedocs.org/projects/scorio/badge/?version=latest"></a>
  <a href="https://mohsenhariri.github.io/scorio/julia/"><img alt="Julia Docs" src="https://img.shields.io/badge/docs-Julia-9558B2.svg"></a>
</p>

---

## Documentation

[mohsenhariri.github.io/scorio](https://mohsenhariri.github.io/scorio/)

| APIs | Documentation | Status |
|----------|--------------|--------|
| **Python** | [scorio.readthedocs.io](https://scorio.readthedocs.io/en/latest/) | [![ReadTheDocs](https://readthedocs.org/projects/scorio/badge/?version=latest)](https://scorio.readthedocs.io/en/latest/) |
| **Julia** | [mohsenhariri.github.io/scorio/julia](https://mohsenhariri.github.io/scorio/julia/) | [![GitHub Pages](https://img.shields.io/badge/docs-stable-blue.svg)](https://mohsenhariri.github.io/scorio/julia/) |

---

## News

- **August 2026**: Preprint of our paper ["Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility"](https://arxiv.org/abs/2608.04001) is available. We will release reasoning traces and evaluation scripts soon.

- **July 2026** 📦: [Scorio](https://www.npmjs.com/package/scorio) is now available on **npm**. Read [our blog post](https://mohsenhariri.github.io/posts.html).

- **April 2026** 🎉: Our ranking paper ["Ranking Reasoning LLMs under Test-Time Scaling"](https://aclanthology.org/2026.acl-long.1544/) has been accepted to **ACL 2026 Main Conference**!

- **January 2026** 🎉: Our paper ["Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation"](https://iclr.cc/virtual/2026/poster/10009669) has been accepted to **ICLR 2026**!

---

## Packages

This repository contains three packages:

1. **[`scorio`](https://github.com/mohsenhariri/scorio/tree/main/scorio)** - Python implementation
2. **[`Scorio.jl`](https://github.com/mohsenhariri/scorio/tree/main/julia/Scorio.jl)** - Julia implementation
3. **[`scorio`](https://github.com/mohsenhariri/scorio/tree/main/js)** - JS/TS implementation

---

## Quick Start

### Python (scorio)

#### Installation

```bash
# Install from PyPI
pip install scorio

# Install latest from GitHub
pip install "git+https://github.com/mohsenhariri/scorio.git"

# Install a specific tag
pip install "git+https://github.com/mohsenhariri/scorio.git@v0.2.2"

# Install from local repository
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

# Weighted average
accuracy, accuracy_sigma = eval.avg(R, w)
print(f"Average = {accuracy:.6f}, σ = {accuracy_sigma:.6f}")
```

### Julia (Scorio.jl)

#### Installation

```julia
using Pkg

# From local development
Pkg.develop(path="./julia/Scorio.jl")

# Or from Julia General Registry
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

# Weighted average
accuracy, accuracy_sigma = avg(R, w)
println("Average = $accuracy, σ = $accuracy_sigma")
```

---


### Evaluation Functions

#### `bayes(R, w, R0=None)`
Bayesian performance evaluation with uncertainty quantification using the Bayes@N framework.

- **`R`**: `M × N` integer matrix with entries in `{0, ..., C}` (outcomes for M questions over N trials)
- **`w`**: length `C+1` float vector of rubric weights mapping categories to scores
- **`R0`** (optional): `M × D` integer matrix of prior outcomes
- **Returns**: `(mu, sigma)` - posterior estimate and uncertainty


## Data and Shape Conventions

- **Categories**: Encode outcomes per trial as integers in `{0, ..., C}`
- **Weights**: Choose rubric weights `w` of length `C+1` (e.g., `[0, 1]` for binary outcomes)
- **Shapes**:
  - `R` is `M × N` (M questions, N trials)
  - `R0` is `M × D` (M questions, D prior trials)
  - Both must share the same `M` and category set

---

## Requirements

### Python
- Python 3.10+
- NumPy 2.0+

### Julia
- Julia 1.6 or higher

### JavaScript
- Node.js 18 or higher

---

## Citation

If you use Scorio in your research, please cite the relevant papers:

### Bayesian Evaluation Framework

```bibtex
@inproceedings{hariri2026dont,
  title={Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author={Hariri, Mohsen and Samandar, Amirhossein and Hinczewski, Michael and Chaudhary, Vipin},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://proceedings.iclr.cc/paper_files/paper/2026/file/f04edfc65463d020629673a4bc4c58e7-Paper-Conference.pdf},
  eprint={2510.04265},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  note={Latest version available at \url{https://arxiv.org/abs/2510.04265}}
}
```

### Ranking Methods

```bibtex
@inproceedings{hariri2026ranking,
  title={Ranking Reasoning {LLM}s under Test-Time Scaling},
  author={Hariri, Mohsen and Hinczewski, Michael and Ma, Jing and Chaudhary, Vipin},
  booktitle={Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  year={2026},
  pages={33437--33478},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2026.acl-long.1544},
  url={https://aclanthology.org/2026.acl-long.1544/},
  eprint={2603.10960},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  note={Latest version available at \url{https://arxiv.org/abs/2603.10960}}
}
```


### Aggregation Methods

```bibtex
@misc{hariri2026testtime,
  title         = {Test-Time Scaling in Reasoning {LLM}s: Inference Regimes, Evaluation, and Reproducibility},
  author        = {Hariri, Mohsen and Chen, Weicong and Shahini, Nahal and Singh, Vikash and Ye, Kai and Samandar, Amirhossein and Ganguly, Debargha and Sankar, Sreehari and Zhang, Yanyan and Wang, Shouren and Peng, Jerry and Zhang, Biyao and Hinczewski, Michael and Chaudhary, Vipin},
  year          = {2026},
  eprint        = {2608.04001},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2608.04001},
  url           = {https://arxiv.org/abs/2608.04001}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/mohsenhariri/scorio/blob/main/LICENSE) file for details.

---

## Links

- **Landing Page**: [mohsenhariri.github.io/scorio](https://mohsenhariri.github.io/scorio/)
- **Python Docs**: [scorio.readthedocs.io](https://scorio.readthedocs.io/en/latest/)
- **Julia Docs**: [mohsenhariri.github.io/scorio/julia](https://mohsenhariri.github.io/scorio/julia/)
- **Repository**: [github.com/mohsenhariri/scorio](https://github.com/mohsenhariri/scorio)
- **Issues**: [github.com/mohsenhariri/scorio/issues](https://github.com/mohsenhariri/scorio/issues)
- **Papers**:
  - [Don't Pass@k (ICLR 2026)](https://iclr.cc/virtual/2026/poster/10009669) | [arXiv](https://arxiv.org/abs/2510.04265)
  - [Ranking Reasoning LLMs (ACL 2026)](https://aclanthology.org/2026.acl-long.1544/) | [arXiv](https://arxiv.org/abs/2603.10960)
  - [Test-Time Scaling in Reasoning LLMs](https://arxiv.org/abs/2608.04001)
