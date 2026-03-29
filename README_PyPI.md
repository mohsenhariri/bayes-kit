# Scorio

`scorio` implements the Bayes@N framework introduced in [Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](https://arxiv.org/abs/2510.04265).

[![arXiv (Bayes Evaluation)](https://img.shields.io/badge/arXiv-2510.04265-b31b1b.svg)](https://arxiv.org/abs/2510.04265)
[![arXiv (Bayes Ranking)](https://img.shields.io/badge/arXiv-2603.10960-b31b1b.svg)](https://arxiv.org/abs/2603.10960)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/virtual/2026/poster/10009669)
[![PyPI version](https://img.shields.io/pypi/v/scorio.svg)](https://pypi.org/project/scorio/)
[![Python versions](https://img.shields.io/pypi/pyversions/scorio.svg)](https://pypi.org/project/scorio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python Docs](https://readthedocs.org/projects/scorio/badge/?version=latest)](https://scorio.readthedocs.io/en/latest/)

---

## News

- **March 2026**: New paper released: ["Ranking Reasoning LLMs under Test-Time Scaling"](https://arxiv.org/abs/2603.10960)
- **February 2026**: Our paper ["Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation"](https://iclr.cc/virtual/2026/poster/10009669) has been accepted to **ICLR 2026**.
- **February 2026**: Reasoning traces will be released in about 2 weeks.

---

## Installation

```bash
# Install from PyPI
pip install scorio

# Install latest from GitHub
pip install "git+https://github.com/mohsenhariri/scorio.git"

# Install a specific tag
pip install "git+https://github.com/mohsenhariri/scorio.git@v0.2.0"

# Install from local repository
pip install -e .
```

Requires Python 3.10+, NumPy, SciPy.

## Data and shape conventions

- Categories: encode outcomes per trial as integers in `{0, ..., C}`.
- Weights: choose rubric weights `w` of length `C+1` (e.g., `[0, 1]` for binary outcomes).
- Shapes: `R` is `M x N`, `R0` is `M x D` (if provided); both must share the same `M` and category set.

## APIs

- `scorio.eval.bayes(R, w, R0=None) -> (mu: float, sigma: float)`
  - `R`: `M x N` int array with entries in `{0, ..., C}`
  - `w`: length `C+1` float array of rubric weights
  - `R0` (optional): `M x D` int array of prior outcomes (same category set as `R`)
  - Returns posterior estimate `mu` of the rubric-weighted performance and uncertainty `sigma`.
- `scorio.eval.avg(R) -> float`
  - Returns the naive mean of elements in `R`. For binary accuracy, encode incorrect=0 and correct=1.

## How to use

```python
import numpy as np
from scorio import eval

# Outcomes R: shape (M, N) with integer categories in {0, ..., C}
R = np.array([[0, 1, 2, 2, 1], [1, 1, 0, 2, 2]])

# Rubric weights w: length C+1
# Here: 0=incorrect(0.0), 1=partial(0.5), 2=correct(1.0)
w = np.array([0.0, 0.5, 1.0])

# Optional prior outcomes R0: shape (M, D)
R0 = np.array([[0, 2], [1, 2]])

# Bayesian evaluation with prior
mu, sigma = eval.bayes(R, w, R0)
print(f"mu = {mu:.6f}, sigma = {sigma:.6f}")
# Expected: mu ~ 0.575, sigma ~ 0.084275

# Bayesian evaluation without prior
mu2, sigma2 = eval.bayes(R, w)
print(f"mu = {mu2:.6f}, sigma = {sigma2:.6f}")
# Expected: mu ~ 0.5625, sigma ~ 0.091998

# Simple average
accuracy = eval.avg(R)
print(f"Average: {accuracy:.6f}")
```

## Citing

If you use `scorio` in your research, please cite:

```bibtex
@inproceedings{hariri2026don,
  title={Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author={Hariri, Mohsen and Samandar, Amirhossein and Hinczewski, Michael and Chaudhary, Vipin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2510.04265}
}
```

```bibtex
@article{hariri2026ranking,
  title={Ranking Reasoning LLMs under Test-Time Scaling},
  author={Hariri, Mohsen and Hinczewski, Michael and Ma, Jing and Chaudhary, Vipin},
  journal={arXiv preprint arXiv:2603.10960},
  year={2026},
  doi={10.48550/arXiv.2603.10960},
  url={https://arxiv.org/abs/2603.10960}
}
```

## License

MIT License. See the `LICENSE` file for details.

## Links

- Landing page: https://mohsenhariri.github.io/scorio/
- Documentation: https://scorio.readthedocs.io/en/latest/
- Repository: https://github.com/mohsenhariri/scorio
- Issues: https://github.com/mohsenhariri/scorio/issues
- ICLR 2026 poster: https://iclr.cc/virtual/2026/poster/10009669
