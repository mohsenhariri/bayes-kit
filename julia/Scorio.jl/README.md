# Scorio.jl

`Scorio.jl` is a Julia implementation of the Bayes@N framework introduced in [Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](https://arxiv.org/abs/2504.11651)

[![arXiv](https://img.shields.io/badge/arXiv-2504.11651-b31b1b.svg)](https://arxiv.org/abs/2504.11651)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

---

## Installation

### From Julia General Registry
```julia
using Pkg
Pkg.add("Scorio")
```

### Development Installation
```julia
using Pkg
Pkg.develop(path="/path/to/Scorio.jl")
```

**Requirements:** Julia 1.6 or higher

## Data and Shape Conventions

- **Categories**: encode outcomes per trial as integers in `{0, ..., C}`.
- **Weights**: choose rubric weights `w` of length `C+1` (e.g., `[0, 1]` for binary R).
- **Shapes**: `R` is `M × N`, `R0` is `M × D` (if provided); both must share the same `M` and category set.

## APIs

### Evaluation Functions

#### `bayes(R, w, R0=nothing) -> (mu::Float64, sigma::Float64)`
- **`R`**: `M × N` integer matrix with entries in `{0, ..., C}`
- **`w`**: length `C+1` float vector of rubric weights
- **`R0`** (optional): `M × D` integer matrix of prior outcomes (same category set as `R`)
- **Returns**: posterior estimate `mu` of the rubric-weighted performance and its uncertainty `sigma`.

#### `avg(R) -> Float64`
- Returns the naive mean of elements in `R`. For binary accuracy, encode incorrect=0, correct=1.

### Ranking Functions

#### `elo()`
- Not yet implemented. Placeholder for future ELO ranking functionality.

### Utility Functions

#### `competition_ranks_from_scores(scores; tol=1e-12) -> Vector{Int}`
- Computes competition ranks from scores, handling ties appropriately.

## Usage Example

```julia
using Scorio

# Outcomes R: shape (M, N) with integer categories in {0, ..., C}
R = [0 1 2 2 1;   # Item 1, N=5 trials
     1 1 0 2 2]   # Item 2, N=5 trials

# Rubric weights w: length C+1. Here: 0=incorrect, 1=partial(0.5), 2=correct(1.0)
w = [0.0, 0.5, 1.0]

# Optional prior outcomes R0: shape (M, D). If omitted, D=0.
R0 = [0 2;
      1 2]

# With prior (D=2 → T=10)
mu, sigma = bayes(R, w, R0)
println("μ = $mu, σ = $sigma")  # Expected: μ ≈ 0.575, σ ≈ 0.084275

# Without prior (D=0 → T=8)
mu2, sigma2 = bayes(R, w)
println("μ = $mu2, σ = $sigma2")  # Expected: μ ≈ 0.5625, σ ≈ 0.091998

# Simple average
accuracy = avg(R)
println("Average: $accuracy")

# Competition ranks
scores = [0.95, 0.87, 0.87, 0.72, 0.65]
ranks = competition_ranks_from_scores(scores)
println("Ranks: $ranks")  # Expected: [1, 2, 2, 4, 5]
```

## Citing

If you use `Scorio.jl` or Bayes@N, please cite:

```bibtex
@article{hariri2025dontpassk,
  title   = {Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author  = {Hariri, Mohsen and Samandar, Amirhossein and Hinczewski, Michael and Chaudhary, Vipin},
  journal = {arXiv preprint arXiv:2504.11651},
  year    = {2025},
  url     = {https://mohsenhariri.github.io/bayes-kit/}
}
```

## License

MIT License. See the `LICENSE` file for details.

## Support

- Documentation and updates: https://mohsenhariri.github.io/bayes-kit/
- Issues and feature requests: https://github.com/mohsenhariri/bayes-kit/issues

## Related Packages

- Python implementation: [`scorio`](https://pypi.org/project/scorio/) (`pip install scorio`)
