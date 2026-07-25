# Scorio.jl

`Scorio.jl` is a Julia implementation of the Bayes@N framework introduced in [Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](https://arxiv.org/abs/2510.04265)

[![arXiv](https://img.shields.io/badge/arXiv-2510.04265-b31b1b.svg)](https://arxiv.org/abs/2510.04265)
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

#### `avg(R, w=nothing) -> (mu::Float64, sigma::Float64)`
- Returns the weighted average and Bayes-scaled uncertainty. For binary
  accuracy, encode incorrect=0 and correct=1; categorical inputs require `w`.

#### `pass_at_k(R, k) -> Float64`
- Unbiased Pass@k estimator. Computes the probability that at least one of k randomly selected samples is correct, averaged over all M questions.

#### `pass_hat_k(R, k) -> Float64`
- Pass^k (Pass-hat@k): probability that all k selected trials are correct.

#### `g_pass_at_k(R, k) -> Float64`
- Alias for `pass_hat_k`.

#### `g_pass_at_k_tau(R, k, tau) -> Float64`
- G-Pass@k_τ: Generalized Pass@k with threshold τ. Computes the probability that at least ⌈τ·k⌉ of k randomly selected samples are correct.

#### `mg_pass_at_k(R, k) -> Float64`
- mG-Pass@k: mean Generalized Pass@k. Computes the mean of G-Pass@k_τ over τ ∈ [0.5, 1.0], providing a comprehensive stability metric.

### Ranking Functions

The complete Python ranking surface is available under `Scorio.Rank`, including
evaluation-based, pairwise, voting, IRT, graph, centrality, and listwise methods.

### Utility Functions

#### `competition_ranks_from_scores(scores; tol=1e-12) -> Vector{Int}`
- Computes competition ranks from scores, handling ties appropriately.

### Aggregation Functions

Test-time scaling APIs are namespaced under `Scorio.Aggregate` (short aliases
`Scorio.Agg` and `Scorio.agg`). They include confidence signals, PRM reduction,
Best-of-N and voting rules, and online stopping:

```julia
answers = ["A", "A", "B"]
scores = [0.3, 0.4, 0.9]

Scorio.Aggregate.majority_vote(answers)            # "A"
Scorio.Aggregate.best_of_n(answers, scores)         # "B"
Scorio.Aggregate.weighted_majority_vote(answers, scores)
```

Candidate indices returned with `return_index=true` match Python: they are
0-based, with `-1` as the sentinel when no answer is valid.

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
average, average_sigma = avg(R, w)
println("Average: $average, uncertainty: $average_sigma")

# Competition ranks
scores = [0.95, 0.87, 0.87, 0.72, 0.65]
ranks = competition_ranks_from_scores(scores)
println("Ranks: $ranks")  # Expected: [1, 2, 2, 4, 5]
```

## Citing

If you use `Scorio.jl` or Scorio in your research, please cite the relevant papers:

### Bayesian Evaluation Framework

```bibtex
@inproceedings{hariri2026don,
  title={Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author={Hariri, Mohsen and Samandar, Amirhossein and Hinczewski, Michael and Chaudhary, Vipin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=PTXi3Ef4sT},
  doi={10.48550/arXiv.2510.04265}
}
```

### Ranking Methods

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

## Support

- Documentation and updates: https://mohsenhariri.github.io/scorio/
- Issues and feature requests: https://github.com/mohsenhariri/scorio/issues

## Related Packages

- Python implementation: [`scorio`](https://pypi.org/project/scorio/) (`pip install scorio`)
