"""
    Scorio

A Julia package implementing the Bayes@N framework for evaluating Large Language Models and other systems.

Based on the paper: "Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation"
https://arxiv.org/abs/2504.11651

# Main APIs

## Evaluation Functions (from `eval` submodule)
- `bayes(R, w, R0=nothing)`: Bayesian performance evaluation with uncertainty quantification
- `avg(R)`: Simple average of outcomes

## Ranking Functions (from `rank` submodule)
- `elo()`: ELO ranking (not yet implemented)

## Utility Functions
- `competition_ranks_from_scores(scores)`: Compute competition ranks from scores

# Example Usage

```julia
using Scorio

# Outcomes R: shape (M, N) with integer categories in {0, ..., C}
R = [0 1 2 2 1;
     1 1 0 2 2]

# Rubric weights w: length C+1. Here: 0=incorrect, 1=partial(0.5), 2=correct(1.0)
w = [0.0, 0.5, 1.0]

# Optional prior outcomes R0: shape (M, D). If omitted, D=0.
R0 = [0 2;
      1 2]

# With prior (D=2 → T=10)
mu, sigma = bayes(R, w, R0)
println("μ = \$mu, σ = \$sigma")  # Expected: μ ≈ 0.575, σ ≈ 0.084275

# Without prior (D=0 → T=8)
mu2, sigma2 = bayes(R, w)
println("μ = \$mu2, σ = \$sigma2")  # Expected: μ ≈ 0.5625, σ ≈ 0.091998

# Simple average
accuracy = avg(R)
println("Average: \$accuracy")
```

# Installation

Once registered, install with:
```julia
using Pkg
Pkg.add("Scorio")
```

For development installation:
```julia
using Pkg
Pkg.develop(path="/path/to/Scorio.jl")
```
"""
module Scorio

# Version
const VERSION = v"0.1.0"

# Include submodules
include("eval.jl")
include("rank.jl")
include("utils.jl")

# Re-export main APIs
export bayes, avg, pass_at  # from eval.jl
export elo  # from rank.jl
export competition_ranks_from_scores  # from utils.jl

end # module Scorio
