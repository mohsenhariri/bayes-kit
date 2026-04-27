# Scorio.jl

```@raw html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img src="assets/logo.svg" alt="Scorio" width="220">
  </picture>
</p>
```

Scorio.jl is a Bayesian evaluation and ranking toolkit for comparing LLMs.

It provides:
- Evaluation metrics on `(M, N)` outcomes (for example `bayes`, `pass_at_k`, `mg_pass_at_k`)
- Ranking methods on `(L, M, N)` response tensors across multiple families:
  paired-comparison, Bayesian, voting, IRT, graph, and listwise models
- Tie-aware score-to-rank utilities (`competition_ranks_from_scores`, `rank_scores`)

## Installation

```julia
using Pkg
Pkg.add("Scorio")
```

## Quick Start

```julia
using Scorio

# Evaluation: R is (M, N)
R_eval = [0 1 2 2 1;
          1 1 0 2 2]
w = [0.0, 0.5, 1.0]
mu, sigma = bayes(R_eval, w)
println("Bayes score = ", mu, ", uncertainty = ", sigma)

# Ranking: R is (L, M, N)
R_rank = reshape([
    1 1 0 1 0;
    1 0 0 1 0;
    0 1 0 1 1;
    0 0 0 1 0
], 4, 5, 1)

ranks, scores = Scorio.bradley_terry(R_rank; return_scores=true)
println("Ranks  = ", ranks)
println("Scores = ", scores)
```

## Documentation

- [API Reference](api.md)
- [Examples](examples.md)

## References

See [Citation](@ref) for BibTeX entries for the Bayesian evaluation framework
and ranking methods.

## Module Reference

```@docs
Scorio
```
