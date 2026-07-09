# Ranking API

## Priors

```@docs
Prior
GaussianPrior
LaplacePrior
CauchyPrior
UniformPrior
CustomPrior
EmpiricalPrior
```

## Evaluation-based Ranking

### Bayes

```@docs; canonical=false
bayes(::AbstractArray{<:Integer, 3}, ::Any; R0, quantile, method, return_scores)
```

### Avg

```@docs; canonical=false
avg(::Any; method, return_scores)
```

### Pass@k Family

```@docs
pass_at_k(::AbstractArray{<:Integer, 3}, ::Any; method, return_scores)
pass_hat_k(::AbstractArray{<:Integer, 3}, ::Any; method, return_scores)
g_pass_at_k_tau(::AbstractArray{<:Integer, 3}, ::Any, ::Any; method, return_scores)
mg_pass_at_k(::AbstractArray{<:Integer, 3}, ::Any; method, return_scores)
```

## Pointwise Methods

```@docs
inverse_difficulty
```

## Pairwise Methods

```@docs
elo
trueskill
glicko
```

## Paired-Comparison Probabilistic Models

```@docs
bradley_terry
bradley_terry_map
bradley_terry_davidson
bradley_terry_davidson_map
rao_kupper
rao_kupper_map
```

## Bayesian Ranking

```@docs
thompson
bayesian_mcmc
```

## Item Response Theory

```@docs
rasch
rasch_map
rasch_2pl
rasch_2pl_map
rasch_3pl
rasch_3pl_map
dynamic_irt
rasch_mml
rasch_mml_credible
mirt
```

## Voting Methods

```@docs
borda
copeland
win_rate
minimax
schulze
ranked_pairs
kemeny_young
nanson
baldwin
majority_judgment
```

## Graph-based Methods

```@docs
pagerank
spectral
alpharank
nash
```

## Centrality and Spectral Variants

```@docs
rank_centrality
serial_rank
hodge_rank
```

## Listwise and Choice Models

```@docs
plackett_luce
plackett_luce_map
davidson_luce
davidson_luce_map
bradley_terry_luce
bradley_terry_luce_map
```
