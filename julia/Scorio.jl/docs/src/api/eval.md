# Evaluation API

Evaluation methods operate on outcome matrices `R` with shape `(M, N)` (or vectors
coerced to `1 x N`).

## Bayes Family

```@docs
bayes
bayes_ci
```

## Avg Family

```@docs
avg
avg_ci
```

## Pass Family (Point Metrics)

```@docs
pass_at_k(::Union{AbstractVector, AbstractMatrix}, ::Integer)
pass_hat_k(::Union{AbstractVector, AbstractMatrix}, ::Integer)
g_pass_at_k(::Union{AbstractVector, AbstractMatrix}, ::Integer)
g_pass_at_k_tau(::Union{AbstractVector, AbstractMatrix}, ::Integer, ::Real)
mg_pass_at_k(::Union{AbstractVector, AbstractMatrix}, ::Integer)
unanimous_at_k
```

## Pass Family (Posterior + CI)

```@docs
pass_at_k_ci
pass_hat_k_ci
g_pass_at_k_ci
g_pass_at_k_tau_ci
mg_pass_at_k_ci
unanimous_at_k_ci
```

## AUC and Majority Families

```@docs
auc_at_k
auc_at_k_ci
maj_at_k
maj_at_k_ci
```

## Max-Reward and Threshold Spectrum

```@docs
max_at_k
max_at_k_ci
threshold_spectrum_at_k
threshold_spectrum_at_k_ci
```

## Geometric Families

```@docs
geom_at_k
geom_at_k_ci
geom_ds_at_k
geom_ds_at_k_ci
geo_spectrum_at_k
geo_spectrum_at_k_ci
geo_spectrum_star_at_k
geo_spectrum_star_at_k_ci
```
