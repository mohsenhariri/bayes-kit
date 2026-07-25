"""
Scalar evaluation metrics and uncertainty estimators.

This module implements evaluation methods for outcome matrices `R` used in
Scorio evaluation workflows.

Notation
--------
Let ``R in {0, ..., C}^{M x N}`` be an outcome matrix.

- `M` is the number of questions.
- `N` is the number of trials per question.
- `R[a, i]` is the outcome for question `a` on trial `i`.

Binary metrics use ``R in {0,1}^{M x N}``.

Return Pattern
--------------
- `avg` and `bayes` return `(mu, sigma)`; the remaining point estimators
  return a scalar score, matching Python.
- Companion `*_ci` functions return `(mu, sigma, lo, hi)` where:
  - `mu` is the estimated score,
  - `sigma` is the posterior standard deviation under method assumptions,
  - `lo, hi` are normal-approximation credible interval bounds.

Available Families
------------------
- Bayes family: `bayes`, `bayes_ci`.
- Avg family: `avg`, `avg_ci`.
- Pass family: `pass_at_k`, `pass_hat_k`, `g_pass_at_k`, `g_pass_at_k_tau`,
  `mg_pass_at_k`, and their `*_ci` variants.
- AUC, majority, max-reward, and geometric/threshold-spectrum families.
"""

include("eval/utils.jl")
include("eval/bayes.jl")
include("eval/avg.jl")
include("eval/pass_at_k.jl")
include("eval/auc.jl")
include("eval/maj.jl")
include("eval/max_reward.jl")
include("eval/geom.jl")
include("eval/keyword_forwarding.jl")

const unanimous_at_k = pass_hat_k
const unanimous_at_k_ci = pass_hat_k_ci

export bayes,
    bayes_ci,
    avg,
    avg_ci,
    pass_at_k,
    pass_at_k_ci,
    pass_hat_k,
    pass_hat_k_ci,
    g_pass_at_k,
    g_pass_at_k_ci,
    g_pass_at_k_tau,
    g_pass_at_k_tau_ci,
    mg_pass_at_k,
    mg_pass_at_k_ci,
    unanimous_at_k,
    unanimous_at_k_ci,
    auc_at_k,
    auc_at_k_ci,
    maj_at_k,
    maj_at_k_ci,
    max_at_k,
    max_at_k_ci,
    threshold_spectrum_at_k,
    threshold_spectrum_at_k_ci,
    geom_at_k,
    geom_at_k_ci,
    geom_ds_at_k,
    geom_ds_at_k_ci,
    geo_spectrum_at_k,
    geo_spectrum_at_k_ci,
    geo_spectrum_star_at_k,
    geo_spectrum_star_at_k_ci
