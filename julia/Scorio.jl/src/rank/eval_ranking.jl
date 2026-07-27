"""Eval-metric-based ranking methods."""

using SpecialFunctions: erfcinv

"""
    avg(R; method="competition", return_scores=false)

Rank models by per-model mean accuracy across all questions and trials.

For each model `l`, compute the scalar score:

```math
s_l^{\\mathrm{avg}} = \\frac{1}{MN}\\sum_{m=1}^{M}\\sum_{n=1}^{N} R_{lmn}
```

Higher scores are better; ranking is produced by `rank_scores`.

# Arguments
- `R`: binary response tensor `(L, M, N)` or matrix `(L, M)` promoted to `(L, M, 1)`.
- `method`: tie-handling rule for `rank_scores`.
- `return_scores`: if `true`, return `(ranking, scores)`.
"""
function _rank_avg(R; method="competition", return_scores=false)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        mu, _ = avg(@view Rv[model, :, :])
        scores[model] = mu
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

avg(R::AbstractArray{<:Any,3}; method="competition", return_scores=false) =
    _rank_avg(R; method=method, return_scores=return_scores)

function _norm_ppf(p::Float64)::Float64
    p == 0.0 && return -Inf
    p == 1.0 && return Inf
    return -sqrt(2.0) * erfcinv(2.0 * p)
end

"""
    bayes(
        R::AbstractArray{<:Integer, 3},
        w=nothing;
        R0=nothing,
        quantile=nothing,
        method="competition",
        return_scores=false,
    )

Rank models by Bayes@N scores computed independently per model.

If `quantile` is provided, models are ranked by `mu + z_q * sigma`; otherwise
by posterior mean `mu`.

# References
Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
*The Fourteenth International Conference on Learning Representations*.
https://openreview.net/forum?id=PTXi3Ef4sT

# Formula
For each model `l`, let `(mu_l, sigma_l) = Scorio.bayes(R_l, w, R0_l)`.

```math
s_l =
\\begin{cases}
\\mu_l, & \\text{if quantile is not set} \\\\
\\mu_l + \\Phi^{-1}(q)\\,\\sigma_l, & \\text{if quantile}=q \\in [0,1]
\\end{cases}
```

# Arguments
- `R`: integer tensor `(L, M, N)` with values in `{0, ..., C}`.
- `w`: class weights of length `C+1`. If not provided and R is binary (contains only 0 and 1),
  defaults to `[0.0, 1.0]`. For non-binary R, w is required.
- `R0`: optional shared prior `(M, D)` or model-specific prior `(L, M, D)`.
- `quantile`: optional value in `[0, 1]` for quantile-adjusted ranking.
- `method`, `return_scores`: ranking output controls.
"""
function _rank_bayes(
    R,
    w=nothing;
    R0=nothing,
    quantile=nothing,
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R; binary_only=false)
    L, M, _ = size(Rv)

    z = nothing
    if !isnothing(quantile)
        q = Float64(quantile)
        if !isfinite(q) || !(0.0 < q < 1.0)
            error("quantile must be in (0, 1); got $quantile")
        end
        z = _norm_ppf(q)
    end

    R0_shared = nothing
    R0_per_model = nothing

    if !isnothing(R0)
        raw_R0 = _coerce_rank_array_like(R0)
        if isnothing(raw_R0)
            error("R0 must contain real, finite integer-valued outcomes")
        end
        if !(eltype(raw_R0) <: Number) || eltype(raw_R0) <: Complex ||
           any(x -> !isfinite(x) || x != floor(x), raw_R0)
            error("R0 must contain real, finite integer-valued outcomes")
        end
        R0_arr = Int.(raw_R0)

        if ndims(R0_arr) == 2
            if size(R0_arr, 1) != M
                error("Shared R0 must have shape (M=$M, D), got $(size(R0_arr))")
            end
            R0_shared = R0_arr
        elseif ndims(R0_arr) == 3
            if size(R0_arr, 1) != L || size(R0_arr, 2) != M
                error(
                    "Model-specific R0 must have shape (L=$L, M=$M, D), got $(size(R0_arr))",
                )
            end
            R0_per_model = R0_arr
        else
            error(
                "R0 must be shape (M, D) or (L, M, D); got ndim=$(ndims(R0_arr)) with shape $(size(R0_arr))",
            )
        end
    end

    scores = zeros(Float64, L)
    for model in 1:L
        model_R0 = isnothing(R0_shared) ? nothing : R0_shared
        if !isnothing(R0_per_model)
            model_R0 = @view R0_per_model[model, :, :]
        end

        mu, sigma = bayes(@view(Rv[model, :, :]), w, model_R0)
        scores[model] = isnothing(z) ? mu : (mu + z * sigma)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

function bayes(
    R::AbstractArray{<:Integer,3},
    w=nothing;
    R0=nothing,
    quantile=nothing,
    method="competition",
    return_scores=false,
)
    return _rank_bayes(
        R,
        w;
        R0=R0,
        quantile=quantile,
        method=method,
        return_scores=return_scores,
    )
end

"""
    pass_at_k(R::AbstractArray{<:Integer, 3}, k; method="competition", return_scores=false)

Rank models by per-model Pass@k scores.

For each model `l`, define per-question success counts
``nu_{lm} = \\sum_{n=1}^{N} R_{lmn}``. Then:

```math
s_l^{\\mathrm{Pass@}k}
= \\frac{1}{M}\\sum_{m=1}^{M}
\\left(1 - \\frac{\\binom{N-\\nu_{lm}}{k}}{\\binom{N}{k}}\\right)
```

# References
Chen, M., Tworek, J., Jun, H., et al. (2021).
Evaluating Large Language Models Trained on Code.
*arXiv:2107.03374*. https://arxiv.org/abs/2107.03374
"""
function _rank_pass_at_k(
    R,
    k;
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = pass_at_k(@view(Rv[model, :, :]), k)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

pass_at_k(R::AbstractArray{<:Integer,3}, k; method="competition", return_scores=false) =
    _rank_pass_at_k(R, k; method=method, return_scores=return_scores)

"""
    pass_hat_k(R::AbstractArray{<:Integer, 3}, k; method="competition", return_scores=false)

Rank models by per-model Pass-hat@k (G-Pass@k) scores.

With ``nu_{lm} = \\sum_{n=1}^{N} R_{lmn}``:

```math
s_l^{\\widehat{\\mathrm{Pass@}k}}
= \\frac{1}{M}\\sum_{m=1}^{M}
\\frac{\\binom{\\nu_{lm}}{k}}{\\binom{N}{k}}
```

# References
Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
*arXiv:2406.12045*. https://arxiv.org/abs/2406.12045
"""
function _rank_pass_hat_k(
    R,
    k;
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = pass_hat_k(@view(Rv[model, :, :]), k)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

pass_hat_k(R::AbstractArray{<:Integer,3}, k; method="competition", return_scores=false) =
    _rank_pass_hat_k(R, k; method=method, return_scores=return_scores)

"""
    g_pass_at_k_tau(
        R::AbstractArray{<:Integer, 3},
        k,
        tau;
        method="competition",
        return_scores=false,
)

Rank models by generalized G-Pass@k_τ per model.

Let ``X_{lm} ~ Hypergeometric(N, nu_{lm}, k)`` where
``nu_{lm} = \\sum_{n=1}^{N} R_{lmn}``. The score is:

```math
s_l^{\\mathrm{G\\text{-}Pass@}k_{\\tau}}
= \\frac{1}{M}\\sum_{m=1}^{M}
\\Pr\\!\\left(X_{lm}\\ge \\lceil \\tau k \\rceil\\right)
```

```math
\\Pr(X_{lm}\\ge \\lceil \\tau k \\rceil)
= \\sum_{j=\\lceil \\tau k \\rceil}^{k}
\\frac{\\binom{\\nu_{lm}}{j}\\binom{N-\\nu_{lm}}{k-j}}{\\binom{N}{k}}
```

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv:2412.13147*. https://arxiv.org/abs/2412.13147
"""
function _rank_g_pass_at_k_tau(
    R,
    k,
    tau;
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = g_pass_at_k_tau(@view(Rv[model, :, :]), k, tau)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

function g_pass_at_k_tau(
    R::AbstractArray{<:Integer,3},
    k,
    tau;
    method="competition",
    return_scores=false,
)
    return _rank_g_pass_at_k_tau(R, k, tau; method=method, return_scores=return_scores)
end

"""
    mg_pass_at_k(R::AbstractArray{<:Integer, 3}, k; method="competition", return_scores=false)

Rank models by per-model mG-Pass@k scores.

With ``X_{lm} ~ Hypergeometric(N, nu_{lm}, k)`` and ``m_0 = \\lceil k/2 \\rceil``:

```math
s_l^{\\mathrm{mG\\text{-}Pass@}k}
= \\frac{1}{M}\\sum_{m=1}^{M}
\\frac{2}{k}\\,\\mathbb{E}\\!\\left[(X_{lm}-m_0)_+\\right]
```

Equivalent discrete form:

```math
\\frac{2}{k}\\sum_{i=m_0+1}^{k}\\Pr(X_{lm}\\ge i)
```

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv:2412.13147*. https://arxiv.org/abs/2412.13147
"""
function _rank_mg_pass_at_k(
    R,
    k;
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = mg_pass_at_k(@view(Rv[model, :, :]), k)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

mg_pass_at_k(R::AbstractArray{<:Integer,3}, k; method="competition", return_scores=false) =
    _rank_mg_pass_at_k(R, k; method=method, return_scores=return_scores)
