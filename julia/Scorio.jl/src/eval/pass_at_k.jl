"""Pass-family eval metrics and Bayesian uncertainty estimators."""

function _comb_float(n::Integer, k::Integer)::Float64
    if n < 0 || k < 0 || k > n
        return 0.0
    end
    return Float64(binomial(BigInt(n), BigInt(k)))
end

function _log_comb(n::Integer, k::Integer)::Float64
    if n < 0 || k < 0 || k > n
        return -Inf
    end
    kk = min(k, n - k)
    out = 0.0
    @inbounds for i in 1:kk
        out += log(Float64(n - kk + i)) - log(Float64(i))
    end
    return out
end

function _hypergeom_pmf(
    population::Integer,
    successes::Integer,
    draws::Integer,
    selected_successes::Integer,
)::Float64
    failures = population - successes
    if selected_successes < 0 ||
       selected_successes > successes ||
       draws - selected_successes < 0 ||
       draws - selected_successes > failures
        return 0.0
    end
    logp = _log_comb(successes, selected_successes) +
           _log_comb(failures, draws - selected_successes) -
           _log_comb(population, draws)
    return clamp(exp(logp), 0.0, 1.0)
end

function _pass_probability(population::Integer, successes::Integer, draws::Integer)::Float64
    failures = population - successes
    if draws > failures
        return 1.0
    end
    log_failure = _log_comb(failures, draws) - _log_comb(population, draws)
    return clamp(-expm1(log_failure), 0.0, 1.0)
end

"""
    pass_at_k(R, k) -> Float64

Unbiased Pass@k estimator for binary outcomes.

Computes the probability that at least one of `k` selected samples is correct,
averaged over all questions.

# References
Chen, M., Tworek, J., Jun, H., et al. (2021).
Evaluating Large Language Models Trained on Code.
*arXiv preprint arXiv:2107.03374*.
https://arxiv.org/abs/2107.03374

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  binary outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  After coercion, ``R \\in \\{0,1\\}^{M \\times N}`` with
  ``R_{\\alpha i}=1`` indicating success.
- `k::Integer`:
  number of selected samples, constrained by ``1 \\le k \\le N``.

# Returns
- `Float64`: average Pass@k across all ``M`` questions.

# Notation
For each question ``\\alpha``:

```math
\\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i}
```

where ``\\nu_\\alpha`` is the number of successful trials.
Let ``C(a,b)=\\binom{a}{b}``.

# Formula

```math
\\mathrm{Pass@k}_\\alpha = 1 - \\frac{C(N - \\nu_\\alpha, k)}{C(N, k)}
```

```math
\\mathrm{Pass@k} = \\frac{1}{M}\\sum_{\\alpha=1}^{M}\\mathrm{Pass@k}_\\alpha
```

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

s = pass_at_k(R, 2)
```
"""
function pass_at_k(R, k::Integer)::Float64
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    M, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    nu = vec(sum(Rm, dims=2))
    vals = [_pass_probability(N, Int(n), k) for n in nu]

    return Float64(sum(vals) / M)
end

"""
    pass_hat_k(R, k) -> Float64

Pass-hat@k (Pass^k): probability all `k` selected samples are correct.

Also known as G-Pass@k.

# References
Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
*arXiv preprint arXiv:2406.12045*.
https://arxiv.org/abs/2406.12045

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  binary outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  After coercion, ``R \\in \\{0,1\\}^{M \\times N}``.
- `k::Integer`:
  number of selected samples, constrained by ``1 \\le k \\le N``.

# Returns
- `Float64`: average Pass-hat@k (Pass^k) across all ``M`` questions.

# Notation
For each question ``\\alpha``:

```math
\\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i}
```

with ``C(a,b)=\\binom{a}{b}``.

# Formula

```math
\\widehat{\\mathrm{Pass@k}}_\\alpha = \\frac{C(\\nu_\\alpha, k)}{C(N, k)}
```

```math
\\widehat{\\mathrm{Pass@k}}
= \\frac{1}{M}\\sum_{\\alpha=1}^{M}\\widehat{\\mathrm{Pass@k}}_\\alpha
```

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

s = pass_hat_k(R, 2)
```
"""
function pass_hat_k(R, k::Integer)::Float64
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    M, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    nu = vec(sum(Rm, dims=2))
    vals = [_hypergeom_pmf(N, Int(n), k, k) for n in nu]

    return Float64(sum(vals) / M)
end

"""
    g_pass_at_k(R, k) -> Float64

Alias for `pass_hat_k`.

# References
Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
*arXiv preprint arXiv:2406.12045*.
https://arxiv.org/abs/2406.12045

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  same contract as [`pass_hat_k`](@ref): ``R \\in \\{0,1\\}^{M \\times N}``
  after coercion.
- `k::Integer`:
  same contract as [`pass_hat_k`](@ref): ``1 \\le k \\le N``.

# Returns
- `Float64`: G-Pass@k score.

# Notation
Let ``\\widehat{\\mathrm{Pass@k}}`` be the value computed by [`pass_hat_k`](@ref)
on the same `R` and `k`.

# Formula

```math
\\mathrm{G\\text{-}Pass@k} = \\widehat{\\mathrm{Pass@k}}
```

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

s = g_pass_at_k(R, 2)
```
"""
function g_pass_at_k(R, k::Integer)::Float64
    return pass_hat_k(R, k)
end

"""
    g_pass_at_k_tau(R, k, tau) -> Float64

Generalized Pass@k with threshold `tau`.

Computes the probability of at least `ceil(tau * k)` successes in `k` draws,
averaged across questions. Interpolates between Pass@k (`tau=0`) and
Pass-hat@k (`tau=1`).

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv preprint arXiv:2412.13147*.
https://arxiv.org/abs/2412.13147

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  binary outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  After coercion, ``R \\in \\{0,1\\}^{M \\times N}``.
- `k::Integer`:
  number of selected samples, constrained by ``1 \\le k \\le N``.
- `tau::Real`:
  threshold ``\\tau \\in [0,1]``.
  ``\\tau = 0`` recovers Pass@k, and ``\\tau = 1`` recovers Pass-hat@k.

# Returns
- `Float64`: average generalized pass score across questions.

# Notation
For each question ``\\alpha``:

```math
\\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i}
```

Let ``j_0 = \\lceil \\tau k \\rceil`` and ``C(a,b)=\\binom{a}{b}``.

# Formula

```math
\\mathrm{G\\text{-}Pass@k}_{\\tau,\\alpha}
= \\sum_{j=j_0}^{k}
\\frac{C(\\nu_\\alpha,j)\\,C(N-\\nu_\\alpha,k-j)}{C(N,k)}
```

```math
\\mathrm{G\\text{-}Pass@k}_{\\tau}
= \\frac{1}{M}\\sum_{\\alpha=1}^{M} \\mathrm{G\\text{-}Pass@k}_{\\tau,\\alpha}
```

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

s = g_pass_at_k_tau(R, 3, 0.67)
```
"""
function g_pass_at_k_tau(
    R,
    k::Integer,
    tau::Real,
)::Float64
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    M, N = size(Rm)
    tau_f = Float64(tau)

    if !(0.0 <= tau_f <= 1.0)
        error("tau must be in [0, 1]; got $tau")
    end
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    if tau_f <= 0.0
        return pass_at_k(Rm, k)
    end

    nu = vec(sum(Rm, dims=2))
    j0 = Int(ceil(tau_f * k))
    if j0 > k
        return 0.0
    end

    vals = zeros(Float64, M)
    for j in j0:k
        for (idx, n) in enumerate(nu)
            vals[idx] += _hypergeom_pmf(N, Int(n), k, j)
        end
    end

    vals .= clamp.(vals, 0.0, 1.0)

    return Float64(sum(vals) / M)
end

"""
    mg_pass_at_k(R, k) -> Float64

Mean generalized pass metric over `tau in [0.5, 1.0]`.

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv preprint arXiv:2412.13147*.
https://arxiv.org/abs/2412.13147

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  binary outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  After coercion, ``R \\in \\{0,1\\}^{M \\times N}``.
- `k::Integer`:
  number of selected samples, constrained by ``1 \\le k \\le N``.

# Returns
- `Float64`: average mG-Pass@k score.

# Notation
For each question ``\\alpha``:

```math
\\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i}
```

Let ``m = \\lceil k/2 \\rceil`` and ``X_\\alpha \\sim \\mathrm{Hypergeom}(N,\\nu_\\alpha,k)``.

# Formula

```math
\\mathrm{mG\\text{-}Pass@k}_\\alpha
= \\frac{2}{k} \\sum_{j=m+1}^{k} (j-m) \\cdot P(X_\\alpha = j)
```

```math
\\mathrm{mG\\text{-}Pass@k}
= \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\mathrm{mG\\text{-}Pass@k}_\\alpha
```

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

s = mg_pass_at_k(R, 3)
```
"""
function mg_pass_at_k(R, k::Integer)::Float64
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    M, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    nu = vec(sum(Rm, dims=2))
    majority = Int(ceil(0.5 * k))
    if majority >= k
        return 0.0
    end

    vals = zeros(Float64, M)
    for j in (majority + 1):k
        for (idx, n) in enumerate(nu)
            pmf = _hypergeom_pmf(N, Int(n), k, j)
            vals[idx] += (j - majority) * pmf
        end
    end

    vals .*= 2.0 / k
    vals .= clamp.(vals, 0.0, 1.0)
    return Float64(sum(vals) / M)
end

function _log_rising_factorial(x::Float64, n::Int)::Float64
    if n <= 0
        return 0.0
    end

    out = 0.0
    for i in 0:(n - 1)
        out += log(x + i)
    end
    return out
end

function _beta_ratio(alpha::Float64, beta::Float64, a::Int, b::Int)::Float64
    log_ratio = _log_rising_factorial(alpha, a) +
                _log_rising_factorial(beta, b) -
                _log_rising_factorial(alpha + beta, a + b)
    return Float64(exp(log_ratio))
end

function _binary_beta_posterior_params(
    R;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Vector{Float64}, Vector{Float64}}
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    _, N = size(Rm)
    c = Float64.(vec(sum(Rm, dims=2)))
    alpha = Float64(alpha0) .+ c
    beta = Float64(beta0) .+ (N .- c)
    return alpha, beta
end

function _pass_at_k_bayes(
    R,
    k::Integer;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64}
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    M, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    alpha, beta = _binary_beta_posterior_params(Rm; alpha0=alpha0, beta0=beta0)
    means = zeros(Float64, M)
    vars_ = zeros(Float64, M)

    for i in 1:M
        a_i = alpha[i]
        b_i = beta[i]
        e_qk = _beta_ratio(a_i, b_i, 0, k)
        e_q2k = _beta_ratio(a_i, b_i, 0, 2 * k)

        m = 1.0 - e_qk
        e2 = 1.0 - 2.0 * e_qk + e_q2k
        v = max(0.0, e2 - m * m)

        means[i] = m
        vars_[i] = v
    end

    mu = Float64(sum(means) / M)
    sigma = Float64(sqrt(sum(vars_)) / M)
    return mu, sigma
end

function _pass_hat_k_bayes(
    R,
    k::Integer;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64}
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    M, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    alpha, beta = _binary_beta_posterior_params(Rm; alpha0=alpha0, beta0=beta0)
    means = zeros(Float64, M)
    vars_ = zeros(Float64, M)

    for i in 1:M
        a_i = alpha[i]
        b_i = beta[i]
        e_pk = _beta_ratio(a_i, b_i, k, 0)
        e_p2k = _beta_ratio(a_i, b_i, 2 * k, 0)

        m = e_pk
        v = max(0.0, e_p2k - m * m)

        means[i] = m
        vars_[i] = v
    end

    mu = Float64(sum(means) / M)
    sigma = Float64(sqrt(sum(vars_)) / M)
    return mu, sigma
end

function _g_pass_at_k_tau_bayes(
    R,
    k::Integer,
    tau::Real;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64}
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    M, N = size(Rm)
    tau_f = Float64(tau)

    if !(0.0 <= tau_f <= 1.0)
        error("tau must be in [0, 1]; got $tau")
    end
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    if tau_f <= 0.0
        return _pass_at_k_bayes(Rm, k; alpha0=alpha0, beta0=beta0)
    end
    if tau_f >= 1.0
        return _pass_hat_k_bayes(Rm, k; alpha0=alpha0, beta0=beta0)
    end

    j0 = Int(ceil(tau_f * k))
    alpha, beta = _binary_beta_posterior_params(Rm; alpha0=alpha0, beta0=beta0)

    means = zeros(Float64, M)
    vars_ = zeros(Float64, M)

    js = collect(j0:k)
    coeff = [_comb_float(k, j) for j in js]

    for i in 1:M
        a_i = alpha[i]
        b_i = beta[i]

        m = 0.0
        for idx_j in eachindex(js)
            j = js[idx_j]
            c_j = coeff[idx_j]
            m += c_j * _beta_ratio(a_i, b_i, j, k - j)
        end

        e2 = 0.0
        for idx_j in eachindex(js)
            j = js[idx_j]
            c_j = coeff[idx_j]
            for idx_l in eachindex(js)
                l = js[idx_l]
                c_l = coeff[idx_l]
                e2 += c_j * c_l * _beta_ratio(a_i, b_i, j + l, 2 * k - (j + l))
            end
        end

        v = max(0.0, e2 - m * m)
        means[i] = m
        vars_[i] = v
    end

    mu = Float64(sum(means) / M)
    sigma = Float64(sqrt(sum(vars_)) / M)
    return mu, sigma
end

function _mg_pass_at_k_bayes(
    R,
    k::Integer;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64}
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)

    M, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    alpha, beta = _binary_beta_posterior_params(Rm; alpha0=alpha0, beta0=beta0)

    majority = Int(ceil(0.5 * k))
    if majority >= k
        return 0.0, 0.0
    end

    js = collect((majority + 1):k)
    coeff = [Float64((2.0 / k) * (j - majority) * _comb_float(k, j)) for j in js]

    means = zeros(Float64, M)
    vars_ = zeros(Float64, M)

    for i in 1:M
        a_i = alpha[i]
        b_i = beta[i]

        m = 0.0
        for idx_j in eachindex(js)
            j = js[idx_j]
            c_j = coeff[idx_j]
            m += c_j * _beta_ratio(a_i, b_i, j, k - j)
        end

        e2 = 0.0
        for idx_j in eachindex(js)
            j = js[idx_j]
            c_j = coeff[idx_j]
            for idx_l in eachindex(js)
                l = js[idx_l]
                c_l = coeff[idx_l]
                e2 += c_j * c_l * _beta_ratio(a_i, b_i, j + l, 2 * k - (j + l))
            end
        end

        v = max(0.0, e2 - m * m)
        means[i] = m
        vars_[i] = v
    end

    mu = Float64(sum(means) / M)
    sigma = Float64(sqrt(sum(vars_)) / M)
    return mu, sigma
end

"""
    pass_at_k_ci(R, k, confidence=0.95, bounds=(0,1), alpha0=1, beta0=1)
        -> (mu, sigma, lo, hi)

Bayesian Pass@k posterior summary and normal-approximation credible interval.

# References
Chen, M., Tworek, J., Jun, H., et al. (2021).
Evaluating Large Language Models Trained on Code.
*arXiv preprint arXiv:2107.03374*.
https://arxiv.org/abs/2107.03374

Beta-Binomial posterior moments for the `*_ci` estimators follow the Scorio
package implementation.

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  binary outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  After coercion, ``R \\in \\{0,1\\}^{M \\times N}``.
- `k::Integer`:
  number of selected samples, constrained by ``1 \\le k \\le N``.
- `confidence::Real`:
  credibility level ``\\gamma \\in (0,1)``.
- `bounds::Tuple{<:Real, <:Real}`:
  clipping interval ``(\\ell, u)`` applied to returned bounds.
- `alpha0::Real`, `beta0::Real`:
  Beta prior parameters with ``\\alpha_0 > 0`` and ``\\beta_0 > 0``.

# Returns
- `Tuple{Float64, Float64, Float64, Float64}`:
  ``(\\mu, \\sigma, \\mathrm{lo}, \\mathrm{hi})``.

# Notation
After coercion, ``R \\in \\{0,1\\}^{M \\times N}``.
For question ``\\alpha``, let:

```math
c_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i}
```

and latent success probability:

```math
p_\\alpha \\mid R \\sim \\mathrm{Beta}(\\alpha_0 + c_\\alpha,\\; \\beta_0 + N - c_\\alpha)
```

Define ``g(p) = 1 - (1-p)^k`` and ``\\gamma = \\texttt{confidence}``.

# Formula
Dataset-level moments are:

```math
\\mu = \\frac{1}{M}\\sum_{\\alpha=1}^{M} \\mathbb{E}[g(p_\\alpha)], \\quad
\\sigma = \\frac{1}{M}\\sqrt{\\sum_{\\alpha=1}^{M}\\mathrm{Var}[g(p_\\alpha)]}
```

Credible interval:

```math
(\\mathrm{lo}, \\mathrm{hi}) = \\mu \\pm z_{(1+\\gamma)/2}\\sigma
```

then clipped to `bounds`.

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

mu, sigma, lo, hi = pass_at_k_ci(R, 2, 0.95, (0.0, 1.0), 1.0, 1.0)
```
"""
function pass_at_k_ci(
    R,
    k::Integer,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    mu, sigma = _pass_at_k_bayes(R, k; alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end

"""
    pass_hat_k_ci(R, k, confidence=0.95, bounds=(0,1), alpha0=1, beta0=1)
        -> (mu, sigma, lo, hi)

Bayesian Pass-hat@k (Pass^k) posterior summary and credible interval.

# References
Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
*arXiv preprint arXiv:2406.12045*.
https://arxiv.org/abs/2406.12045

Beta-Binomial posterior moments for the `*_ci` estimators follow the Scorio
package implementation.

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  same contract as [`pass_at_k_ci`](@ref): ``R \\in \\{0,1\\}^{M \\times N}``
  after coercion.
- `k::Integer`:
  same contract as [`pass_at_k_ci`](@ref): ``1 \\le k \\le N``.
- `confidence::Real`:
  credibility level ``\\gamma \\in (0,1)``.
- `bounds::Tuple{<:Real, <:Real}`:
  clipping interval ``(\\ell, u)``.
- `alpha0::Real`, `beta0::Real`:
  Beta prior parameters.

# Returns
- `Tuple{Float64, Float64, Float64, Float64}`:
  ``(\\mu, \\sigma, \\mathrm{lo}, \\mathrm{hi})``.

# Notation
Use the same posterior model and symbols as [`pass_at_k_ci`](@ref), and define:

```math
g(p) = p^k
```

with ``\\gamma = \\texttt{confidence}``.

# Formula
Dataset-level moments are:

```math
\\mu = \\frac{1}{M}\\sum_{\\alpha=1}^{M} \\mathbb{E}[g(p_\\alpha)], \\quad
\\sigma = \\frac{1}{M}\\sqrt{\\sum_{\\alpha=1}^{M}\\mathrm{Var}[g(p_\\alpha)]}
```

Credible interval:

```math
(\\mathrm{lo}, \\mathrm{hi}) = \\mu \\pm z_{(1+\\gamma)/2}\\sigma
```

then clipped to `bounds`.

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

mu, sigma, lo, hi = pass_hat_k_ci(R, 2, 0.95, (0.0, 1.0), 1.0, 1.0)
```
"""
function pass_hat_k_ci(
    R,
    k::Integer,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    mu, sigma = _pass_hat_k_bayes(R, k; alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end

"""
    g_pass_at_k_ci(R, k, confidence=0.95, bounds=(0,1), alpha0=1, beta0=1)
        -> (mu, sigma, lo, hi)

Alias for `pass_hat_k_ci`.

# References
Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
*arXiv preprint arXiv:2406.12045*.
https://arxiv.org/abs/2406.12045

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  same contract as [`pass_hat_k_ci`](@ref): ``R \\in \\{0,1\\}^{M \\times N}``
  after coercion.
- `k::Integer`:
  same contract as [`pass_hat_k_ci`](@ref): ``1 \\le k \\le N``.
- `confidence::Real`:
  credibility level ``\\gamma \\in (0,1)``.
- `bounds::Tuple{<:Real, <:Real}`:
  clipping interval ``(\\ell, u)``.
- `alpha0::Real`, `beta0::Real`:
  Beta prior parameters.

# Returns
- `Tuple{Float64, Float64, Float64, Float64}`:
  ``(\\mu, \\sigma, \\mathrm{lo}, \\mathrm{hi})``.

# Notation
Let ``(\\mu, \\sigma, \\mathrm{lo}, \\mathrm{hi})`` be the result from
[`pass_hat_k_ci`](@ref) on the same inputs.

# Formula

```math
\\mathrm{G\\text{-}Pass@k}_{\\mathrm{ci}} = \\widehat{\\mathrm{Pass@k}}_{\\mathrm{ci}}
```

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

mu, sigma, lo, hi = g_pass_at_k_ci(R, 2)
```
"""
function g_pass_at_k_ci(
    R,
    k::Integer,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    return pass_hat_k_ci(R, k, confidence, bounds, alpha0, beta0)
end

"""
    g_pass_at_k_tau_ci(R, k, tau, confidence=0.95, bounds=(0,1), alpha0=1, beta0=1)
        -> (mu, sigma, lo, hi)

Bayesian generalized Pass@k with threshold ``\\tau`` and credible interval.

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv preprint arXiv:2412.13147*.
https://arxiv.org/abs/2412.13147

Beta-Binomial posterior moments for the `*_ci` estimators follow the Scorio
package implementation.

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  binary outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  After coercion, ``R \\in \\{0,1\\}^{M \\times N}``.
- `k::Integer`:
  number of selected samples, constrained by ``1 \\le k \\le N``.
- `tau::Real`:
  threshold ``\\tau \\in [0,1]``.
- `confidence::Real`:
  credibility level ``\\gamma \\in (0,1)``.
- `bounds::Tuple{<:Real, <:Real}`:
  clipping interval ``(\\ell, u)``.
- `alpha0::Real`, `beta0::Real`:
  Beta prior parameters.

# Returns
- `Tuple{Float64, Float64, Float64, Float64}`:
  ``(\\mu, \\sigma, \\mathrm{lo}, \\mathrm{hi})``.

# Notation
Use the same posterior model and symbols as [`pass_at_k_ci`](@ref).
Let:

```math
j_0 = \\lceil \\tau k \\rceil
```

and:

```math
g(p) = \\sum_{j=j_0}^{k}
\\binom{k}{j} p^j (1-p)^{k-j}
```

with ``\\gamma = \\texttt{confidence}``.

# Formula
Dataset-level moments are:

```math
\\mu = \\frac{1}{M}\\sum_{\\alpha=1}^{M} \\mathbb{E}[g(p_\\alpha)], \\quad
\\sigma = \\frac{1}{M}\\sqrt{\\sum_{\\alpha=1}^{M}\\mathrm{Var}[g(p_\\alpha)]}
```

Credible interval:

```math
(\\mathrm{lo}, \\mathrm{hi}) = \\mu \\pm z_{(1+\\gamma)/2}\\sigma
```

then clipped to `bounds`.

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

mu, sigma, lo, hi = g_pass_at_k_tau_ci(R, 3, 0.67)
```
"""
function g_pass_at_k_tau_ci(
    R,
    k::Integer,
    tau::Real,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    mu, sigma = _g_pass_at_k_tau_bayes(R, k, tau; alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end

"""
    mg_pass_at_k_ci(R, k, confidence=0.95, bounds=(0,1), alpha0=1, beta0=1)
        -> (mu, sigma, lo, hi)

Bayesian mG-Pass@k posterior summary and credible interval.

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv preprint arXiv:2412.13147*.
https://arxiv.org/abs/2412.13147

Beta-Binomial posterior moments for the `*_ci` estimators follow the Scorio
package implementation.

# Arguments
- `R::Union{AbstractVector, AbstractMatrix}`:
  binary outcomes. A 1D input with length `N` is reshaped to ``1 \\times N``.
  After coercion, ``R \\in \\{0,1\\}^{M \\times N}``.
- `k::Integer`:
  number of selected samples, constrained by ``1 \\le k \\le N``.
- `confidence::Real`:
  credibility level ``\\gamma \\in (0,1)``.
- `bounds::Tuple{<:Real, <:Real}`:
  clipping interval ``(\\ell, u)``.
- `alpha0::Real`, `beta0::Real`:
  Beta prior parameters.

# Returns
- `Tuple{Float64, Float64, Float64, Float64}`:
  ``(\\mu, \\sigma, \\mathrm{lo}, \\mathrm{hi})``.

# Notation
Use the same posterior model and symbols as [`pass_at_k_ci`](@ref).
Let ``m = \\lceil k/2 \\rceil`` and:

```math
g(p)=\\frac{2}{k}\\sum_{j=m+1}^{k}(j-m)\\binom{k}{j}p^j(1-p)^{k-j}
```

with ``\\gamma = \\texttt{confidence}``.

# Formula
Dataset-level moments are:

```math
\\mu = \\frac{1}{M}\\sum_{\\alpha=1}^{M} \\mathbb{E}[g(p_\\alpha)], \\quad
\\sigma = \\frac{1}{M}\\sqrt{\\sum_{\\alpha=1}^{M}\\mathrm{Var}[g(p_\\alpha)]}
```

Credible interval:

```math
(\\mathrm{lo}, \\mathrm{hi}) = \\mu \\pm z_{(1+\\gamma)/2}\\sigma
```

then clipped to `bounds`.

# Examples
```julia
R = [1 0 1 0;
     0 0 1 1]

mu, sigma, lo, hi = mg_pass_at_k_ci(R, 3)
```
"""
function mg_pass_at_k_ci(
    R,
    k::Integer,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    mu, sigma = _mg_pass_at_k_bayes(R, k; alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end
