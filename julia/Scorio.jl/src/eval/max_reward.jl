"""Max-reward evaluation metrics for categorical outcomes."""

using SpecialFunctions: loggamma

function _prepare_categorical_input(R, w=nothing, R0=nothing)
    Rm = _as_2d_int_matrix(R)
    if isnothing(w)
        unique_vals = unique(Rm)
        is_binary = length(unique_vals) <= 2 && all(v -> v == 0 || v == 1, unique_vals)
        if !is_binary
            unique_str = join(sort(unique_vals), ", ")
            error(
                "R contains more than 2 unique values ($unique_str), so weight vector 'w' must be provided. " *
                "Please specify a weight vector of length $(length(unique_vals)) to map each category to a score.",
            )
        end
        wv = [0.0, 1.0]
    else
        wv = Float64.(collect(w))
    end

    M, _ = size(Rm)
    C = length(wv) - 1
    _validate_matrix_range(Rm, 0, C, "R")

    if isnothing(R0)
        R0m = zeros(Int, M, 0)
    else
        R0m = _as_eval_int_array(R0, "R0")
        if ndims(R0m) == 1
            try
                # Match NumPy's row-major `reshape(M, -1)` for flat priors.
                R0m = permutedims(reshape(R0m, :, M))
            catch
                error("R0 must have the same number of rows (M) as R.")
            end
        elseif ndims(R0m) != 2
            error("R0 must be a 1D or 2D array.")
        end
        if size(R0m, 1) != M
            error("R0 must have the same number of rows (M) as R.")
        end
        _validate_matrix_range(R0m, 0, C, "R0")
    end
    return Rm, wv, R0m
end

function _row_bincount_eval(A::AbstractMatrix{<:Integer}, width::Integer)::Matrix{Int}
    out = zeros(Int, size(A, 1), width)
    @inbounds for row in axes(A, 1)
        for col in axes(A, 2)
            out[row, A[row, col] + 1] += 1
        end
    end
    return out
end

function _grouped_posterior_params(R, w=nothing, R0=nothing)
    Rm, wv, R0m = _prepare_categorical_input(R, w, R0)
    C = length(wv) - 1
    levels = sort(unique(wv))
    n_counts = _row_bincount_eval(Rm, C + 1)
    n0_counts = _row_bincount_eval(R0m, C + 1) .+ 1
    alpha_cat = n_counts .+ n0_counts
    gamma = zeros(Float64, size(Rm, 1), length(levels))

    @inbounds for cat in 1:(C + 1)
        level_idx = findfirst(isequal(wv[cat]), levels)
        gamma[:, level_idx] .+= alpha_cat[:, cat]
    end
    return gamma, levels
end

function _eval_logsumexp(values::AbstractVector{<:Real})::Float64
    max_value = maximum(values)
    if max_value == -Inf
        return -Inf
    end
    return Float64(max_value + log(sum(exp(Float64(v) - max_value) for v in values)))
end

function _dirichlet_nested_cumulative_moment(
    total::Real,
    a::Real,
    b::Real,
    k::Integer,
)::Float64
    total_f = Float64(total)
    a_f = Float64(a)
    b_f = Float64(b)
    if b_f <= 0.0
        error("b must be > 0 for nested cumulative moments")
    end

    log_denom = loggamma(total_f + 2.0 * k) - loggamma(total_f)
    log_terms = zeros(Float64, k + 1)
    @inbounds for r in 0:k
        log_terms[r + 1] = loggamma(k + 1.0) - loggamma(r + 1.0) -
                           loggamma(k - r + 1.0) +
                           loggamma(a_f + k + r) - loggamma(a_f) +
                           loggamma(b_f + k - r) - loggamma(b_f) -
                           log_denom
    end
    return Float64(exp(_eval_logsumexp(log_terms)))
end

"""
    max_at_k(R, k, w=nothing) -> Float64

Expected best reward among `k` samples drawn without replacement from each
question's observed response bank.
"""
function max_at_k(R, k::Integer, w)::Float64
    Rm, wv, _ = _prepare_categorical_input(R, w, nothing)
    M, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end

    coeff = [exp(_log_comb(i - 1, k - 1) - _log_comb(N, k)) for i in k:N]
    vals = zeros(Float64, M)
    @inbounds for row in 1:M
        rewards = sort([wv[Rm[row, col] + 1] for col in 1:N])
        vals[row] = sum(coeff .* rewards[k:N])
    end
    return Float64(sum(vals) / M)
end

function _max_at_k_bayes(R, k::Integer, w=nothing, R0=nothing)
    gamma, levels = _grouped_posterior_params(R, w, R0)
    M, L = size(gamma)
    total = Float64(sum(gamma[1, :]))
    if k < 1
        error("k must be >= 1; got $k")
    end
    if L == 1
        return Float64(levels[1]), 0.0, levels
    end

    gaps = diff(levels)
    top = Float64(levels[end])
    means = zeros(Float64, M)
    vars_ = zeros(Float64, M)

    @inbounds for row in 1:M
        cum = cumsum(gamma[row, :])[1:(end - 1)]
        e_ak = zeros(Float64, L - 1)
        e_a2k = zeros(Float64, L - 1)
        for idx in 1:(L - 1)
            a = Float64(cum[idx])
            b = total - a
            e_ak[idx] = _beta_ratio(a, b, k, 0)
            e_a2k[idx] = _beta_ratio(a, b, 2 * k, 0)
        end

        m = top - sum(gaps .* e_ak)
        cross = zeros(Float64, L - 1, L - 1)
        for i in 1:(L - 1)
            cross[i, i] = e_a2k[i]
            for j in (i + 1):(L - 1)
                a = Float64(cum[i])
                b = Float64(cum[j] - cum[i])
                moment = _dirichlet_nested_cumulative_moment(total, a, b, k)
                cross[i, j] = moment
                cross[j, i] = moment
            end
        end

        e2 = top * top - 2.0 * top * sum(gaps .* e_ak)
        e2 += sum(gaps .* (cross * gaps))
        means[row] = m
        vars_[row] = max(0.0, e2 - m * m)
    end

    mu = Float64(sum(means) / M)
    sigma = Float64(sqrt(sum(vars_)) / M)
    return mu, sigma, levels
end

"""
    max_at_k_ci(R, k, w=nothing, R0=nothing, confidence=0.95, bounds=nothing)

Bayesian posterior `(mu, sigma, lo, hi)` for latent Max@k.
"""
function max_at_k_ci(
    R,
    k::Integer,
    w,
    R0,
    confidence::Real,
    bounds,
)::Tuple{Float64, Float64, Float64, Float64}
    if k == 1
        return bayes_ci(R, w, R0, confidence, bounds)
    end

    mu, sigma, levels = _max_at_k_bayes(R, k, w, R0)
    interval_bounds = isnothing(bounds) ? (Float64(minimum(levels)), Float64(maximum(levels))) : bounds
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=interval_bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end
