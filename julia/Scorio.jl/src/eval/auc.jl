"""AUC@K evaluation metrics for binary outcomes."""

function _validate_auc_k(N::Integer, k::Integer)::Nothing
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end
    return nothing
end

function _auc_at_k_coefficients(k::Integer)::Vector{Float64}
    if k < 1
        error("k must be >= 1; got $k")
    end
    if k == 1
        return [1.0]
    end
    coeff = fill(1.0 / (k - 1), k)
    coeff[1] = 0.5 / (k - 1)
    coeff[end] = 0.5 / (k - 1)
    return coeff
end

"""
    auc_at_k(R, k) -> Float64

Normalized trapezoidal area under the finite-bank Pass@1 through Pass@k
curve, averaged across questions.
"""
function auc_at_k(R, k::Integer)::Float64
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = size(Rm)
    _validate_auc_k(N, k)

    if k == 1
        return pass_at_k(Rm, 1)
    end

    nu = vec(sum(Rm, dims=2))
    coeff = _auc_at_k_coefficients(k)
    vals = zeros(Float64, M)
    @inbounds for j in 1:k
        c_j = coeff[j]
        for i in 1:M
            vals[i] += c_j * _pass_probability(N, Int(nu[i]), j)
        end
    end
    return Float64(sum(vals) / M)
end

function _auc_at_k_bayes(
    R,
    k::Integer;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64}
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = size(Rm)
    _validate_auc_k(N, k)

    alpha, beta = _binary_beta_posterior_params(Rm; alpha0=alpha0, beta0=beta0)
    coeff = _auc_at_k_coefficients(k)
    means = zeros(Float64, M)
    vars_ = zeros(Float64, M)

    @inbounds for i in 1:M
        a_i = alpha[i]
        b_i = beta[i]
        eq = [_beta_ratio(a_i, b_i, 0, j) for j in 1:k]
        weighted_eq = sum(coeff .* eq)
        m = 1.0 - weighted_eq
        e2 = 1.0 - 2.0 * weighted_eq
        for j in 1:k
            for l in 1:k
                e2 += coeff[j] * coeff[l] * _beta_ratio(a_i, b_i, 0, j + l)
            end
        end
        means[i] = m
        vars_[i] = max(0.0, e2 - m * m)
    end

    return Float64(sum(means) / M), Float64(sqrt(sum(vars_)) / M)
end

"""
    auc_at_k_ci(R, k, confidence=0.95, bounds=(0, 1), alpha0=1, beta0=1)

Bayesian posterior `(mu, sigma, lo, hi)` for latent AUC@k.
"""
function auc_at_k_ci(
    R,
    k::Integer,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    if k == 1
        return pass_at_k_ci(R, 1, confidence, bounds, alpha0, beta0)
    end
    mu, sigma = _auc_at_k_bayes(R, k; alpha0=alpha0, beta0=beta0)
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end
