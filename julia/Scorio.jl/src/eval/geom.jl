"""Geometric and threshold-spectrum evaluation metrics."""

function _weighted_geometric_mean(
    x::Real,
    y::Real,
    x_weight::Real,
    y_weight::Real,
)::Float64
    x_f = Float64(x)
    y_f = Float64(y)
    xw = Float64(x_weight)
    yw = Float64(y_weight)
    if xw == 0.0 && yw == 0.0
        error("at least one power must be non-zero")
    end
    if x_f == 0.0 && xw < 0.0
        if y_f == 0.0 && yw > 0.0
            return 0.0
        end
        error("x_power must be non-negative when x is zero; got x_power=$x_weight")
    end
    if y_f == 0.0 && yw < 0.0
        if x_f == 0.0 && xw > 0.0
            return 0.0
        end
        error("y_power must be non-negative when y is zero; got y_power=$y_weight")
    end
    return Float64(x_f^xw * y_f^yw)
end

function _validate_beta_prior(alpha0::Real, beta0::Real)::Nothing
    if alpha0 <= 0.0 || beta0 <= 0.0
        error(
            "alpha0 and beta0 must both be > 0 for a Beta prior; got $alpha0, $beta0",
        )
    end
    return nothing
end

function _validate_finite_bank_k(N::Integer, k::Integer)::Nothing
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end
    return nothing
end

function _validate_latent_k(k::Integer)::Nothing
    if k < 1
        error("k must be >= 1; got k=$k")
    end
    return nothing
end

function _resolve_lambda(lam::Real, lambda_=nothing)::Float64
    lam_f = Float64(lam)
    if !isnothing(lambda_)
        if lam_f != 0.5
            error("Specify at most one of 'lam' and 'lambda_'.")
        end
        lam_f = Float64(lambda_)
    end
    if !(0.0 <= lam_f <= 1.0)
        error("lam must be in [0, 1]; got $lam_f")
    end
    return lam_f
end

function _unanimous_spectrum_weights(k::Integer)::Vector{Float64}
    _validate_latent_k(k)
    weights = zeros(Float64, k)
    weights[end] = 1.0
    return weights
end

function _mg_spectrum_weights(k::Integer)::Vector{Float64}
    _validate_latent_k(k)
    weights = zeros(Float64, k)
    first_active = Int(ceil(k / 2.0)) + 1
    weights[first_active:end] .= 2.0 / k
    return weights
end

function _validate_spectrum_weights(weights, k::Integer)::Vector{Float64}
    w = Float64.(collect(weights))
    if ndims(w) != 1 || length(w) != k
        error("weights must be a length-$k 1D array; got shape $(size(w))")
    end
    if !all(isfinite, w)
        error("weights must be finite")
    end
    if any(w .< 0.0)
        error("weights must be non-negative")
    end
    weight_sum = Float64(sum(w))
    if weight_sum > 1.0 + 1e-12
        error("weights must satisfy sum(weights) <= 1; got sum=$weight_sum")
    end
    return w
end

_event_score_levels(weights::AbstractVector{<:Real}) = vcat(0.0, cumsum(weights))

"""
    threshold_spectrum_at_k(R, k, weights) -> Float64

Expected cumulative threshold credit under `k` draws without replacement.
`weights[j]` is the incremental credit earned at threshold `j`.
"""
function threshold_spectrum_at_k(R, k::Integer, weights)::Float64
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = size(Rm)
    _validate_finite_bank_k(N, k)
    w = _validate_spectrum_weights(weights, k)
    levels = _event_score_levels(w)
    nu = vec(sum(Rm, dims=2))
    vals = zeros(Float64, M)

    @inbounds for j in 1:k
        credit = Float64(levels[j + 1])
        if credit == 0.0
            continue
        end
        for i in 1:M
            vals[i] += credit * _hypergeom_pmf(N, Int(nu[i]), k, j)
        end
    end
    return Float64(sum(vals) / M)
end

"""Dataset-level geometric blend of Pass@k and unanimous@k."""
function geom_ds_at_k(
    R,
    k::Integer,
    pass_power::Real,
    unanimous_power::Real,
)::Float64
    pass_score = pass_at_k(R, k)
    unanimous_score = pass_hat_k(R, k)
    return _weighted_geometric_mean(
        pass_score,
        unanimous_score,
        pass_power,
        unanimous_power,
    )
end

"""Questionwise geometric blend of finite-bank Pass@k and unanimous@k."""
function geom_at_k(
    R,
    k::Integer,
    pass_power::Real,
    unanimous_power::Real,
)::Float64
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, N = size(Rm)
    _validate_finite_bank_k(N, k)
    nu = vec(sum(Rm, dims=2))
    vals = zeros(Float64, M)
    @inbounds for i in 1:M
        pass_value = _pass_probability(N, Int(nu[i]), k)
        unanimous_value = _hypergeom_pmf(N, Int(nu[i]), k, k)
        vals[i] = _weighted_geometric_mean(
            pass_value,
            unanimous_value,
            pass_power,
            unanimous_power,
        )
    end
    return Float64(sum(vals) / M)
end

"""Geometric blend of Pass@k and a configurable threshold spectrum."""
function geo_spectrum_at_k(
    R,
    k::Integer,
    lam::Real,
    weights,
    lambda_,
)::Float64
    lam_f = _resolve_lambda(lam, lambda_)
    pass_score = pass_at_k(R, k)
    if lam_f == 1.0
        return pass_score
    end
    w = isnothing(weights) ? _mg_spectrum_weights(k) : _validate_spectrum_weights(weights, k)
    spectrum_score = threshold_spectrum_at_k(R, k, w)
    return _weighted_geometric_mean(pass_score, spectrum_score, lam_f, 1.0 - lam_f)
end

function _pass_and_spectrum_row_posterior_moments(
    R,
    k::Integer,
    weights;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)
    _validate_latent_k(k)
    _validate_beta_prior(alpha0, beta0)
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, _ = size(Rm)
    w = _validate_spectrum_weights(weights, k)
    alpha, beta = _binary_beta_posterior_params(Rm; alpha0=alpha0, beta0=beta0)
    levels = _event_score_levels(w)
    coeff = zeros(Float64, k + 1)
    @inbounds for j in 1:k
        coeff[j + 1] = Float64(levels[j + 1] * _comb_float(k, j))
    end
    active_js = [j for j in 1:k if coeff[j + 1] != 0.0]

    mean_pass = zeros(Float64, M)
    var_pass = zeros(Float64, M)
    mean_spec = zeros(Float64, M)
    var_spec = zeros(Float64, M)
    cov_ps = zeros(Float64, M)

    @inbounds for i in 1:M
        a_i = alpha[i]
        b_i = beta[i]
        eqk = _beta_ratio(a_i, b_i, 0, k)
        eq2k = _beta_ratio(a_i, b_i, 0, 2 * k)
        m_pass = 1.0 - eqk
        v_pass = max(0.0, eq2k - eqk * eqk)
        m_spec = 0.0
        e2_spec = 0.0
        e_ps = 0.0

        for j in active_js
            c_j = coeff[j + 1]
            moment_j = _beta_ratio(a_i, b_i, j, k - j)
            m_spec += c_j * moment_j
            e_ps += c_j * (moment_j - _beta_ratio(a_i, b_i, j, 2 * k - j))
            for l in active_js
                c_l = coeff[l + 1]
                e2_spec += c_j * c_l *
                           _beta_ratio(a_i, b_i, j + l, 2 * k - (j + l))
            end
        end

        mean_pass[i] = m_pass
        var_pass[i] = v_pass
        mean_spec[i] = m_spec
        var_spec[i] = max(0.0, e2_spec - m_spec * m_spec)
        cov_ps[i] = e_ps - m_pass * m_spec
    end

    return mean_pass, var_pass, mean_spec, var_spec, cov_ps
end

function _pass_and_spectrum_posterior_moments(
    R,
    k::Integer,
    weights;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64, Float64, Float64, Float64}
    mean_pass, var_pass, mean_spec, var_spec, cov_ps =
        _pass_and_spectrum_row_posterior_moments(
            R,
            k,
            weights;
            alpha0=alpha0,
            beta0=beta0,
        )
    M = length(mean_pass)
    return Float64(sum(mean_pass) / M),
    Float64(sum(var_pass) / M^2),
    Float64(sum(mean_spec) / M),
    Float64(sum(var_spec) / M^2),
    Float64(sum(cov_ps) / M^2)
end

function _geo_spectrum_at_k_bayes(
    R,
    k::Integer,
    lam::Real,
    weights;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64}
    lam_f = _resolve_lambda(lam)
    mu_pass, var_pass, mu_spec, var_spec, cov_ps =
        _pass_and_spectrum_posterior_moments(
            R,
            k,
            weights;
            alpha0=alpha0,
            beta0=beta0,
        )
    if lam_f == 0.0
        return mu_spec, Float64(sqrt(max(0.0, var_spec)))
    end
    if lam_f == 1.0
        return mu_pass, Float64(sqrt(max(0.0, var_pass)))
    end

    mu = _weighted_geometric_mean(mu_pass, mu_spec, lam_f, 1.0 - lam_f)
    if mu == 0.0
        return 0.0, 0.0
    end
    grad_pass = lam_f * mu_pass^(lam_f - 1.0) * mu_spec^(1.0 - lam_f)
    grad_spec = (1.0 - lam_f) * mu_pass^lam_f * mu_spec^(-lam_f)
    sigma2 = grad_pass^2 * var_pass + grad_spec^2 * var_spec +
             2.0 * grad_pass * grad_spec * cov_ps
    return Float64(mu), Float64(sqrt(max(0.0, sigma2)))
end

function _geom_at_k_bayes(
    R,
    k::Integer,
    pass_power::Real=0.5,
    unanimous_power::Real=0.5;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64}
    mean_pass, var_pass, mean_unanimous, var_unanimous, cov_pu =
        _pass_and_spectrum_row_posterior_moments(
            R,
            k,
            _unanimous_spectrum_weights(k);
            alpha0=alpha0,
            beta0=beta0,
        )
    M = length(mean_pass)
    means = zeros(Float64, M)
    variances = zeros(Float64, M)

    @inbounds for i in 1:M
        mu_pass = mean_pass[i]
        mu_unanimous = mean_unanimous[i]
        mu = _weighted_geometric_mean(
            mu_pass,
            mu_unanimous,
            pass_power,
            unanimous_power,
        )
        means[i] = mu
        if mu == 0.0
            variances[i] = 0.0
            continue
        end
        grad_pass = pass_power == 0.0 ? 0.0 :
                    pass_power * mu_pass^(pass_power - 1.0) *
                    mu_unanimous^unanimous_power
        grad_unanimous = unanimous_power == 0.0 ? 0.0 :
                         unanimous_power * mu_pass^pass_power *
                         mu_unanimous^(unanimous_power - 1.0)
        variances[i] = max(
            0.0,
            grad_pass^2 * var_pass[i] +
            grad_unanimous^2 * var_unanimous[i] +
            2.0 * grad_pass * grad_unanimous * cov_pu[i],
        )
    end
    return Float64(sum(means) / M), Float64(sqrt(sum(variances)) / M)
end

function _geom_ds_at_k_bayes(
    R,
    k::Integer,
    pass_power::Real=0.5,
    unanimous_power::Real=0.5;
    alpha0::Real=1.0,
    beta0::Real=1.0,
)::Tuple{Float64, Float64}
    mu_pass, var_pass, mu_unanimous, var_unanimous, cov_pu =
        _pass_and_spectrum_posterior_moments(
            R,
            k,
            _unanimous_spectrum_weights(k);
            alpha0=alpha0,
            beta0=beta0,
        )
    mu = _weighted_geometric_mean(
        mu_pass,
        mu_unanimous,
        pass_power,
        unanimous_power,
    )
    if mu == 0.0
        return 0.0, 0.0
    end
    grad_pass = pass_power == 0.0 ? 0.0 :
                pass_power * mu_pass^(pass_power - 1.0) *
                mu_unanimous^unanimous_power
    grad_unanimous = unanimous_power == 0.0 ? 0.0 :
                     unanimous_power * mu_pass^pass_power *
                     mu_unanimous^(unanimous_power - 1.0)
    sigma2 = grad_pass^2 * var_pass + grad_unanimous^2 * var_unanimous +
             2.0 * grad_pass * grad_unanimous * cov_pu
    return Float64(mu), Float64(sqrt(max(0.0, sigma2)))
end

"""Bayesian posterior `(mu, sigma, lo, hi)` for a threshold spectrum."""
function threshold_spectrum_at_k_ci(
    R,
    k::Integer,
    weights,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    w = _validate_spectrum_weights(weights, k)
    _, _, mu_spec, var_spec, _ = _pass_and_spectrum_posterior_moments(
        R,
        k,
        w;
        alpha0=alpha0,
        beta0=beta0,
    )
    sigma = Float64(sqrt(max(0.0, var_spec)))
    lo, hi = normal_credible_interval(
        mu_spec,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu_spec), sigma, Float64(lo), Float64(hi)
end

"""Bayesian posterior `(mu, sigma, lo, hi)` for questionwise Geom@k."""
function geom_at_k_ci(
    R,
    k::Integer,
    pass_power::Real,
    unanimous_power::Real,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    mu, sigma = _geom_at_k_bayes(
        R,
        k,
        pass_power,
        unanimous_power;
        alpha0=alpha0,
        beta0=beta0,
    )
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end

"""Bayesian posterior `(mu, sigma, lo, hi)` for dataset-level Geom@k."""
function geom_ds_at_k_ci(
    R,
    k::Integer,
    pass_power::Real,
    unanimous_power::Real,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    mu, sigma = _geom_ds_at_k_bayes(
        R,
        k,
        pass_power,
        unanimous_power;
        alpha0=alpha0,
        beta0=beta0,
    )
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end

"""Bayesian posterior `(mu, sigma, lo, hi)` for GeoSpectrum@k."""
function geo_spectrum_at_k_ci(
    R,
    k::Integer,
    lam::Real,
    weights,
    lambda_,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    lam_f = _resolve_lambda(lam, lambda_)
    w = if lam_f != 1.0
        isnothing(weights) ? _mg_spectrum_weights(k) : _validate_spectrum_weights(weights, k)
    else
        _unanimous_spectrum_weights(k)
    end
    mu, sigma = _geo_spectrum_at_k_bayes(
        R,
        k,
        lam_f,
        w;
        alpha0=alpha0,
        beta0=beta0,
    )
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end

"""Canonical GeoSpectrum@k with equal Pass/mG-spectrum powers."""
function geo_spectrum_star_at_k(R, k::Integer)::Float64
    return geo_spectrum_at_k(R, k, 0.5, _mg_spectrum_weights(k))
end

"""Bayesian posterior for canonical GeoSpectrum@k."""
function geo_spectrum_star_at_k_ci(
    R,
    k::Integer,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    return geo_spectrum_at_k_ci(
        R,
        k,
        0.5,
        nothing,
        nothing,
        confidence,
        bounds,
        alpha0,
        beta0,
    )
end
