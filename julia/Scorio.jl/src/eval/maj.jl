"""Majority@k metrics for binary outcomes."""

_majority_tau(k::Integer)::Float64 = Float64((fld(k, 2) + 1) / k)

"""Strict-majority success probability among `k` finite-bank draws."""
function maj_at_k(R, k::Integer)::Float64
    Rm = _as_2d_int_matrix(R)
    _, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end
    return g_pass_at_k_tau(Rm, k, _majority_tau(k))
end

"""Bayesian posterior `(mu, sigma, lo, hi)` for latent Majority@k."""
function maj_at_k_ci(
    R,
    k::Integer,
    confidence::Real,
    bounds,
    alpha0::Real,
    beta0::Real,
)::Tuple{Float64, Float64, Float64, Float64}
    Rm = _as_2d_int_matrix(R)
    _, N = size(Rm)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end
    return g_pass_at_k_tau_ci(
        Rm,
        k,
        _majority_tau(k),
        confidence,
        bounds,
        alpha0,
        beta0,
    )
end
