r""":math:`\mathrm{GeoSpectrum}_{\lambda,w}@k`, and approximate Bayesian credible intervals.

This module implements the finite-bank definitions from "Geom@k: Fast to Converge, Slow to Drift" in addition to
first-order Beta-Bernoulli posterior approximation for latent resampling quantities.

Notation
--------
For a binary matrix :math:`R \in \{0,1\}^{M \times N}`, fixed budget
:math:`k`, and threshold weights :math:`w = (w_1, \ldots, w_k)` with
non-negative entries and :math:`\sum_r w_r \le 1`, define the
threshold-spectrum summary

.. math::

    S_{w,k}(R) = \sum_{r=1}^k w_r T_{r,k}(R),

where :math:`T_{r,k}(R)` is the dataset-level probability that a uniformly
sampled subset of size :math:`k` without replacement contains at least
:math:`r` correct trials.

The GeoSpectrum family is then

.. math::

    \mathrm{GeoSpectrum}_{\lambda,w}@k(R)
    = \mathrm{Pass}@k(R)^\lambda \, S_{w,k}(R)^{1-\lambda},

with the endpoint conventions :math:`\lambda = 0 \to S_{w,k}` and
:math:`\lambda = 1 \to \mathrm{Pass}@k`. The named operating points are:

- ``Geom@k``: :math:`\lambda = 1/2` with endpoint weights
  :math:`w_r = 1\{r = k\}`.
- ``GeoSpectrum*@k``: :math:`\lambda = 1/2` with upper-half weights
  :math:`w_r = (2/k)\,1\{r \ge \lceil k/2 \rceil + 1\}`.

The ``*_ci`` functions implement the approximate posterior
credible intervals for the corresponding latent i.i.d. quantities under a
Beta-Bernoulli model.
"""

import math

import numpy as np
from scipy.special import comb

from .pass_at_k import (
    _beta_ratio,
    _binary_beta_posterior_params,
)
from .pass_at_k import (
    pass_at_k as _pass_at_k,
)
from .pass_at_k import (
    pass_hat_k as _pass_hat_k,
)
from .utils import _as_2d_int_matrix, _validate_binary, normal_credible_interval


def _geometric_mean(x: float, y: float) -> float:
    return float(math.sqrt(max(0.0, x * y)))


def _weighted_geometric_mean(
    x: float, y: float, x_weight: float, y_weight: float
) -> float:
    if x_weight < 0.0 or y_weight < 0.0:
        raise ValueError(f"weights must be non-negative; got {x_weight}, {y_weight}")
    if x_weight == 0.0 and y_weight == 0.0:
        raise ValueError("at least one weight must be positive")
    return float((max(0.0, x) ** x_weight) * (max(0.0, y) ** y_weight))


def _sqrt_pu_over_p_score(x: float, y: float) -> float:
    if x <= 0.0:
        return 0.0
    return float(_geometric_mean(x, y) / x)


def _pass_and_unanimous_scores(R: np.ndarray, k: int) -> tuple[float, float]:
    return _pass_at_k(R, k), _pass_hat_k(R, k)


def _validate_beta_prior(alpha0: float, beta0: float) -> None:
    if alpha0 <= 0.0 or beta0 <= 0.0:
        raise ValueError(
            f"alpha0 and beta0 must both be > 0 for a Beta prior; got {alpha0}, {beta0}"
        )


def _validate_finite_bank_k(N: int, k: int) -> None:
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")


def _validate_latent_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1; got k={k}")


def _resolve_lambda(lam: float, lambda_: float | None = None) -> float:
    if lambda_ is not None:
        if lam != 0.5:
            raise TypeError("Specify at most one of 'lam' and 'lambda_'.")
        lam = lambda_
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1]; got {lam}")
    return float(lam)


def unanimous_spectrum_weights(k: int) -> np.ndarray:
    r"""Return the endpoint weights :math:`w_r = 1\{r = k\}` used by ``Geom@k``."""
    _validate_latent_k(k)
    weights = np.zeros(k, dtype=float)
    weights[-1] = 1.0
    return weights


def mg_spectrum_weights(k: int) -> np.ndarray:
    r"""Return the upper-half weights used by ``GeoSpectrum*@k``.

    These weights are given by

    .. math::

        w^{mG}_{r,k} = \frac{2}{k} 1\{r \ge \lceil k/2 \rceil + 1\}.
    """
    _validate_latent_k(k)
    weights = np.zeros(k, dtype=float)
    weights[int(math.ceil(k / 2.0)) :] = 2.0 / k
    return weights


def _validate_spectrum_weights(
    weights: np.ndarray | list[float] | tuple[float, ...],
    k: int,
) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] != k:
        raise ValueError(f"weights must be a length-{k} 1D array; got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    weight_sum = float(np.sum(w))
    if weight_sum > 1.0 + 1e-12:
        raise ValueError(
            f"weights must satisfy sum(weights) <= 1; got sum={weight_sum}"
        )
    return w


def _event_score_levels(weights: np.ndarray) -> np.ndarray:
    r"""Return :math:`A_j = \sum_{r \le j} w_r` with :math:`A_0 = 0`.

    :math:`A_j` is the credit assigned to a sampled subset of size :math:`k`
    that contains exactly :math:`j` correct trials.
    """
    return np.concatenate(([0.0], np.cumsum(weights, dtype=float)))


def _spectrum_binomial_coefficients(weights: np.ndarray) -> np.ndarray:
    r"""Return coefficients :math:`c_j = A_j \binom{k}{j}` for :math:`j = 0, \ldots, k`."""
    k = int(weights.shape[0])
    levels = _event_score_levels(weights)
    coeff = np.zeros(k + 1, dtype=float)
    for j in range(1, k + 1):
        coeff[j] = float(levels[j] * comb(k, j))
    return coeff


def _threshold_spectrum_values_from_counts(
    nu: np.ndarray, N: int, k: int, weights: np.ndarray
) -> np.ndarray:
    """Per-row finite-bank threshold-spectrum values from success counts."""
    levels = _event_score_levels(weights)
    denom = float(comb(N, k))
    vals = np.zeros_like(nu, dtype=float)
    for j in range(1, k + 1):
        credit = float(levels[j])
        if credit == 0.0:
            continue
        vals += credit * comb(nu, j) * comb(N - nu, k - j) / denom
    return vals


def _combine_pass_and_spectrum(
    pass_score: float, spectrum_score: float, lam: float
) -> float:
    lam = _resolve_lambda(lam)
    x = max(0.0, float(pass_score))
    y = max(0.0, float(spectrum_score))
    if lam == 0.0:
        return y
    if lam == 1.0:
        return x
    if x == 0.0 or y == 0.0:
        return 0.0
    return float((x**lam) * (y ** (1.0 - lam)))


def threshold_spectrum_at_k(
    R: np.ndarray,
    k: int,
    weights: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    r"""Finite-bank threshold-spectrum summary :math:`S_{w,k}(R)`.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Sampling budget with :math:`1 \le k \le N`.
        weights: Non-negative length-:math:`k` weights with
            :math:`\sum_r w_r \le 1`.

    Returns:
        float: :math:`S_{w,k}(R)` averaged across questions.

    Notes:
        This summary is defined by

        .. math::

            S_{w,k}(R) = \sum_{r=1}^k w_r T_{r,k}(R).

        The implementation uses the equivalent event-score representation from
        Appendix C.3.
    """
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    _, N = Rm.shape
    _validate_finite_bank_k(N, k)
    w = _validate_spectrum_weights(weights, k)

    nu = np.sum(Rm, axis=1)
    vals = _threshold_spectrum_values_from_counts(nu, N, k, w)
    return float(np.mean(vals))


def geom_at_k(R: np.ndarray, k: int) -> float:
    r"""``Geom@k``: :math:`\sqrt{\mathrm{Pass}@k(R)\,\mathrm{Unanimous}@k(R)}`."""
    pass_score, unanimous_score = _pass_and_unanimous_scores(R, k)
    return _geometric_mean(pass_score, unanimous_score)


def geo_spectrum_at_k(
    R: np.ndarray,
    k: int,
    lam: float = 0.5,
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lambda_: float | None = None,
) -> float:
    r""":math:`\mathrm{GeoSpectrum}_{\lambda,w}@k` on the observed finite bank.

    By default ``weights=None`` selects the upper-half ``mG`` weights,
    so the two-argument call ``geo_spectrum_at_k(R, k)`` remains the special
    case

    .. math::

        \mathrm{GeoSpectrum}^*@k(R)
        = \sqrt{\mathrm{Pass}@k(R)\,\mathrm{mG\text{-}Pass}@k(R)}.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Sampling budget with :math:`1 \le k \le N`.
        lam: The coupling parameter :math:`\lambda` in :math:`[0,1]`.
        weights: Spectrum weights :math:`w`. If omitted, uses
            ``mg_spectrum_weights(k)``.
        lambda_: Optional alias for ``lam``.

    Returns:
        float: :math:`\mathrm{GeoSpectrum}_{\lambda,w}@k(R)`.
    """
    lam = _resolve_lambda(lam, lambda_)
    pass_score = _pass_at_k(R, k)
    if lam == 1.0:
        return pass_score

    w = (
        mg_spectrum_weights(k)
        if weights is None
        else _validate_spectrum_weights(weights, k)
    )
    spectrum_score = threshold_spectrum_at_k(R, k, w)
    return _combine_pass_and_spectrum(pass_score, spectrum_score, lam)


def geometric_pass_favoring_at_k(R: np.ndarray, k: int) -> float:
    r"""Pass-favoring geometric blend of :math:`\mathrm{Pass}@k` and :math:`\mathrm{Unanimous}@k`.

    .. math::

        \mathrm{PassFavoring}@k(R)
        = \mathrm{Pass}@k(R)^{3/4}\,\mathrm{Unanimous}@k(R)^{1/4}.
    """
    pass_score, unanimous_score = _pass_and_unanimous_scores(R, k)
    return _weighted_geometric_mean(pass_score, unanimous_score, 0.75, 0.25)


def geometric_unanimous_favoring_at_k(R: np.ndarray, k: int) -> float:
    r"""Unanimous-favoring geometric blend of :math:`\mathrm{Pass}@k` and :math:`\mathrm{Unanimous}@k`.

    .. math::

        \mathrm{UnanimousFavoring}@k(R)
        = \mathrm{Pass}@k(R)^{1/4}\,\mathrm{Unanimous}@k(R)^{3/4}.
    """
    pass_score, unanimous_score = _pass_and_unanimous_scores(R, k)
    return _weighted_geometric_mean(pass_score, unanimous_score, 0.25, 0.75)


def sqrt_pu_over_p_at_k(R: np.ndarray, k: int) -> float:
    r"""Dataset-level stability factor for :math:`\mathrm{Pass}@k` vs. :math:`\mathrm{Unanimous}@k`.

    .. math::

        \frac{\sqrt{\mathrm{Pass}@k(R)\,\mathrm{Unanimous}@k(R)}}
        {\mathrm{Pass}@k(R)}
        = \sqrt{\frac{\mathrm{Unanimous}@k(R)}{\mathrm{Pass}@k(R)}}.
    """
    pass_score, unanimous_score = _pass_and_unanimous_scores(R, k)
    return _sqrt_pu_over_p_score(pass_score, unanimous_score)


def _pass_and_spectrum_posterior_moments(
    R: np.ndarray,
    k: int,
    weights: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float, float]:
    r"""Posterior moments for latent :math:`\mathrm{Pass}@k` and latent spectrum :math:`S_{w,k}`.

    Returns:
        ``(mu_pass, var_pass, mu_spectrum, var_spectrum, cov_pass_spectrum)``
        for the dataset-level latent quantities under independent row
        posteriors.

    Notes:
        Unlike the observed finite-bank metrics, these latent quantities are
        defined for any integer :math:`k \ge 1`. The implementation therefore
        does *not* restrict :math:`k` by the observed trial count :math:`N`.
    """
    _validate_latent_k(k)
    _validate_beta_prior(alpha0, beta0)

    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    M, _ = Rm.shape
    w = _validate_spectrum_weights(weights, k)

    alpha, beta = _binary_beta_posterior_params(Rm, alpha0=alpha0, beta0=beta0)
    coeff = _spectrum_binomial_coefficients(w)
    active_js = [j for j in range(1, k + 1) if coeff[j] != 0.0]

    mean_pass = np.empty(M, dtype=float)
    var_pass = np.empty(M, dtype=float)
    mean_spec = np.empty(M, dtype=float)
    var_spec = np.empty(M, dtype=float)
    cov_ps = np.empty(M, dtype=float)

    for i in range(M):
        a_i = float(alpha[i])
        b_i = float(beta[i])

        eqk = _beta_ratio(a_i, b_i, 0, k)
        eq2k = _beta_ratio(a_i, b_i, 0, 2 * k)
        m_pass = 1.0 - eqk
        v_pass = max(0.0, eq2k - eqk * eqk)

        m_spec = 0.0
        e2_spec = 0.0
        e_ps = 0.0

        for j in active_js:
            c_j = float(coeff[j])
            moment_j = _beta_ratio(a_i, b_i, j, k - j)
            m_spec += c_j * moment_j
            e_ps += c_j * (moment_j - _beta_ratio(a_i, b_i, j, 2 * k - j))
            for l in active_js:
                c_l = float(coeff[l])
                e2_spec += c_j * c_l * _beta_ratio(a_i, b_i, j + l, 2 * k - (j + l))

        v_spec = max(0.0, e2_spec - m_spec * m_spec)
        cov = e_ps - m_pass * m_spec

        mean_pass[i] = m_pass
        var_pass[i] = v_pass
        mean_spec[i] = m_spec
        var_spec[i] = v_spec
        cov_ps[i] = cov

    mu_pass = float(np.mean(mean_pass))
    mu_spec = float(np.mean(mean_spec))
    var_pass_dataset = float(np.sum(var_pass) / (M**2))
    var_spec_dataset = float(np.sum(var_spec) / (M**2))
    cov_dataset = float(np.sum(cov_ps) / (M**2))
    return mu_pass, var_pass_dataset, mu_spec, var_spec_dataset, cov_dataset


def _geo_spectrum_at_k_bayes(
    R: np.ndarray,
    k: int,
    lam: float,
    weights: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float]:
    r"""Approximate posterior mean/std for latent :math:`\mathrm{GeoSpectrum}_{\lambda,w}@k`."""
    lam = _resolve_lambda(lam)
    mu_pass, var_pass, mu_spec, var_spec, cov_ps = _pass_and_spectrum_posterior_moments(
        R,
        k,
        weights,
        alpha0=alpha0,
        beta0=beta0,
    )

    if lam == 0.0:
        return mu_spec, float(math.sqrt(max(0.0, var_spec)))
    if lam == 1.0:
        return mu_pass, float(math.sqrt(max(0.0, var_pass)))

    mu = _combine_pass_and_spectrum(mu_pass, mu_spec, lam)
    if mu == 0.0:
        return 0.0, 0.0

    grad_pass = lam * (mu_pass ** (lam - 1.0)) * (mu_spec ** (1.0 - lam))
    grad_spec = (1.0 - lam) * (mu_pass**lam) * (mu_spec ** (-lam))
    sigma2 = (
        (grad_pass**2) * var_pass
        + (grad_spec**2) * var_spec
        + 2.0 * grad_pass * grad_spec * cov_ps
    )
    return float(mu), float(math.sqrt(max(0.0, sigma2)))


def threshold_spectrum_at_k_ci(
    R: np.ndarray,
    k: int,
    weights: np.ndarray | list[float] | tuple[float, ...],
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""Approximate posterior summary for the latent spectrum :math:`S_{w,k}(p)`."""
    w = _validate_spectrum_weights(weights, k)
    _, _, mu_spec, var_spec, _ = _pass_and_spectrum_posterior_moments(
        R,
        k,
        w,
        alpha0=alpha0,
        beta0=beta0,
    )
    sigma = float(math.sqrt(max(0.0, var_spec)))
    lo, hi = normal_credible_interval(
        mu_spec, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu_spec), sigma, float(lo), float(hi)


def geom_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""Approximate posterior summary for latent ``Geom@k``.

    This matches Section 3.3 and Appendix C.2: the posterior mean
    is approximated by :math:`\sqrt{\mu_P \mu_U}` and the posterior variance is
    obtained by first-order delta propagation through
    :math:`g(x, y) = \sqrt{x y}`.
    """
    mu, sigma = _geo_spectrum_at_k_bayes(
        R,
        k,
        0.5,
        unanimous_spectrum_weights(k),
        alpha0=alpha0,
        beta0=beta0,
    )
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


def geo_spectrum_at_k_ci(
    R: np.ndarray,
    k: int,
    lam: float = 0.5,
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lambda_: float | None = None,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""Approximate posterior summary for latent :math:`\mathrm{GeoSpectrum}_{\lambda,w}@k`.

    As in :func:`geo_spectrum_at_k`, omitting ``weights`` selects the
    ``GeoSpectrum*@k`` operating point.
    """
    lam = _resolve_lambda(lam, lambda_)
    w = None
    if lam != 1.0:
        w = (
            mg_spectrum_weights(k)
            if weights is None
            else _validate_spectrum_weights(weights, k)
        )
    else:
        # GeoSpectrum_{1,w}@k is exactly Pass@k, so ``w`` is irrelevant.
        w = unanimous_spectrum_weights(k)

    mu, sigma = _geo_spectrum_at_k_bayes(
        R,
        k,
        lam,
        w,
        alpha0=alpha0,
        beta0=beta0,
    )
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


def geo_spectrum_star_at_k(R: np.ndarray, k: int) -> float:
    """Explicit alias for the default ``GeoSpectrum*@k`` operating point."""
    return geo_spectrum_at_k(R, k)


def geo_spectrum_star_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    """Approximate posterior summary for latent ``GeoSpectrum*@k``."""
    return geo_spectrum_at_k_ci(
        R,
        k,
        confidence=confidence,
        bounds=bounds,
        alpha0=alpha0,
        beta0=beta0,
    )


__all__ = [
    "geom_at_k",
    "geo_spectrum_at_k",
    "geo_spectrum_star_at_k",
    "geom_at_k_ci",
    "geo_spectrum_at_k_ci",
    "geo_spectrum_star_at_k_ci",
    "threshold_spectrum_at_k",
    "threshold_spectrum_at_k_ci",
    "unanimous_spectrum_weights",
    "mg_spectrum_weights",
    "geometric_pass_favoring_at_k",
    "geometric_unanimous_favoring_at_k",
    "sqrt_pu_over_p_at_k",
]
