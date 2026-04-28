r"""Geometric pass/spectrum metrics for binary outcomes.

This module implements finite-bank geometric and threshold-spectrum metrics
together with approximate Beta-Bernoulli posterior summaries for latent
resampling quantities. The paper ``Geom@k: Fast to Converge, Slow to Drift``
defines a dataset-level endpoint blend; ``scorio`` also exposes a
questionwise Geom@k variant as the primary ``geom_at_k`` metric.

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
    = P_k(R)^\lambda \, S_{w,k}(R)^{1-\lambda},

where :math:`P_k(R)` is dataset-level Pass@k. The endpoint conventions are
:math:`\lambda = 0 \to S_{w,k}` and :math:`\lambda = 1 \to P_k`. The named
operating points are:

- ``geom_ds_at_k``: dataset-level endpoint blend with
  :math:`\lambda = 1/2` and :math:`w_r = 1\{r = k\}`.
- ``geom_at_k``: questionwise endpoint blend, computed before averaging
  across questions.
- ``GeoSpectrum*@k``: :math:`\lambda = 1/2` with upper-half weights
  :math:`w_r = (2/k)\,1\{r \ge \lceil k/2 \rceil + 1\}`.

The ``*_ci`` functions implement the approximate posterior
credible intervals for the corresponding latent i.i.d. quantities under a
Beta-Bernoulli model.

Available API
-------------
- ``geom_at_k`` and ``geom_at_k_ci`` for the questionwise Pass/Unanimous
  geometric blend.
- ``geom_ds_at_k`` and ``geom_ds_at_k_ci`` for the dataset-level
  Pass/Unanimous blend.
- ``geo_spectrum_at_k`` and ``geo_spectrum_at_k_ci`` for
  :math:`\mathrm{GeoSpectrum}_{\lambda,w}@k`.
- ``geo_spectrum_star_at_k`` and ``geo_spectrum_star_at_k_ci`` for the default
  upper-half operating point.
- ``threshold_spectrum_at_k`` and ``threshold_spectrum_at_k_ci`` for
  :math:`S_{w,k}`.
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


def _weighted_geometric_mean(
    x: float, y: float, x_weight: float, y_weight: float
) -> float:
    if x_weight == 0.0 and y_weight == 0.0:
        raise ValueError("at least one power must be non-zero")

    if x == 0.0 and x_weight < 0.0:
        if y == 0.0 and y_weight > 0.0:
            return 0.0
        raise ValueError(
            f"x_power must be non-negative when x is zero; got x_power={x_weight}"
        )

    if y == 0.0 and y_weight < 0.0:
        if x == 0.0 and x_weight > 0.0:
            return 0.0
        raise ValueError(
            f"y_power must be non-negative when y is zero; got y_power={y_weight}"
        )

    return float((x**x_weight) * (y**y_weight))


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


def _unanimous_spectrum_weights(k: int) -> np.ndarray:
    r"""Return endpoint weights :math:`w_r = 1\{r = k\}`."""
    _validate_latent_k(k)
    weights = np.zeros(k, dtype=float)
    weights[-1] = 1.0
    return weights


def _mg_spectrum_weights(k: int) -> np.ndarray:
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
        Appendix C.4.
    """
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    _, N = Rm.shape
    _validate_finite_bank_k(N, k)
    w = _validate_spectrum_weights(weights, k)

    nu = np.sum(Rm, axis=1)
    levels = _event_score_levels(w)
    denom = float(comb(N, k))
    vals = np.zeros_like(nu, dtype=float)
    for j in range(1, k + 1):
        credit = float(levels[j])
        if credit == 0.0:
            continue
        vals += credit * comb(nu, j) * comb(N - nu, k - j) / denom
    return float(np.mean(vals))


def geom_ds_at_k(
    R: np.ndarray, k: int, pass_power: float = 0.5, unanimous_power: float = 0.5
) -> float:
    r"""
    Dataset-level Pass/Unanimous geometric blend.

    This is the endpoint GeoSpectrum operating point from the paper: it first
    averages Pass@k and Unanimous@k across questions, then applies the
    geometric blend. For the questionwise metric that blends before averaging,
    use :func:`geom_at_k`.

    The default operating point is the geometric mean of dataset-level
    Pass@k and Unanimous@k (equivalently Pass^k). The same API also exposes
    nearby operating points by letting callers adjust the exponents on the
    Pass@k and Unanimous@k terms directly.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
           :math:`R_{\alpha i} = 1` if trial :math:`i` for question
           :math:`\alpha` passed, 0 otherwise.
        k: Sampling budget with :math:`1 \le k \le N`.
        pass_power: Exponent applied to ``Pass@k``.
        unanimous_power: Exponent applied to ``Unanimous@k``.

    Returns:
        float: The dataset-level endpoint score.

    Formula:
        .. math::

            G_{\mathrm{ds},k}(R; a, b)
            = \mathrm{Pass}@k(R)^a\,
              \mathrm{Unanimous}@k(R)^b

        with the default dataset-level endpoint operating point given by
        :math:`a = b = 1/2`.

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(geom_ds_at_k(R, 2), 6)
        0.653835
    """
    pass_score = _pass_at_k(R, k)
    unanimous_score = _pass_hat_k(R, k)

    return _weighted_geometric_mean(
        pass_score, unanimous_score, pass_power, unanimous_power
    )


def geom_at_k(
    R: np.ndarray, k: int, pass_power: float = 0.5, unanimous_power: float = 0.5
) -> float:
    r"""
    Questionwise Geom@k averaged across questions.

    This is ``scorio``'s primary Geom@k metric. Unlike
    :func:`geom_ds_at_k`, which blends dataset-level Pass@k and dataset-level
    Unanimous@k, this function first computes the per-question quantities

    .. math::

        P_{\alpha,k} =
        1 - \frac{\binom{N - \nu_\alpha}{k}}{\binom{N}{k}}

    .. math::

        U_{\alpha,k} =
        \frac{\binom{\nu_\alpha}{k}}{\binom{N}{k}}

    forms the geometric blend

    .. math::

        G_{\alpha,k} = P_{\alpha,k}^{a}\,U_{\alpha,k}^{b},

    and only then averages across questions.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Sampling budget with :math:`1 \le k \le N`.
        pass_power: Exponent applied to per-question ``Pass@k``.
        unanimous_power: Exponent applied to per-question ``Unanimous@k``.

    Returns:
        float: The average questionwise ``Geom@k`` score.

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(geom_at_k(R, 2), 6)
        0.647106
    """
    Rm = _as_2d_int_matrix(R)
    _validate_binary(Rm)
    _, N = Rm.shape
    _validate_finite_bank_k(N, k)

    nu = np.sum(Rm, axis=1)
    denom = float(comb(N, k))
    pass_vals = 1.0 - comb(N - nu, k) / denom
    unanimous_vals = comb(nu, k) / denom

    vals = np.empty(Rm.shape[0], dtype=float)
    for i in range(Rm.shape[0]):
        vals[i] = _weighted_geometric_mean(
            float(pass_vals[i]),
            float(unanimous_vals[i]),
            pass_power,
            unanimous_power,
        )

    return float(np.mean(vals))


def geo_spectrum_at_k(
    R: np.ndarray,
    k: int,
    lam: float = 0.5,
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lambda_: float | None = None,
) -> float:
    r"""
    :math:`\mathrm{GeoSpectrum}_{\lambda,w}@k` on the observed finite bank.

    By default ``weights=None`` selects the upper-half ``mG`` weights,
    so the two-argument call ``geo_spectrum_at_k(R, k)`` remains the special
    case

    .. math::

        \mathrm{GeoSpectrum}^*@k(R)
        = \sqrt{\mathrm{Pass}@k(R)\,\mathrm{mG\text{-}Pass}@k(R)}.

    This function also accepts the keyword alias ``lambda_=...`` for callers
    that prefer naming the coupling parameter after the mathematical symbol.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Sampling budget with :math:`1 \le k \le N`.
        lam: The coupling parameter :math:`\lambda` in :math:`[0,1]`.
        weights: Spectrum weights :math:`w`. If omitted, uses the built-in
            upper-half mG weights. Custom weights must be length-:math:`k`,
            non-negative, finite, and satisfy :math:`\sum_r w_r \le 1`.

    Returns:
        float: :math:`\mathrm{GeoSpectrum}_{\lambda,w}@k(R)`.

    Formula:
        .. math::

            \mathrm{GeoSpectrum}_{\lambda,w}@k(R)
            = \mathrm{Pass}@k(R)^\lambda \, S_{w,k}(R)^{1-\lambda}

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(geo_spectrum_at_k(R, 3), 6)
        0.408248
        >>> round(geo_spectrum_at_k(R, 3, lam=1.0), 6)
        1.0
    """
    lam = _resolve_lambda(lam, lambda_)
    pass_score = _pass_at_k(R, k)
    if lam == 1.0:
        return pass_score

    w = (
        _mg_spectrum_weights(k)
        if weights is None
        else _validate_spectrum_weights(weights, k)
    )
    spectrum_score = threshold_spectrum_at_k(R, k, w)
    return _weighted_geometric_mean(pass_score, spectrum_score, lam, 1.0 - lam)


def _pass_and_spectrum_row_posterior_moments(
    R: np.ndarray,
    k: int,
    weights: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Per-question posterior moments for latent Pass@k and spectrum scores.

    Returns:
        ``(mean_pass, var_pass, mean_spectrum, var_spectrum, cov_pass_spectrum)``
        arrays, one entry per question.

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
    levels = _event_score_levels(w)
    coeff = np.zeros(k + 1, dtype=float)
    for j in range(1, k + 1):
        coeff[j] = float(levels[j] * comb(k, j))
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

    return mean_pass, var_pass, mean_spec, var_spec, cov_ps


def _pass_and_spectrum_posterior_moments(
    R: np.ndarray,
    k: int,
    weights: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float, float]:
    r"""Dataset-level posterior moments for latent Pass@k and spectrum scores."""
    mean_pass, var_pass, mean_spec, var_spec, cov_ps = (
        _pass_and_spectrum_row_posterior_moments(
            R,
            k,
            weights,
            alpha0=alpha0,
            beta0=beta0,
        )
    )
    M = mean_pass.size
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

    mu = _weighted_geometric_mean(mu_pass, mu_spec, lam, 1.0 - lam)
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


def _geom_at_k_bayes(
    R: np.ndarray,
    k: int,
    pass_power: float = 0.5,
    unanimous_power: float = 0.5,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float]:
    r"""Approximate posterior mean/std for latent questionwise Geom@k."""
    (
        mean_pass,
        var_pass,
        mean_unanimous,
        var_unanimous,
        cov_pu,
    ) = _pass_and_spectrum_row_posterior_moments(
        R,
        k,
        _unanimous_spectrum_weights(k),
        alpha0=alpha0,
        beta0=beta0,
    )

    means = np.empty_like(mean_pass, dtype=float)
    variances = np.empty_like(mean_pass, dtype=float)
    for i in range(mean_pass.size):
        mu_pass = float(mean_pass[i])
        mu_unanimous = float(mean_unanimous[i])
        mu = _weighted_geometric_mean(
            mu_pass,
            mu_unanimous,
            pass_power,
            unanimous_power,
        )
        means[i] = mu
        if mu == 0.0:
            variances[i] = 0.0
            continue

        grad_pass = 0.0
        if pass_power != 0.0:
            grad_pass = (
                pass_power
                * (mu_pass ** (pass_power - 1.0))
                * (mu_unanimous**unanimous_power)
            )

        grad_unanimous = 0.0
        if unanimous_power != 0.0:
            grad_unanimous = (
                unanimous_power
                * (mu_pass**pass_power)
                * (mu_unanimous ** (unanimous_power - 1.0))
            )

        variances[i] = max(
            0.0,
            (grad_pass**2) * float(var_pass[i])
            + (grad_unanimous**2) * float(var_unanimous[i])
            + 2.0 * grad_pass * grad_unanimous * float(cov_pu[i]),
        )

    mu = float(np.mean(means))
    sigma = float(math.sqrt(float(np.sum(variances))) / mean_pass.size)
    return mu, sigma


def _geom_ds_at_k_bayes(
    R: np.ndarray,
    k: int,
    pass_power: float = 0.5,
    unanimous_power: float = 0.5,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float]:
    r"""Approximate posterior mean/std for latent dataset-level Geom@k."""
    (
        mu_pass,
        var_pass,
        mu_unanimous,
        var_unanimous,
        cov_pu,
    ) = _pass_and_spectrum_posterior_moments(
        R,
        k,
        _unanimous_spectrum_weights(k),
        alpha0=alpha0,
        beta0=beta0,
    )

    mu = _weighted_geometric_mean(
        mu_pass,
        mu_unanimous,
        pass_power,
        unanimous_power,
    )
    if mu == 0.0:
        return 0.0, 0.0

    grad_pass = 0.0
    if pass_power != 0.0:
        grad_pass = (
            pass_power
            * (mu_pass ** (pass_power - 1.0))
            * (mu_unanimous**unanimous_power)
        )

    grad_unanimous = 0.0
    if unanimous_power != 0.0:
        grad_unanimous = (
            unanimous_power
            * (mu_pass**pass_power)
            * (mu_unanimous ** (unanimous_power - 1.0))
        )

    sigma2 = (
        (grad_pass**2) * var_pass
        + (grad_unanimous**2) * var_unanimous
        + 2.0 * grad_pass * grad_unanimous * cov_pu
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
    r"""
    Approximate posterior summary for the latent spectrum :math:`S_{w,k}(p)`.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Latent resampling budget. Once the posterior is defined, any integer
           :math:`k \ge 1` is allowed.
        weights: Non-negative length-:math:`k` weights with
            :math:`\sum_r w_r \le 1`.
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`

    Notes:
        Unlike :func:`threshold_spectrum_at_k`, the posterior target is defined
        for latent i.i.d. resampling and therefore does not require
        :math:`k \le N`.

    Formula:
        Let :math:`A_j = \sum_{r \le j} w_r`. The per-question latent target is

        .. math::

            g(p) = \sum_{j=1}^{k} A_j \binom{k}{j} p^j (1-p)^{k-j}.

        Dataset-level aggregation uses

        .. math::

            \mu = \frac{1}{M} \sum_{\alpha=1}^{M} \mathbb{E}[g(p_\alpha)]

        .. math::

            \sigma = \frac{1}{M} \sqrt{
                \sum_{\alpha=1}^{M} \mathrm{Var}[g(p_\alpha)]
            }.
    """
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
    pass_power: float = 0.5,
    unanimous_power: float = 0.5,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Approximate posterior summary for the questionwise Geom@k target.

    This is the uncertainty counterpart of :func:`geom_at_k`: it applies a
    first-order delta method to each question's latent Pass@k and
    Unanimous@k quantities, then averages the resulting question-level
    geometric blends.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Latent resampling budget. Once the posterior is defined, any integer
           :math:`k \ge 1` is allowed.
        pass_power: Exponent applied to each question's latent ``Pass@k``.
        unanimous_power: Exponent applied to each question's latent
            ``Unanimous@k``.
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`

    Formula:
        Let :math:`\mu_{P,\alpha}` and :math:`\mu_{U,\alpha}` denote the
        posterior means of question :math:`\alpha`'s latent Pass@k and
        Unanimous@k quantities. Then

        .. math::

            \mu \approx \frac{1}{M}\sum_\alpha
                \mu_{P,\alpha}^{a}\,\mu_{U,\alpha}^{b}

        and :math:`\sigma` is computed by per-question first-order delta
        propagation through :math:`g(x, y) = x^a y^b`.

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = geom_at_k_ci(R, 2)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.610666, 0.133107, 0.3498, 0.8716)
    """
    mu, sigma = _geom_at_k_bayes(
        R,
        k,
        pass_power=pass_power,
        unanimous_power=unanimous_power,
        alpha0=alpha0,
        beta0=beta0,
    )
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


def geom_ds_at_k_ci(
    R: np.ndarray,
    k: int,
    pass_power: float = 0.5,
    unanimous_power: float = 0.5,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Approximate posterior summary for the dataset-level Geom@k target.

    This is the uncertainty counterpart of :func:`geom_ds_at_k` and matches
    the dataset-level latent quantity introduced in the paper when
    ``pass_power = unanimous_power = 0.5``.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Latent resampling budget. Once the posterior is defined, any integer
           :math:`k \ge 1` is allowed.
        pass_power: Exponent applied to latent dataset-level ``Pass@k``.
        unanimous_power: Exponent applied to latent dataset-level
            ``Unanimous@k``.
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`

    Formula:
        Let :math:`\mu_P` and :math:`\mu_U` denote the posterior means of the
        latent dataset-level Pass@k and Unanimous@k quantities. Then

        .. math::

            \mu \approx \mu_P^a\,\mu_U^b

        and :math:`\sigma` is computed by first-order delta propagation through
        :math:`g(x, y) = x^a y^b`.

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> mu, sigma, lo, hi = geom_ds_at_k_ci(R, 2)
        >>> round(mu, 6), round(sigma, 6), round(lo, 4), round(hi, 4)
        (0.612112, 0.132755, 0.3519, 0.8723)
    """
    mu, sigma = _geom_ds_at_k_bayes(
        R,
        k,
        pass_power=pass_power,
        unanimous_power=unanimous_power,
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
    r"""
    Approximate posterior summary for latent
    :math:`\mathrm{GeoSpectrum}_{\lambda,w}@k`.

    As in :func:`geo_spectrum_at_k`, omitting ``weights`` selects the
    ``GeoSpectrum*@k`` operating point.

    This function also accepts the keyword alias ``lambda_=...`` for callers
    that prefer naming the coupling parameter after the mathematical symbol.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Latent resampling budget. Once the posterior is defined, any integer
           :math:`k \ge 1` is allowed.
        lam: The coupling parameter :math:`\lambda` in :math:`[0,1]`.
        weights: Spectrum weights :math:`w`. If omitted, uses the built-in
            upper-half mG weights unless :math:`\lambda = 1`, in which case
            the spectrum term is irrelevant. Custom weights must be
            length-:math:`k`, non-negative, finite, and satisfy
            :math:`\sum_r w_r \le 1`.
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`

    Formula:
        Let :math:`x` denote latent Pass@k and :math:`y` denote the latent
        spectrum :math:`S_{w,k}`. The posterior mean is approximated by

        .. math::

            \mu \approx x^\lambda y^{1-\lambda}

        evaluated at the posterior means of :math:`x` and :math:`y`, and
        :math:`\sigma` is obtained by first-order delta propagation through
        :math:`g(x, y) = x^\lambda y^{1-\lambda}`.
    """
    lam = _resolve_lambda(lam, lambda_)
    w = None
    if lam != 1.0:
        w = (
            _mg_spectrum_weights(k)
            if weights is None
            else _validate_spectrum_weights(weights, k)
        )
    else:
        # GeoSpectrum_{1,w}@k is exactly Pass@k, so ``w`` is irrelevant.
        w = _unanimous_spectrum_weights(k)

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
    r"""
    Explicit alias for the default ``GeoSpectrum*@k`` operating point.

    Equivalent to calling :func:`geo_spectrum_at_k` with the default
    upper-half ``mG`` spectrum weights.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Sampling budget with :math:`1 \le k \le N`.

    Returns:
        float: The ``GeoSpectrum*@k`` score.
    """

    return geo_spectrum_at_k(R, k, lam=0.5, weights=_mg_spectrum_weights(k))


def geo_spectrum_star_at_k_ci(
    R: np.ndarray,
    k: int,
    confidence: float = 0.95,
    bounds: tuple[float, float] = (0.0, 1.0),
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float, float, float]:
    r"""
    Approximate posterior summary for latent ``GeoSpectrum*@k``.

    Equivalent to :func:`geo_spectrum_at_k_ci` with the default upper-half
    ``mG`` spectrum weights.

    Args:
        R: :math:`M \times N` binary matrix with entries in :math:`\{0,1\}`.
        k: Latent resampling budget. Once the posterior is defined, any integer
           :math:`k \ge 1` is allowed.
        confidence: credibility level of the interval (default 0.95).
        bounds: ``(lo, hi)`` clipping bounds for the interval
                (default ``(0, 1)``).
        alpha0: Beta prior parameter :math:`\alpha_0` (default 1).
        beta0: Beta prior parameter :math:`\beta_0` (default 1).

    Returns:
        tuple[float, float, float, float]:
            :math:`(\mu,\; \sigma,\; \text{lo},\; \text{hi})`
    """
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
    "geom_at_k_ci",
    "geom_ds_at_k",
    "geom_ds_at_k_ci",
    "geo_spectrum_at_k",
    "geo_spectrum_at_k_ci",
    "geo_spectrum_star_at_k",
    "geo_spectrum_star_at_k_ci",
    "threshold_spectrum_at_k",
    "threshold_spectrum_at_k_ci",
]
