r"""Geometric pass/spectrum metrics for binary outcomes.

This module implements finite-bank geometric and threshold-spectrum metrics
together with approximate Beta-Bernoulli posterior summaries for latent
resampling quantities. The paper ``Geom@k: Stable Evaluation and Fast Rank
Recovery for LLM Reasoning`` defines a dataset-level endpoint blend;
``scorio`` also exposes a distinct questionwise Geom@k variant.

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

from ._count_score import CountScore
from ._inputs import (
    BinaryBank,
    prepare_binary_bank,
    validate_finite_k,
    validate_latent_k,
)
from ._posterior import (
    JointPosteriorMoments,
    joint_posterior_moments,
    posterior_moments,
)
from .utils import normal_credible_interval


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


def _geometric_delta_mean_variance(
    mean_x: float,
    variance_x: float,
    mean_y: float,
    variance_y: float,
    covariance: float,
    x_weight: float,
    y_weight: float,
) -> tuple[float, float]:
    """Apply first-order uncertainty propagation to a geometric blend."""
    mean = _weighted_geometric_mean(mean_x, mean_y, x_weight, y_weight)
    if mean == 0.0:
        return 0.0, 0.0

    gradient_x = 0.0
    if x_weight != 0.0:
        gradient_x = x_weight * (mean_x ** (x_weight - 1.0)) * (mean_y**y_weight)

    gradient_y = 0.0
    if y_weight != 0.0:
        gradient_y = y_weight * (mean_x**x_weight) * (mean_y ** (y_weight - 1.0))

    variance = (
        (gradient_x**2) * variance_x
        + (gradient_y**2) * variance_y
        + 2.0 * gradient_x * gradient_y * covariance
    )
    return mean, float(max(0.0, variance))


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
    k = validate_latent_k(k)
    weights = np.zeros(k, dtype=float)
    weights[-1] = 1.0
    return weights


def _mg_spectrum_weights(k: int) -> np.ndarray:
    r"""Return the upper-half weights used by ``GeoSpectrum*@k``.

    These weights are given by

    .. math::

        w^{mG}_{r,k} = \frac{2}{k} 1\{r \ge \lceil k/2 \rceil + 1\}.
    """
    k = validate_latent_k(k)
    weights = np.zeros(k, dtype=float)
    weights[(k + 1) // 2 :] = 2.0 / k
    return weights


def _spectrum_score(
    weights: np.ndarray | list[float] | tuple[float, ...],
    k: int,
) -> CountScore:
    w = np.asarray(weights)
    if w.ndim != 1 or w.shape[0] != k:
        raise ValueError(f"weights must be a length-{k} 1D array; got shape {w.shape}")
    return CountScore.spectrum(w)


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
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    return _spectrum_score(weights, k).mean(bank)


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
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    pass_score = CountScore.pass_at_k(k).mean(bank)
    unanimous_score = CountScore.unanimous_at_k(k).mean(bank)

    return _weighted_geometric_mean(
        pass_score, unanimous_score, pass_power, unanimous_power
    )


def geom_at_k(
    R: np.ndarray, k: int, pass_power: float = 0.5, unanimous_power: float = 0.5
) -> float:
    r"""
    Questionwise Geom@k averaged across questions.

    This is ``scorio``'s questionwise Geom@k variant. Unlike
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
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    pass_vals = CountScore.pass_at_k(k).state_scores(bank)
    unanimous_vals = CountScore.unanimous_at_k(k).state_scores(bank)

    vals = np.empty(bank.unique_successes.size, dtype=float)
    for i in range(vals.size):
        vals[i] = _weighted_geometric_mean(
            float(pass_vals[i]),
            float(unanimous_vals[i]),
            pass_power,
            unanimous_power,
        )

    return float(np.dot(bank.frequencies, vals) / bank.question_count)


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
    bank = prepare_binary_bank(R)
    k = validate_finite_k(bank.trial_count, k)
    if lam == 1.0:
        return CountScore.pass_at_k(k).mean(bank)

    weights = _mg_spectrum_weights(k) if weights is None else weights
    spectrum_score = _spectrum_score(weights, k).mean(bank)
    if lam == 0.0:
        return spectrum_score

    pass_score = CountScore.pass_at_k(k).mean(bank)
    return _weighted_geometric_mean(pass_score, spectrum_score, lam, 1.0 - lam)


def _pass_and_spectrum_joint_posterior_moments(
    R: np.ndarray,
    k: int,
    weights: np.ndarray,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[BinaryBank, JointPosteriorMoments]:
    """Prepare a bank and compute joint Pass/spectrum posterior moments."""
    k = validate_latent_k(k)
    spectrum_score = _spectrum_score(weights, k)
    bank = prepare_binary_bank(R)
    moments = joint_posterior_moments(
        bank,
        CountScore.pass_at_k(k),
        spectrum_score,
        alpha0=alpha0,
        beta0=beta0,
    )
    return bank, moments


def _geo_spectrum_at_k_bayes(
    R: np.ndarray,
    k: int,
    lam: float,
    weights: np.ndarray | list[float] | tuple[float, ...] | None,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float]:
    r"""Approximate posterior mean/std for latent :math:`\mathrm{GeoSpectrum}_{\lambda,w}@k`."""
    lam = _resolve_lambda(lam)
    k = validate_latent_k(k)
    bank = prepare_binary_bank(R)
    pass_score = CountScore.pass_at_k(k)

    if lam == 1.0:
        score_moments = posterior_moments(
            bank,
            pass_score,
            alpha0=alpha0,
            beta0=beta0,
        )
        return score_moments.mean, float(math.sqrt(score_moments.variance))

    weights = _mg_spectrum_weights(k) if weights is None else weights
    spectrum_score = _spectrum_score(weights, k)
    if lam == 0.0:
        score_moments = posterior_moments(
            bank,
            spectrum_score,
            alpha0=alpha0,
            beta0=beta0,
        )
        return score_moments.mean, float(math.sqrt(score_moments.variance))

    joint_moments = joint_posterior_moments(
        bank,
        pass_score,
        spectrum_score,
        alpha0=alpha0,
        beta0=beta0,
    )
    mu, variance = _geometric_delta_mean_variance(
        joint_moments.left.mean,
        joint_moments.left.variance,
        joint_moments.right.mean,
        joint_moments.right.variance,
        joint_moments.covariance,
        lam,
        1.0 - lam,
    )
    return mu, float(math.sqrt(variance))


def _geom_at_k_bayes(
    R: np.ndarray,
    k: int,
    pass_power: float = 0.5,
    unanimous_power: float = 0.5,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> tuple[float, float]:
    r"""Approximate posterior mean/std for latent questionwise Geom@k."""
    bank, moments = _pass_and_spectrum_joint_posterior_moments(
        R,
        k,
        _unanimous_spectrum_weights(k),
        alpha0=alpha0,
        beta0=beta0,
    )

    means = np.empty_like(moments.left.state_means)
    variances = np.empty_like(moments.left.state_means)
    for i in range(means.size):
        means[i], variances[i] = _geometric_delta_mean_variance(
            float(moments.left.state_means[i]),
            float(moments.left.state_variances[i]),
            float(moments.right.state_means[i]),
            float(moments.right.state_variances[i]),
            float(moments.state_covariances[i]),
            pass_power,
            unanimous_power,
        )

    frequencies = bank.frequencies.astype(float, copy=False)
    mu = float(np.dot(frequencies, means) / bank.question_count)
    variance = float(
        np.dot(frequencies, variances) / (bank.question_count * bank.question_count)
    )
    sigma = float(math.sqrt(variance))
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
    _, moments = _pass_and_spectrum_joint_posterior_moments(
        R,
        k,
        _unanimous_spectrum_weights(k),
        alpha0=alpha0,
        beta0=beta0,
    )

    mu, variance = _geometric_delta_mean_variance(
        moments.left.mean,
        moments.left.variance,
        moments.right.mean,
        moments.right.variance,
        moments.covariance,
        pass_power,
        unanimous_power,
    )
    return mu, float(math.sqrt(variance))


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
    k = validate_latent_k(k)
    spectrum_score = _spectrum_score(weights, k)
    bank = prepare_binary_bank(R)
    moments = posterior_moments(
        bank,
        spectrum_score,
        alpha0=alpha0,
        beta0=beta0,
    )
    mu_spec = moments.mean
    sigma = float(math.sqrt(moments.variance))
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
    w = None if lam == 1.0 else weights

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

    return geo_spectrum_at_k(R, k)


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
