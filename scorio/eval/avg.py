r"""Average accuracy metric with Bayesian uncertainty calibration.

Let :math:`R \in \{0,\ldots,C\}^{M \times N}` be outcomes and
:math:`w \in \mathbb{R}^{C+1}` be optional category weights. The weighted
average maps each entry :math:`R_{\alpha i}` to :math:`w_{R_{\alpha i}}` and
averages across questions and trials.

"""

import numpy as np

from ._categorical import (
    CategoricalBank,
    bayes_moments,
    observed_mean,
    prepare_categorical_bank,
)
from .utils import normal_credible_interval


def _avg_bank(R: np.ndarray, w: np.ndarray | None) -> CategoricalBank:
    """Prepare Avg inputs while preserving its public binary error text."""
    try:
        return prepare_categorical_bank(R, w=w)
    except ValueError as exc:
        if w is None and "weight vector 'w' must be provided" in str(exc):
            raise ValueError("Entries of R must be integers in [0, 1].") from exc
        raise


def avg(
    R: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[float, float]:
    r"""
    Avg@N plus a Bayesian uncertainty estimate (uniform prior, no R0).

    Under a uniform Dirichlet prior (:math:`D = 0`), the Bayesian posterior
    mean :math:`\mu` is an affine transform of the naive (weighted) average
    *a*, and the standard deviations are related by Eq. 20 of the Bayes@N
    paper:

    .. math::

        \sigma_{\text{avg}} = \frac{T}{N}\,\sigma_{\text{Bayes}}

    This lets you report the familiar **avg@N** while using the Bayesian
    framework of ``scorio`` to compute uncertainty on the same scale, without
    relying on CLT/Wald intervals or bootstrap resampling.

    Args:
        R: :math:`M \times N` int matrix with entries in
           :math:`\{0, \ldots, C\}`.
           Row :math:`\alpha` contains the *N* outcomes for question
           :math:`\alpha`.
        w: optional length :math:`(C+1)` weight vector
           :math:`(w_0, \ldots, w_C)`.
           If *None*, *R* must be binary and :math:`w = (0, 1)` is used.

    Returns:
        tuple[float, float]:
            :math:`(a,\; \sigma_a)` where *a* is the (weighted) average and
            :math:`\sigma_a` is the Bayesian uncertainty rescaled to the
            avg@N scale.

    Formula:
        Let :math:`T = 1 + C + N` (uniform prior, :math:`D = 0`).

        .. math::

            a &= \text{avg}(R, w)

            \sigma_a &= \frac{T}{N}\,\sigma_{\text{Bayes}}(R, w)

    Examples:
        Binary (no weights):

        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> a, sigma = avg(R)
        >>> round(a, 6), round(sigma, 6)
        (0.7, 0.165831)

        Weighted:

        >>> R = np.array([[0, 1, 2, 2, 1],
        ...               [1, 1, 0, 2, 2]])
        >>> w = np.array([0.0, 0.5, 1.0])
        >>> a, sigma = avg(R, w)
        >>> round(a, 6), round(sigma, 6)
        (0.6, 0.147196)
    """
    bank = _avg_bank(R, w)
    _, sigma_bayes = bayes_moments(bank)
    total = bank.category_count + bank.trial_count
    sigma_avg = (total / bank.trial_count) * sigma_bayes
    return observed_mean(bank), float(sigma_avg)


def avg_ci(
    R: np.ndarray,
    w: np.ndarray | None = None,
    confidence: float = 0.95,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    r"""
    Avg@N with Bayesian :math:`\sigma` and a normal-approximation
    credible interval (CrI).

    Combines :func:`avg` with a symmetric
    normal credible interval clipped to optional ``bounds``.

    Args:
        R: :math:`M \times N` int matrix with entries in
           :math:`\{0, \ldots, C\}`.
           Row :math:`\alpha` contains the *N* outcomes for question
           :math:`\alpha`.
        w: optional length :math:`(C+1)` weight vector
           :math:`(w_0, \ldots, w_C)`.
           If *None*, *R* must be binary and :math:`w = (0, 1)` is used.
        confidence: credibility level of the interval (default 0.95).
        bounds: optional ``(lo, hi)`` clipping bounds for the interval.

    Returns:
        tuple[float, float, float, float]:
            :math:`(a,\; \sigma_a,\; \text{lo},\; \text{hi})`

    Formula:
        .. math::

            \text{lo},\; \text{hi}
              = a \pm z_{(1+\gamma)/2}\,\sigma_a

        where :math:`\gamma` is the requested ``confidence`` level and
        the interval is clipped to ``bounds`` when provided.

    Examples:
        Binary (no weights):

        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> a, sigma, lo, hi = avg_ci(R, bounds=(0.0, 1.0))
        >>> round(a, 4), round(sigma, 4), round(lo, 4), round(hi, 4)
        (0.7, 0.1658, 0.375, 1.0)

        Weighted:

        >>> R = np.array([[0, 1, 2, 2, 1],
        ...               [1, 1, 0, 2, 2]])
        >>> w = np.array([0.0, 0.5, 1.0])
        >>> a, sigma, lo, hi = avg_ci(R, w, confidence=0.95)
        >>> round(a, 4), round(sigma, 4), round(lo, 4), round(hi, 4)
        (0.6, 0.1472, 0.3115, 0.8885)
    """
    a, sigma = avg(R, w)
    lo, hi = normal_credible_interval(
        a, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(a), float(sigma), float(lo), float(hi)


__all__ = [
    "avg",
    "avg_ci",
]
