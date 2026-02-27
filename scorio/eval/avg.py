"""Average family metrics with Bayesian uncertainty calibration.


Let :math:`R \\in \\{0,\\ldots,C\\}^{M \\times N}` be outcomes and
:math:`w \\in \\mathbb{R}^{C+1}` be optional category weights. The weighted
average maps each entry :math:`R_{\\alpha i}` to :math:`w_{R_{\\alpha i}}` and
averages across questions and trials.

"""

import numpy as np

from .bayes import bayes
from .utils import (
    _as_2d_int_matrix,
    _validate_binary,
    _validate_matrix_range,
    normal_credible_interval,
)


def _avg(
    R: np.ndarray,
    w: np.ndarray | None = None,
) -> float:
    """
    Simple (optionally weighted) average of all entries in the result matrix.

    When **w** is omitted, *R* must be binary and the function returns the
    arithmetic mean of the entries.  When **w** is supplied, each entry
    :math:`R_{\\alpha i}` is mapped through the weight vector before averaging.

    Args:
        R: :math:`M \\times N` result matrix with entries in
           :math:`\\{0, \\ldots, C\\}`.
           Row :math:`\\alpha` contains the *N* outcomes for question
           :math:`\\alpha`.
        w: optional length :math:`(C+1)` weight vector
           :math:`(w_0, \\ldots, w_C)` that maps category *k* to score
           :math:`w_k`.  If *None*, *R* must be binary and
           :math:`w = (0, 1)` is used.

    Returns:
        float: The (weighted) arithmetic mean of the mapped entries.

    Notation:
        :math:`R_{\\alpha i}` is the outcome for question :math:`\\alpha`
        on trial :math:`i`.

    Formula:
        .. math::

            \\text{avg} = \\frac{1}{M \\cdot N}
                \\sum_{\\alpha=1}^{M} \\sum_{i=1}^{N} w_{R_{\\alpha i}}

        When :math:`w = (0, 1)` this reduces to the plain binary average.

    Examples:
        Binary (no weights):

        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(_avg(R), 6)
        0.7

        Weighted:

        >>> R = np.array([[0, 1, 2, 2, 1],
        ...               [1, 1, 0, 2, 2]])
        >>> w = np.array([0.0, 0.5, 1.0])
        >>> round(_avg(R, w), 6)
        0.6
    """
    Rm = _as_2d_int_matrix(R)
    if w is None:
        _validate_binary(Rm)
        return float(np.mean(Rm))
    wv = np.asarray(w, dtype=float)
    C = wv.size - 1
    _validate_matrix_range(Rm, 0, C, "R")
    return float(np.mean(wv[Rm]))


def avg(
    R: np.ndarray,
    w: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Avg\u0040N plus a Bayesian uncertainty estimate (uniform prior, no R0).

    Under a uniform Dirichlet prior (:math:`D = 0`), the Bayesian posterior
    mean :math:`\\mu` is an affine transform of the naive (weighted) average
    *a*, and the standard deviations are related by (Eq. 20 in the paper):

    .. math::

        \\sigma_{\\text{avg}} = \\frac{T}{N}\\,\\sigma_{\\text{Bayes}}

    This lets you report the familiar **avg\u0040N** while using the Bayesian
    framework of Scorio to compute uncertainty -- no CLT and Wald intervals or
    bootstrap required.

    Args:
        R: :math:`M \\times N` int matrix with entries in
           :math:`\\{0, \\ldots, C\\}`.
           Row :math:`\\alpha` contains the *N* outcomes for question
           :math:`\\alpha`.
        w: optional length :math:`(C+1)` weight vector
           :math:`(w_0, \\ldots, w_C)`.
           If *None*, *R* must be binary and :math:`w = (0, 1)` is used.

    Returns:
        tuple[float, float]:
            :math:`(a,\\; \\sigma_a)` where *a* is the (weighted) average and
            :math:`\\sigma_a` is the Bayesian uncertainty rescaled to the
            avg\u0040N scale.

    Formula:
        Let :math:`T = 1 + C + N` (uniform prior, :math:`D = 0`).

        .. math::

            a &= \\text{avg}(R, w)

            \\sigma_a &= \\frac{T}{N}\\,\\sigma_{\\text{Bayes}}(R, w)

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
    Rm = _as_2d_int_matrix(R)
    if w is None:
        _validate_binary(Rm)
        wv = np.array([0.0, 1.0], dtype=float)
    else:
        wv = np.asarray(w, dtype=float)
    _, N = Rm.shape
    C = wv.size - 1
    if N <= 0:
        raise ValueError("R must have at least one column (N>=1)")

    # Bayesian σ under uniform prior (D=0)
    _, sigma_bayes = bayes(Rm, wv, R0=None)
    T = 1 + C + N  # D=0
    sigma_avg = (T / N) * sigma_bayes
    return _avg(Rm, wv), float(sigma_avg)


def avg_ci(
    R: np.ndarray,
    w: np.ndarray | None = None,
    confidence: float = 0.95,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    """
    Avg\u0040N with Bayesian :math:`\\sigma` and a normal-approximation
    credible interval (CrI).

    Combines :func:`avg` with a symmetric
    normal credible interval clipped to optional ``bounds``.

    Args:
        R: :math:`M \\times N` int matrix with entries in
           :math:`\\{0, \\ldots, C\\}`.
           Row :math:`\\alpha` contains the *N* outcomes for question
           :math:`\\alpha`.
        w: optional length :math:`(C+1)` weight vector
           :math:`(w_0, \\ldots, w_C)`.
           If *None*, *R* must be binary and :math:`w = (0, 1)` is used.
        confidence: credibility level of the interval (default 0.95).
        bounds: optional ``(lo, hi)`` clipping bounds for the interval.

    Returns:
        tuple[float, float, float, float]:
            :math:`(a,\\; \\sigma_a,\\; \\text{lo},\\; \\text{hi})`

    Formula:
        .. math::

            \\text{lo},\\; \\text{hi}
              = a \\pm z_{(1+\\gamma)/2}\\,\\sigma_a

        where :math:`\\gamma` is the requested ``confidence`` level and
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
