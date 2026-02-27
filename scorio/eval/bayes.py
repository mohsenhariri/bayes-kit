"""Bayes family evaluation metrics for categorical outcomes.

This module implements Bayes@N and credible interval.

Estimate a scalar score and uncertainty from repeated outcomes with optional
prior observations. The method supports binary and multi-category outcomes
through a category-weight vector.


Let :math:`R \\in \\{0,\\ldots,C\\}^{M \\times N}` be observed outcomes,
:math:`w \\in \\mathbb{R}^{C+1}` be category weights, and optional
:math:`R^0 \\in \\{0,\\ldots,C\\}^{M \\times D}` be prior outcomes.
For each question :math:`\\alpha` and class :math:`k`, Bayes@N forms counts
from :math:`R` and :math:`R^0`, adds Dirichlet plus one pseudo-counts, and
computes closed-form posterior moments ``mu`` and ``sigma``.

Available Functions
-------------------
- ``bayes`` returns ``(mu, sigma)``.
- ``bayes_ci`` returns ``(mu, sigma, lo, hi)`` using a normal-approximation
  credible interval around ``mu``.
"""

import numpy as np

from .utils import (
    _as_2d_int_matrix,
    _validate_matrix_range,
    normal_credible_interval,
)


def bayes(
    R: np.ndarray,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Performance evaluation using the Bayes@N framework.

    References:
        Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
        Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
        *ICLR 2026*, *arXiv:2510.04265*.
        https://arxiv.org/abs/2510.04265

    Args:
        R: :math:`M \\times N` int matrix with entries in :math:`\\{0,\\ldots,C\\}`.
           Row :math:`\\alpha` are the N outcomes for question :math:`\\alpha`.
        w: length :math:`(C+1)` weight vector :math:`(w_0,\\ldots,w_C)` that maps
           category k to score :math:`w_k`.
        R0: optional :math:`M \\times D` int matrix supplying D prior outcomes per row.
             If omitted, :math:`D=0`.

    Returns:
        tuple[float, float]: :math:`(\\mu, \\sigma)` performance metric estimate and its uncertainty.

    Notation:
        :math:`\\delta_{a,b}` is the Kronecker delta. For each row :math:`\\alpha` and class :math:`k \\in \\{0,\\ldots,C\\}`:

        .. math::

            n_{\\alpha k} &= \\sum_{i=1}^N \\delta_{k, R_{\\alpha i}} \\quad \\text{(counts in R)}

            n^0_{\\alpha k} &= 1 + \\sum_{i=1}^D \\delta_{k, R^0_{\\alpha i}} \\quad \\text{(Dirichlet(+1) prior)}

            \\nu_{\\alpha k} &= n_{\\alpha k} + n^0_{\\alpha k}

        Effective sample size: :math:`T = 1 + C + D + N` (scalar)

    Formula:
        .. math::

            \\mu = w_0 + \\frac{1}{M \\cdot T} \\sum_{\\alpha=1}^M \\sum_{j=0}^C \\nu_{\\alpha j} (w_j - w_0)

        .. math::

            \\sigma = \\sqrt{ \\frac{1}{M^2(T+1)} \\sum_{\\alpha=1}^M \\left[
                \\sum_j \\frac{\\nu_{\\alpha j}}{T} (w_j - w_0)^2
                - \\left( \\sum_j \\frac{\\nu_{\\alpha j}}{T} (w_j - w_0) \\right)^2 \\right] }

    Examples:
        >>> import numpy as np
        >>> R  = np.array([[0, 1, 2, 2, 1],
        ...                [1, 1, 0, 2, 2]])
        >>> w  = np.array([0.0, 0.5, 1.0])
        >>> R0 = np.array([[0, 2],
        ...                [1, 2]])

        With prior (D=2 → T=10):

        >>> mu, sigma = bayes(R, w, R0)
        >>> round(mu, 6), round(sigma, 6)
        (0.575, 0.084275)

        Without prior (D=0 → T=8):

        >>> mu2, sigma2 = bayes(R, w)
        >>> round(mu2, 6), round(sigma2, 6)
        (0.5625, 0.091998)

    """
    R = _as_2d_int_matrix(R)

    # Auto-detect binary matrix and set default w if not provided
    if w is None:
        unique_vals = np.unique(R)
        is_binary = len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1]))

        if is_binary:
            w = np.array([0.0, 1.0])
        else:
            unique_str = ", ".join(map(str, sorted(unique_vals)))
            raise ValueError(
                f"R contains more than 2 unique values ({unique_str}), so weight vector 'w' must be provided. "
                f"Please specify a weight vector of length {len(unique_vals)} to map each category to a score."
            )
    w = np.asarray(w, dtype=float)
    M, N = R.shape
    C = w.size - 1

    if R0 is None:
        D = 0
        R0m = np.zeros((M, 0), dtype=int)
    else:
        R0m = np.asarray(R0, dtype=int)
        if R0m.ndim == 1:
            R0m = R0m.reshape(M, -1)
        if R0m.shape[0] != M:
            raise ValueError("R0 must have the same number of rows (M) as R.")
        D = R0m.shape[1]

    # Validate value ranges
    _validate_matrix_range(R, 0, C, "R")
    _validate_matrix_range(R0m, 0, C, "R0")

    T = 1 + C + D + N

    def row_bincount(A: np.ndarray, length: int) -> np.ndarray:
        """Count occurrences of 0..length-1 in each row of A."""
        if A.shape[1] == 0:
            return np.zeros((A.shape[0], length), dtype=int)
        out = np.zeros((A.shape[0], length), dtype=int)
        rows = np.repeat(np.arange(A.shape[0]), A.shape[1])
        np.add.at(out, (rows, A.ravel()), 1)
        return out

    # n_{αk} and n^0_{αk}
    n_counts = row_bincount(R, C + 1)
    n0_counts = row_bincount(R0m, C + 1) + 1  # add 1 to every class (Dirichlet prior)

    # ν_{αk} = n_{αk} + n^0_{αk}
    nu = n_counts + n0_counts  # shape: (M, C+1)

    # μ = w0 + (1/(M T)) * Σ_α Σ_j ν_{αj} (w_j - w0)
    delta_w = w - w[0]
    mu = w[0] + (nu @ delta_w).sum() / (M * T)

    # σ = [ (1/(M^2 (T+1))) * Σ_α { Σ_j (ν_{αj}/T)(w_j-w0)^2
    #       - ( Σ_j (ν_{αj}/T)(w_j-w0) )^2 } ]^{1/2}
    nu_over_T = nu / T
    termA = (nu_over_T * (delta_w**2)).sum(axis=1)
    termB = (nu_over_T @ delta_w) ** 2
    sigma = np.sqrt(((termA - termB).sum()) / (M**2 * (T + 1)))

    return float(mu), float(sigma)


def bayes_ci(
    R: np.ndarray,
    w: np.ndarray | None = None,
    R0: np.ndarray | None = None,
    confidence: float = 0.95,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Convenience wrapper: Bayes@N mean and std plus a normal-approx credible interval (CrI)."""
    mu, sigma = bayes(R, w, R0)
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, two_sided=True, bounds=bounds
    )
    return float(mu), float(sigma), float(lo), float(hi)


__all__ = [
    "bayes",
    "bayes_ci",
]
