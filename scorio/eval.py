from typing import Optional, Tuple

import numpy as np


def bayes(
    R: np.ndarray,
    w: np.ndarray,
    R0: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Performance evaluation using the Bayes@N framework.

    Args:
      R  : M×N int matrix with entries in {0,…,C}. Row α are the N outcomes for system α.
      w  : length-(C+1) weight vector (w0,…,wC) that maps category k to score wk.
      R0 : optional M×D int matrix supplying D prior outcomes per row. If omitted, D=0.

    Returns:
        (mu, sigma): performance metric estimate and its uncertainty.

    Notation
      δ_{a,b} is the Kronecker delta.  For each row α and class k∈{0,…,C}:
        n_{αk}  = Σ_{i=1..N} δ_{k, R_{αi}}                    (counts in R)
        n^0_{αk} = 1 + Σ_{i=1..D} δ_{k, R^0_{αi}}             (Dirichlet(+1) prior)
        ν_{αk}   = n_{αk} + n^0_{αk}

      T = 1 + C + D + N  (effective sample size; scalar)

    Estimates
      μ = w0 + (1/(M·T)) · Σ_{α=1..M} Σ_{j=0..C} ν_{αj} (w_j − w0)

      σ = sqrt{ (1/(M^2·(T+1))) · Σ_{α=1..M} [
                 Σ_{j} (ν_{αj}/T) (w_j − w0)^2
                 − ( Σ_{j} (ν_{αj}/T) (w_j − w0) )^2 ] }

    Examples
    --------
    >>> import numpy as np
    >>> R  = np.array([[0, 1, 2, 2, 1],
    ...                [1, 1, 0, 2, 2]])
    >>> w  = np.array([0.0, 0.5, 1.0])
    >>> R0 = np.array([[0, 2],
    ...                [1, 2]])

    # With prior (D=2 → T=10)
    >>> mu, sigma = bayes(R, w, R0)
    >>> round(mu, 6), round(sigma, 6)
    (0.575, 0.084275)

    # Without prior (D=0 → T=8)
    >>> mu2, sigma2 = bayes(R, w)
    >>> round(mu2, 6), round(sigma2, 6)
    (0.5625, 0.091998)

    """
    R = np.asarray(R, dtype=int)
    w = np.asarray(w, dtype=float)
    M, N = R.shape if R.ndim == 2 else (int(R.shape[0]) if R.size else 0, 0)
    if R.ndim == 1:
        # treat as single-row
        R = R.reshape(1, -1)
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
    if R.size and (R.min() < 0 or R.max() > C):
        raise ValueError("Entries of R must be integers in [0, C].")
    if R0m.size and (R0m.min() < 0 or R0m.max() > C):
        raise ValueError("Entries of R0 must be integers in [0, C].")

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


def avg(R: np.ndarray):
    return float(np.mean(R))


def pass_at(*args, **kwargs):
    raise NotImplementedError("Not yet implemented.")


__all__ = ["avg", "bayes", "pass_at"]
