import numpy as np
from scipy.special import ndtri


def _as_2d_int_matrix(R: np.ndarray) -> np.ndarray:
    Rm = np.asarray(R, dtype=int)
    if Rm.ndim == 1:
        # treat as single-row
        Rm = Rm.reshape(1, -1)
    elif Rm.ndim != 2:
        raise ValueError("R must be a 1D or 2D array.")
    return Rm


def _validate_matrix_range(R: np.ndarray, low: int, high: int, name: str) -> None:
    """Validate that integer matrix entries are within [low, high]."""
    if R.size == 0:
        return
    if R.min() < low or R.max() > high:
        raise ValueError(f"Entries of {name} must be integers in [{low}, {high}].")


def _validate_binary(R: np.ndarray, name: str = "R") -> None:
    """Validate that an integer matrix is binary (entries in {0,1})."""
    _validate_matrix_range(R, 0, 1, name)


def _z_value(confidence: float, two_sided: bool = True) -> float:
    """Return the standard-normal z value for a desired confidence level.

    Args:
        confidence: Confidence level in (0,1), e.g. 0.95.
        two_sided: If True, returns z for a two-sided central interval (e.g., 0.95 -> 1.96).
                   If False, returns z for a one-sided tail probability (e.g., 0.95 -> 1.645).

    Returns:
        z such that:
          - two_sided: P(|Z| <= z) = confidence
          - one_sided: P(Z <= z)   = confidence
        for Z ~ N(0,1).
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0,1); got {confidence}")
    if two_sided:
        # Central two-sided interval: Phi(z) - Phi(-z) = confidence => Phi(z) = 0.5 + confidence/2
        return float(ndtri(0.5 + 0.5 * confidence))
    return float(ndtri(confidence))


def normal_credible_interval(
    mu: float,
    sigma: float,
    credibility: float = 0.95,
    two_sided: bool = True,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Gaussian-approximate Bayesian credible interval (CrI).

    Interprets (mu, sigma) as **posterior** mean and **posterior** standard deviation.
    The returned interval is the central posterior mass interval under a normal approximation.

    Args:
        mu: Posterior mean.
        sigma: Posterior std (>= 0).
        credibility: Posterior mass in (0,1), e.g. 0.95.
        two_sided: If True, returns a central two-sided CrI. If False, returns a one-sided upper CrI.
        bounds: Optional (lo, hi) clipping bounds.

    Returns:
        (lo, hi)
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")
    z = _z_value(credibility, two_sided=two_sided)
    if two_sided:
        lo, hi = mu - z * sigma, mu + z * sigma
    else:
        lo, hi = float("-inf"), mu + z * sigma

    if bounds is not None:
        b_lo, b_hi = bounds
        if b_lo > b_hi:
            raise ValueError("bounds must satisfy bounds[0] <= bounds[1]")
        lo = max(lo, b_lo)
        hi = min(hi, b_hi)
    return float(lo), float(hi)


__all__ = [
    "normal_credible_interval",
]
