"""Sequential inference helpers for adaptive evaluation workflows.

This module provides lightweight decision helpers that operate on posterior
means and posterior standard deviations.

"""

import math
from collections.abc import Sequence
from typing import Any, Literal, Optional

import numpy as np
from scipy.stats import kendalltau, norm, rankdata, spearmanr, weightedtau

from scorio.eval import avg


def ranking_confidence(
    mu_a: float, sigma_a: float, mu_b: float, sigma_b: float
) -> tuple[float, float]:
    """
    Compute pairwise ranking confidence under a normal approximation.

    For large ``M``, the score difference is approximated as
    ``Δ = π̄_a - π̄_b ~ Normal(mu_a - mu_b, sigma_a^2 + sigma_b^2)``.
    Define
    ``z = |mu_a - mu_b| / sqrt(sigma_a^2 + sigma_b^2)``,
    then ranking confidence is ``rho = Φ(z)``.

    Args:
        mu_a: Posterior mean score for candidate ``a``.
        sigma_a: Posterior standard deviation for candidate ``a``.
        mu_b: Posterior mean score for candidate ``b``.
        sigma_b: Posterior standard deviation for candidate ``b``.

    Returns:
        Tuple ``(rho, z)`` where ``rho`` is pairwise ordering confidence and
        ``z`` is the absolute standardized separation.

    Notes:
        If ``sigma_a == sigma_b == 0``, returns ``(0.5, inf)`` for a tie
        (``mu_a == mu_b``) and ``(1.0, inf)`` otherwise.
    """
    denom = math.sqrt(float(sigma_a) ** 2 + float(sigma_b) ** 2)
    if denom == 0.0:
        # Identical (or both exact) uncertainties: either perfect certainty or tie.
        if float(mu_a) == float(mu_b):
            return 0.5, float("inf")
        return 1.0, float("inf")
    z = abs(float(mu_a) - float(mu_b)) / denom
    rho = float(norm.cdf(z))
    return rho, float(z)


def ci_from_mu_sigma(
    mu: float,
    sigma: float,
    confidence: float = 0.95,
    clip: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Build a normal-approximation credible interval from ``mu`` and ``sigma``.

    For large ``M``, the posterior over an aggregate score is approximately
    Gaussian, so a central interval can be approximated as
    ``mu ± z * sigma`` at the requested confidence level.

    Args:
        mu: Posterior mean.
        sigma: Posterior standard deviation.
        confidence: Central credibility level in ``(0, 1)``.
        clip: Optional ``(lo, hi)`` bounds applied to the returned interval.

    Returns:
        Interval bounds ``(lo, hi)``.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    lo = mu - z * sigma
    hi = mu + z * sigma
    if clip is not None:
        lo = max(clip[0], lo)
        hi = min(clip[1], hi)
    return float(lo), float(hi)


def should_stop(
    sigma: float,
    confidence: float = 0.95,
    max_ci_width: float | None = None,
    max_half_width: float | None = None,
) -> bool:
    """
    Decide whether a scalar metric is precise enough to stop.

    Typical usage is sequential collection of trial outcomes. After each batch,
    compute ``sigma`` for the scalar metric and stop when interval width is
    below a user threshold.

    Args:
        sigma: Posterior standard deviation for the scalar metric.
        confidence: Credibility level used by the interval width rule.
        max_ci_width: Stop if full interval width is at most this value.
        max_half_width: Stop if interval half-width is at most this value.

    Returns:
        ``True`` if the requested width target is satisfied.

    Notes:
        Provide exactly one of ``max_ci_width`` and ``max_half_width``.

        Half-width is computed as ``z * sigma`` where
        ``z = Φ^{-1}(0.5 + confidence/2)``.

    Examples:
        >>> # CI half-width <= 0.02 (i.e., ±2 points at 95%)
        >>> should_stop(0.01, confidence=0.95, max_half_width=0.02)
        True
    """
    if (max_ci_width is None) == (max_half_width is None):
        raise ValueError("Provide exactly one of max_ci_width or max_half_width.")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    half = z * float(sigma)
    if max_half_width is not None:
        return half <= float(max_half_width)
    assert max_ci_width is not None
    return 2.0 * half <= float(max_ci_width)


def should_stop_top1(
    mus_in_id_order: Sequence[float],
    sigmas_in_id_order: Sequence[float],
    confidence: float = 0.95,
    method: Literal["ci_overlap", "zscore"] = "ci_overlap",
) -> dict[str, Any]:
    """
    Decide whether the current top model is resolved with high confidence.

    Two decision rules are supported:

    - ``ci_overlap``: the leader stops only if its lower interval bound is
      above every competitor upper bound at the chosen confidence.
    - ``zscore``: the leader stops only if pairwise ranking confidence against
      every competitor exceeds ``confidence``.

    Args:
        mus_in_id_order: Score means in model identifier order.
        sigmas_in_id_order: Score standard deviations in the same order.
        confidence: Required confidence threshold in ``(0, 1)``.
        method:
            Stop rule name, either ``"ci_overlap"`` or ``"zscore"``.

    Returns:
        Dictionary with keys:
        ``"stop"`` for stop decision, ``"leader"`` for leader index, and
        ``"ambiguous"`` for unresolved competitor indices.
    """
    mus = np.asarray(mus_in_id_order, dtype=float)
    sigmas = np.asarray(sigmas_in_id_order, dtype=float)
    if mus.shape != sigmas.shape or mus.ndim != 1:
        raise ValueError(
            "mus_in_id_order and sigmas_in_id_order must be 1D and same shape."
        )
    if len(mus) == 0:
        raise ValueError("Empty inputs.")

    leader = int(np.argmax(mus))
    ambiguous: list[int] = []

    if method == "ci_overlap":
        # Disjoint-CI criterion
        z = float(norm.ppf(0.5 + confidence / 2.0))
        lo_leader = mus[leader] - z * sigmas[leader]
        for j in range(len(mus)):
            if j == leader:
                continue
            hi_j = mus[j] + z * sigmas[j]
            if lo_leader <= hi_j:
                ambiguous.append(j)
        return {"stop": len(ambiguous) == 0, "leader": leader, "ambiguous": ambiguous}

    if method == "zscore":
        for j in range(len(mus)):
            if j == leader:
                continue
            rho, _ = ranking_confidence(mus[leader], sigmas[leader], mus[j], sigmas[j])
            if rho < confidence:
                ambiguous.append(j)
        return {"stop": len(ambiguous) == 0, "leader": leader, "ambiguous": ambiguous}

    raise ValueError("method must be 'ci_overlap' or 'zscore'.")


def suggest_next_allocation(
    mus_in_id_order: Sequence[float],
    sigmas_in_id_order: Sequence[float],
    confidence: float = 0.95,
    method: Literal["ci_overlap", "zscore"] = "ci_overlap",
) -> tuple[int, int]:
    """
    Suggest the next model pair to sample for faster top-rank resolution.

    The heuristic is:
    1. Identify the current leader by ``mu``.
    2. Identify the most ambiguous competitor under the selected rule.
    3. Allocate additional samples to that leader and competitor pair.

    Returns:
        ``(leader, competitor)`` indices.

    Notes:
        For ``ci_overlap``, ambiguity is measured by the separation margin
        ``leader lower bound minus competitor upper bound``.
        Smaller margins indicate harder-to-separate competitors.

        For ``zscore``, ambiguity is measured by pairwise z separation.
        Smaller z values indicate harder-to-separate competitors.
    """
    mus = np.asarray(mus_in_id_order, dtype=float)
    sigmas = np.asarray(sigmas_in_id_order, dtype=float)
    if mus.shape != sigmas.shape or mus.ndim != 1:
        raise ValueError(
            "mus_in_id_order and sigmas_in_id_order must be 1D and same shape."
        )
    if len(mus) < 2:
        raise ValueError("Need at least two methods to allocate.")

    leader = int(np.argmax(mus))

    candidates = [j for j in range(len(mus)) if j != leader]
    if method == "ci_overlap":
        z = float(norm.ppf(0.5 + confidence / 2.0))
        lo_leader = mus[leader] - z * sigmas[leader]

        # margin > 0 means already separated; smaller margin => more ambiguous
        def margin(j: int) -> float:
            return lo_leader - (mus[j] + z * sigmas[j])

        competitor = min(candidates, key=margin)
        return leader, int(competitor)

    if method == "zscore":

        def zsep(j: int) -> float:
            _, z = ranking_confidence(mus[leader], sigmas[leader], mus[j], sigmas[j])
            return z

        competitor = min(candidates, key=zsep)
        return leader, int(competitor)

    raise ValueError("method must be 'ci_overlap' or 'zscore'.")


__all__ = [
    "ranking_confidence",
    "ci_from_mu_sigma",
    "should_stop",
    "should_stop_top1",
    "suggest_next_allocation",
]
