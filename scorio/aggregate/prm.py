r"""Process / outcome reward-model aggregation for test-time scaling.

A *process reward model* (PRM) scores each reasoning **step** of a trace, while
an *outcome reward model* (ORM) scores only the final answer. To use a PRM's
step scores with the pool-selection rules in :mod:`scorio.aggregate` (which take
one scalar per candidate), the per-step scores must first be reduced to a single
per-trace reward. This module implements the standard step-score aggregation
functions; the resulting reward is then the ``scores`` input to
:func:`~scorio.aggregate.best_of_n` (PRM Best-of-N) or
:func:`~scorio.aggregate.weighted_majority_vote` (PRM-weighted vote).

An ORM needs no aggregation -- its single score is used directly. The
Bayes-optimal *combination* of a PRM score with the generator's own likelihood
is a separate, calibrated step handled by
:func:`~scorio.aggregate.logit_weighted_vote`.

Methods
-------
- ``prm_aggregate``: reduce a trace's per-step reward-model scores to one
  trace-level reward via ``last`` / ``min`` / ``mean`` / ``prod`` / ``max``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["prm_aggregate"]

_METHODS = ("last", "min", "mean", "prod", "max")


def prm_aggregate(step_scores: Any, *, method: str = "last") -> float:
    r"""
    Aggregate a trace's per-step process-reward scores into one trace reward.

    A process reward model emits a score :math:`r_t \in (0, 1)` for each of the
    trace's :math:`L` reasoning steps; this reduces those scores to a single
    reward for reranking or weighted voting. The reduction choice reflects a
    different notion of trajectory quality:

    - ``"last"`` -- the final step's score (the solution-level reward; the common
      default when scaling PRM verification).
    - ``"min"`` -- the worst step (a trace is only as correct as its weakest
      step; Math-Shepherd, and one of the two aggregations of Lightman et al.).
    - ``"prod"`` -- the product of step scores (joint step correctness; the other
      Lightman et al. aggregation).
    - ``"mean"`` -- the average step score (overall trajectory quality).
    - ``"max"`` -- the best step (included for completeness).

    References:
        Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T.,
        Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's Verify
        Step by Step (PRM; product / min aggregation). *ICLR 2024*,
        *arXiv:2305.20050*. https://arxiv.org/abs/2305.20050

        Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., &
        Sui, Z. (2024). Math-Shepherd: Verify and Reinforce LLMs Step-by-step
        without Human Annotations (min / product). *ACL 2024*,
        *arXiv:2312.08935*. https://arxiv.org/abs/2312.08935

        Zhang, L., Gao, J., Ren, X., Cao, Z., et al. (2025). (Benchmarks the
        ``{min, prod, mean, last, max}`` PRM aggregations.) *arXiv:2508.01682*.
        https://arxiv.org/abs/2508.01682

    Args:
        step_scores: ``(L,)`` array-like of per-step reward-model scores for one
            trace (typically in ``(0, 1)``; higher is better). Must be non-empty
            and finite.
        method: Reduction to apply -- ``"last"`` (default), ``"min"``, ``"mean"``,
            ``"prod"``, or ``"max"``.

    Returns:
        The aggregated per-trace reward (a scalar). Higher is better; feed it as
        a ``scores`` entry to a selection rule.

    Formula:
        .. math::

            R_{\min} = \min_t r_t, \quad
            R_{\mathrm{prod}} = \prod_{t=1}^{L} r_t, \quad
            R_{\mathrm{mean}} = \tfrac{1}{L}\sum_{t=1}^{L} r_t, \quad
            R_{\mathrm{last}} = r_L, \quad
            R_{\max} = \max_t r_t.

    Examples:
        >>> prm_aggregate([0.9, 0.8, 0.95], method="last")
        0.95
        >>> prm_aggregate([0.9, 0.4, 0.95], method="min")
        0.4
        >>> round(prm_aggregate([0.9, 0.8, 0.95], method="mean"), 4)
        0.8833
        >>> import numpy as np
        >>> # Build the (M, N) scores matrix for a batch of PRM step-score traces:
        >>> traces = [[[0.9, 0.8], [0.7, 0.6]], [[0.5, 0.9], [0.95, 0.99]]]
        >>> scores = np.array([[prm_aggregate(t, method="min") for t in q]
        ...                    for q in traces])
        >>> scores.tolist()
        [[0.8, 0.6], [0.5, 0.95]]
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}; got {method!r}.")
    r = np.asarray(step_scores, dtype=float).reshape(-1)
    if r.size == 0:
        raise ValueError("step_scores must be non-empty (L >= 1).")
    if not np.all(np.isfinite(r)):
        raise ValueError("step_scores must all be finite.")
    if method == "last":
        return float(r[-1])
    if method == "min":
        return float(np.min(r))
    if method == "mean":
        return float(np.mean(r))
    if method == "max":
        return float(np.max(r))
    return float(np.prod(r))  # "prod"
