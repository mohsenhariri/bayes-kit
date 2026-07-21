r"""Online / early-termination rules for test-time scaling.

The offline rules in :mod:`scorio.aggregate` assume a *fixed* candidate budget
and choose which answer to return. The rules here instead decide **when to stop
spending compute** -- either stopping the outer sampling loop once the answer is
already decided, or terminating a single reasoning trace mid-generation once it
looks unpromising. They turn a fixed-budget aggregator into an adaptive one that
spends fewer samples/tokens on easy questions.

Two granularities:

- **Stop sampling** (across traces): :func:`adaptive_consistency_stop` and
  :func:`esc_stop` watch the stream of extracted answers and report when to stop
  drawing new traces.
- **Stop generating** (within a trace): :func:`deepconf_online_stop` reports the
  token at which a trace's running group confidence falls below a threshold set
  from warmup traces by :func:`deepconf_stop_threshold`.

Both stopping styles are complementary to :mod:`scorio.sinf`, which decides how
many samples to draw from ``(mu, sigma)`` estimates with anytime-valid
guarantees; the rules here operate directly on answer counts / token confidences.

Methods
-------
- ``adaptive_consistency_stop``: stop sampling when the top-two answer counts
  make the current majority statistically decided (Aggarwal et al., 2023).
- ``esc_stop``: stop sampling when a whole window of samples agrees
  (Early-Stopping Self-Consistency; Li et al., 2024).
- ``deepconf_stop_threshold`` / ``deepconf_online_stop``: set an offline-warmup
  confidence threshold, then find a trace's early-termination token (DeepConf
  online; Fu et al., 2025).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import betainc

from ._base import _is_valid
from .confidence import _group_confidences, token_confidence

__all__ = [
    "adaptive_consistency_stop",
    "esc_stop",
    "deepconf_stop_threshold",
    "deepconf_online_stop",
]


def _top_two_counts(answers: Any) -> tuple[int, int]:
    """Counts of the two most frequent valid answers (``0`` if fewer exist)."""
    tally: dict[Any, int] = {}
    for a in answers:
        if _is_valid(a):
            tally[a] = tally.get(a, 0) + 1
    counts = sorted(tally.values(), reverse=True)
    v1 = counts[0] if counts else 0
    v2 = counts[1] if len(counts) > 1 else 0
    return v1, v2


def adaptive_consistency_stop(
    answers: Any,
    *,
    threshold: float = 0.95,
    return_prob: bool = False,
) -> Any:
    r"""
    Adaptive-Consistency stopping: stop sampling once the majority is decided.

    Self-consistency draws a fixed number of samples; Adaptive-Consistency stops
    early, after each new sample, as soon as the currently leading answer is
    *statistically* unlikely to be overtaken. Placing a Dirichlet prior on the
    answer probabilities and reducing to the top two answers, it stops when the
    posterior probability that the leader's true probability exceeds the
    runner-up's clears ``threshold``. Easy questions (an early, dominant majority)
    stop in a few samples; hard ones keep sampling to the budget.

    References:
        Aggarwal, P., Madaan, A., Yang, Y., & Mausam. (2023). Let's Sample Step
        by Step: Adaptive-Consistency for Efficient Reasoning and Coding with
        LLMs. *EMNLP 2023*, *arXiv:2305.11860*. https://arxiv.org/abs/2305.11860

    Args:
        answers: The sequence of extracted answers sampled *so far* (call after
            each new sample). Unparsable entries (``None``, ``""``, ``NaN``) are
            ignored. Answers are compared by equality.
        threshold: Posterior-probability threshold :math:`C \in (0, 1)` to stop
            (default ``0.95``); higher samples more before stopping.
        return_prob: If ``True``, return ``(stop, prob)`` with the current
            posterior probability that the leader stays on top.

    Returns:
        ``True`` if sampling should stop now, else ``False`` (a ``(stop, prob)``
        tuple when ``return_prob``). Always ``False`` until at least one valid
        answer has been seen.

    Formula:
        With the two largest answer counts :math:`v_1 \ge v_2` and a uniform
        Dirichlet prior, the top-two posterior is
        :math:`p \sim \mathrm{Beta}(v_1 + 1, v_2 + 1)` and the rule stops when

        .. math::

            \Pr(p > \tfrac12 \mid v_1, v_2)
            = 1 - I_{1/2}(v_1 + 1,\, v_2 + 1) \ge C,

        with :math:`I` the regularized incomplete beta function.

    Examples:
        >>> adaptive_consistency_stop(["A"] * 8 + ["B"] * 2)   # 8 vs 2 -> decided
        True
        >>> adaptive_consistency_stop(["A", "A", "B", "B"])    # tie -> keep going
        False
        >>> stop, p = adaptive_consistency_stop(["A"] * 5, return_prob=True)
        >>> stop, round(p, 4)
        (True, 0.9844)
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}.")
    v1, v2 = _top_two_counts(answers)
    prob = 0.0 if v1 == 0 else float(1.0 - betainc(v1 + 1, v2 + 1, 0.5))
    stop = prob >= threshold
    return (stop, prob) if return_prob else stop


def esc_stop(window_answers: Any) -> bool:
    r"""
    Early-Stopping Self-Consistency: stop when a whole sampling window agrees.

    The simplest, hyperparameter-light early-stopping rule: draw samples in fixed
    windows and stop the moment one window is *unanimous* (zero answer entropy).
    A fully-agreeing window is strong evidence the question is easy and further
    sampling is wasted; if no window agrees, sampling continues to the budget.

    References:
        Li, Y., Yuan, P., Feng, S., Pan, B., Wang, X., Sun, B., Wang, H., & Li,
        K. (2024). Escape Sky-high Cost: Early-stopping Self-Consistency for
        Multi-step Reasoning. *ICLR 2024*, *arXiv:2401.10480*.
        https://arxiv.org/abs/2401.10480

    Args:
        window_answers: The extracted answers of the most recent sampling window.
            Answers are compared by equality; a window containing an unparsable
            entry (``None``, ``""``, ``NaN``) is not unanimous.

    Returns:
        ``True`` if every answer in the window is valid and identical (so
        sampling should stop), else ``False``. An empty window returns ``False``.

    Formula:
        For window answers :math:`W`, stop iff the answer entropy
        :math:`H(W) = 0`, i.e. all :math:`|W|` answers are equal.

    Examples:
        >>> esc_stop(["A", "A", "A"])
        True
        >>> esc_stop(["A", "B", "A"])
        False
        >>> esc_stop(["A", None, "A"])
        False
    """
    seq = list(window_answers)
    if not seq:
        return False
    first = seq[0]
    if not _is_valid(first):
        return False
    return all(_is_valid(a) and a == first for a in seq[1:])


def deepconf_stop_threshold(warmup_confidences: Any, *, keep: float = 0.1) -> float:
    r"""
    DeepConf online stopping threshold from warmup trace confidences.

    DeepConf-online first generates a small batch of *warmup* traces, scores each
    with a group-confidence measure (e.g. lowest-group confidence from
    :func:`~scorio.aggregate.deepconf_confidence`), and sets a stopping threshold
    ``s`` so that only the most-confident fraction ``keep`` of traces would
    survive. During the run, a trace whose running confidence drops below ``s``
    is terminated (:func:`deepconf_online_stop`).

    References:
        Fu, Y., Wang, X., Tian, Y., & Zhao, J. (2025). Deep Think with Confidence.
        *arXiv:2508.15260*. https://arxiv.org/abs/2508.15260

    Args:
        warmup_confidences: Array-like of per-trace confidence scores from the
            warmup traces (higher = more confident).
        keep: Fraction of most-confident traces to keep (default ``0.10``,
            "DeepConf-low"; ``0.90`` is the more permissive "DeepConf-high").
            The threshold is the ``1 - keep`` quantile of the warmup confidences.

    Returns:
        The stopping threshold ``s`` (a scalar).

    Formula:
        .. math::

            s = Q_{1 - \mathrm{keep}}\big(\{C(y) : y \in \text{warmup}\}\big).

    Examples:
        >>> deepconf_stop_threshold([1.0, 2.0, 3.0, 4.0, 5.0], keep=0.2)
        4.2
    """
    if not 0.0 < keep <= 1.0:
        raise ValueError(f"keep must be in (0, 1]; got {keep}.")
    c = np.asarray(warmup_confidences, dtype=float).reshape(-1)
    if c.size == 0:
        raise ValueError("need at least one warmup confidence.")
    if not np.all(np.isfinite(c)):
        raise ValueError("warmup_confidences must all be finite.")
    return float(np.quantile(c, 1.0 - keep))


def deepconf_online_stop(
    topk_logprobs: Any,
    threshold: float,
    *,
    window: int = 2048,
) -> int | None:
    r"""
    DeepConf online early termination: the token where a trace should be cut.

    Scans a trace's DeepConf group confidence (sliding-window mean of the
    per-token confidences) and reports the first token at which a completed
    window's confidence falls below ``threshold`` -- the point at which
    DeepConf-online would stop generating this trace and discard it. Applied to
    an already-generated trace it *emulates* the online decision offline (true
    mid-generation termination needs generation-time streaming control, which
    lives outside this library).

    References:
        Fu, Y., Wang, X., Tian, Y., & Zhao, J. (2025). Deep Think with Confidence.
        *arXiv:2508.15260*. https://arxiv.org/abs/2508.15260

    Args:
        topk_logprobs: ``(T, k)`` array-like (or ragged list of rows) of
            per-position top-:math:`k` log-probabilities for one trace.
        threshold: Stopping threshold ``s`` (from :func:`deepconf_stop_threshold`
            over warmup traces). A window confidence ``< s`` triggers termination.
        window: Sliding-window (group) size in tokens (paper default ``2048``);
            clamped to the trace length.

    Returns:
        The 0-based token index of the last token generated before termination
        (the end of the first below-threshold window), or ``None`` if no window
        drops below ``threshold`` (the trace runs to completion and is kept).

    Formula:
        With group confidences :math:`C_{G_s}` over windows ending at token
        :math:`s`, the termination token is

        .. math::

            \min\{ s : C_{G_s} < \text{threshold} \},

        or :math:`\varnothing` (``None``) if no such window exists.

    Examples:
        >>> tk = [[0.0, -2.0]] * 3 + [[-4.0, -6.0]] * 3    # conf 1,1,1,5,5,5
        >>> deepconf_online_stop(tk, threshold=2.0, window=3) is None  # all >= 2? no
        False
        >>> # first 3-window mean is 1.0 (< 2.0) ending at token index 2
        >>> deepconf_online_stop(tk, threshold=2.0, window=3)
        2
        >>> deepconf_online_stop(tk, threshold=0.5, window=3) is None  # never < 0.5
        True
    """
    conf = token_confidence(topk_logprobs)
    w = min(int(window), conf.size)
    groups = _group_confidences(conf, w)
    below = np.where(groups < threshold)[0]
    if below.size == 0:
        return None
    return int(below[0] + w - 1)
