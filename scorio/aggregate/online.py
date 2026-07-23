r"""Online / early-termination rules for test-time scaling.

The offline rules in :mod:`scorio.aggregate` assume a *fixed* candidate budget
and choose which answer to return. The rules here instead decide **when to stop
spending compute** -- either stopping the outer sampling loop once the answer is
already decided, or terminating a single reasoning trace mid-generation once it
looks unpromising. They turn a fixed-budget aggregator into an adaptive one that
spends fewer samples/tokens on easy questions.

Two granularities:

- **Stop sampling** (across traces): :func:`adaptive_consistency_stop`, its
  :func:`adaptive_consistency_dirichlet_stop` and
  :func:`adaptive_consistency_crp_stop` variants, and :func:`esc_stop` watch the
  stream of extracted answers and report when to stop drawing new traces.
- **Stop generating** (within a trace): :func:`deepconf_online_stop` reports the
  token at which a trace's running group confidence falls below a threshold set
  from warmup traces by :func:`deepconf_stop_threshold`.

Both stopping styles are complementary to :mod:`scorio.sinf`, which decides how
many samples to draw from ``(mu, sigma)`` estimates with anytime-valid
guarantees; the rules here operate directly on answer counts / token confidences.

Methods
-------
- ``adaptive_consistency_stop``: the paper's fast top-two Beta approximation.
- ``adaptive_consistency_dirichlet_stop``: the full observed-support Dirichlet
  probability from Equation (1) (Aggarwal et al., 2023).
- ``adaptive_consistency_crp_stop``: the paper's unseen-answer CRP Monte Carlo
  comparator (Aggarwal et al., 2023, Appendix C.3).
- ``esc_stop``: stop sampling when a whole window of samples agrees
  (Early-Stopping Self-Consistency; Li et al., 2024).
- ``deepconf_stop_threshold`` / ``deepconf_online_stop``: set an offline-warmup
  confidence threshold, then find a trace's early-termination token (DeepConf
  online; Fu et al., 2025).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import betainc, gammainc, gammaincinv

from ._base import _is_valid
from .confidence import _group_confidences, token_confidence

__all__ = [
    "adaptive_consistency_stop",
    "adaptive_consistency_dirichlet_stop",
    "adaptive_consistency_crp_stop",
    "esc_stop",
    "deepconf_stop_threshold",
    "deepconf_online_stop",
]


def _ordered_answer_counts(answers: Any) -> list[int]:
    """Valid answer counts ordered by first appearance."""
    tally: dict[Any, int] = {}
    for answer in answers:
        if _is_valid(answer):
            tally[answer] = tally.get(answer, 0) + 1
    return list(tally.values())


def _top_two_counts(answers: Any) -> tuple[int, int]:
    """Counts of the two most frequent valid answers (``0`` if fewer exist)."""
    counts = sorted(_ordered_answer_counts(answers), reverse=True)
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


def _dirichlet_leader_probability(counts: list[int]) -> float:
    r"""Probability the count leader is largest under ``Dirichlet(counts + 1)``.

    Independent ``G_j ~ Gamma(alpha_j, 1)`` variables normalized by their sum
    are Dirichlet and preserve component order.  Applying the probability
    integral transform to the leader's Gamma variable gives a bounded integral
    over ``[0, 1]``.  That transform is important here: direct quadrature over
    ``[0, infinity)`` can entirely miss Gamma densities centered at large
    answer counts.
    """
    if not counts:
        return 0.0

    leader = max(range(len(counts)), key=lambda index: counts[index])
    alpha = np.asarray(counts, dtype=float) + 1.0
    alpha_leader = float(alpha[leader])
    other = np.delete(alpha, leader)

    # Exchangeability makes this exact and avoids unnecessary quadrature in a
    # numerically delicate case.
    if np.all(other == alpha_leader):
        return 1.0 / len(counts)

    def integrand(quantile: float) -> float:
        if quantile <= 0.0:
            return 0.0
        if quantile >= 1.0:
            return 1.0
        value = gammaincinv(alpha_leader, quantile)
        cdf = gammainc(other, value)
        if np.any(cdf <= 0.0):
            return 0.0
        return float(np.exp(np.sum(np.log(cdf))))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            probability, error = quad(
                integrand,
                0.0,
                1.0,
                epsabs=1e-10,
                epsrel=1e-10,
                limit=250,
            )
    except IntegrationWarning as exc:
        raise RuntimeError("Dirichlet leader-probability integration failed.") from exc

    tolerance = max(1e-8, 1e-7 * abs(probability))
    if not np.isfinite(probability) or not np.isfinite(error) or error > tolerance:
        raise RuntimeError(
            "Dirichlet leader-probability integration did not converge "
            f"(estimated error {error!r})."
        )
    return float(np.clip(probability, 0.0, 1.0))


def adaptive_consistency_dirichlet_stop(
    answers: Any,
    *,
    threshold: float = 0.95,
    return_prob: bool = False,
) -> Any:
    r"""Full Dirichlet Adaptive-Consistency stopping criterion.

    Unlike :func:`adaptive_consistency_stop`, which retains only the top two
    answer counts, this evaluates Equation (1) over every distinct answer
    observed so far.  With counts ``v`` and the paper's uniform prior, it uses
    ``p | observations ~ Dirichlet(v + 1)`` and stops when the posterior
    probability that the current fixed-tie-broken count leader has the largest
    component reaches ``threshold``.

    For three or more observed answers, the probability is evaluated by an
    exact one-dimensional independent-Gamma integral rather than Monte Carlo.
    Following the authors' released implementation, fewer than three observed
    answers use the top-two Beta criterion.  For two answers the probabilities
    are mathematically identical; for one answer the fallback avoids assigning
    probability one merely because no alternative has appeared yet.

    References:
        Aggarwal, P., Madaan, A., Yang, Y., & Mausam. (2023). Let's Sample Step
        by Step: Adaptive-Consistency for Efficient Reasoning and Coding with
        LLMs. *EMNLP 2023*, *arXiv:2305.11860* (Eq. 1 and Appendix D).
        https://arxiv.org/abs/2305.11860

    Args:
        answers: The one-dimensional sequence of extracted answers sampled so
            far. Unparsable entries are ignored; count ties are broken by
            earliest answer appearance.
        threshold: Posterior leader-probability threshold in ``(0, 1)``.
        return_prob: If ``True``, return ``(stop, probability)``.

    Returns:
        Whether the Dirichlet leader probability has reached ``threshold``.
        With ``return_prob=True``, also returns that probability. No valid
        answer yields ``(False, 0.0)``.

    Formula:
        For observed answer counts :math:`v=(v_1,\ldots,v_m)` and the paper's
        uniform prior,

        .. math::

            p\mid v\sim\operatorname{Dirichlet}(v_1+1,\ldots,v_m+1).

        If answer 1 is the fixed-tie-broken count leader, stop when

        .. math::

            \Pr\!\left(p_1>\max_{j>1}p_j\mid v\right)\ge C.

        For :math:`m\ge3`, the implementation evaluates the equivalent bounded
        integral

        .. math::

            \int_0^1 \prod_{j>1}F_{G_j}
              \!\left(F_{G_1}^{-1}(u)\right)\,du,
            \qquad G_j\sim\operatorname{Gamma}(v_j+1,1).

    Notes:
        "Full" refers to using every currently observed answer category rather
        than only the two largest counts. It does not reserve probability mass
        for an answer category that has never appeared; use
        :func:`adaptive_consistency_crp_stop` for that model. The authors'
        released code truncates its numerical approximation to the five largest
        observed counts; this implementation instead retains every category in
        Equation (1).

    Examples:
        >>> stop, probability = adaptive_consistency_dirichlet_stop(
        ...     ["A"] * 5 + ["B"] * 2 + ["C"], return_prob=True
        ... )
        >>> stop, round(probability, 4)
        (False, 0.818)
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}.")

    counts = _ordered_answer_counts(answers)
    if not counts:
        probability = 0.0
    elif len(counts) < 3:
        ordered = sorted(counts, reverse=True)
        v1 = ordered[0]
        v2 = ordered[1] if len(ordered) > 1 else 0
        probability = float(1.0 - betainc(v1 + 1, v2 + 1, 0.5))
    else:
        probability = _dirichlet_leader_probability(counts)

    stop = bool(counts) and probability >= threshold
    return (stop, probability) if return_prob else stop


_CRP_MAX_CHUNK_SIZE = 8192
_CRP_TARGET_CELLS = 500_000


def _crp_leader_probability(
    counts: list[int],
    *,
    horizon: int,
    n_alpha: int,
    n_simulations: int,
    seed: int | None,
) -> float:
    """Monte Carlo probability that the current leader wins at ``horizon``."""
    counts_array = np.asarray(counts, dtype=np.int64)
    n = int(np.sum(counts_array))
    k = len(counts)
    remaining = horizon - n
    leader = int(np.argmax(counts_array))
    rng = np.random.default_rng(seed)

    rate = 1.0 + float(np.euler_gamma) + math.log(n)
    alpha_draws = rng.gamma(shape=float(k), scale=1.0 / rate, size=n_alpha)
    n_runs = n_alpha * n_simulations

    # Each chunk holds floating-point posterior masses and integer allocations,
    # labels, and cluster counts. Scale it down for unusually large horizons so
    # memory stays bounded independently of the requested Monte Carlo count.
    row_width = 3 * (k + 1) + 2 * remaining
    chunk_size = min(
        _CRP_MAX_CHUNK_SIZE,
        max(1, _CRP_TARGET_CELLS // max(1, row_width)),
    )

    successes = 0
    for start in range(0, n_runs, chunk_size):
        stop = min(start + chunk_size, n_runs)
        alpha_indices = np.arange(start, stop) // n_simulations
        alpha = alpha_draws[alpha_indices]
        batch_size = stop - start

        # Under the DP posterior, masses for existing answers and all unseen
        # answers together are Dirichlet(n_1, ..., n_k, alpha). Conditional on
        # those masses, all remaining draws can be allocated in one multinomial.
        masses = np.empty((batch_size, k + 1), dtype=float)
        masses[:, :k] = rng.gamma(
            shape=counts_array,
            size=(batch_size, k),
        )
        masses[:, k] = rng.gamma(shape=alpha)
        masses /= np.sum(masses, axis=1, keepdims=True)
        allocations = rng.multinomial(remaining, masses)
        final_existing = allocations[:, :k] + counts_array
        unseen_draws = allocations[:, k]

        # Partition only the draws assigned to unseen-answer mass. In a CRP an
        # existing table can be sampled by choosing a previous customer
        # uniformly, so labels avoid a cumulative-count scan at every step.
        labels = np.empty((batch_size, remaining), dtype=np.int32)
        new_counts = np.zeros((batch_size, remaining), dtype=np.int32)
        active_clusters = np.zeros(batch_size, dtype=np.int32)
        rows = np.arange(batch_size)
        for customer in range(remaining):
            active_rows = rows[unseen_draws > customer]
            if active_rows.size == 0:
                break
            draw = rng.random(active_rows.size) * (customer + alpha[active_rows])
            create = draw >= customer
            choice = np.empty(active_rows.size, dtype=np.int32)
            existing = ~create
            if np.any(existing):
                parent = draw[existing].astype(np.intp)
                choice[existing] = labels[active_rows[existing], parent]
            choice[create] = active_clusters[active_rows[create]]
            active_clusters[active_rows[create]] += 1
            labels[active_rows, customer] = choice
            new_counts[active_rows, choice] += 1

        current_wins = np.argmax(final_existing, axis=1) == leader
        current_wins &= final_existing[:, leader] >= np.max(new_counts, axis=1)
        successes += int(np.count_nonzero(current_wins))

    return successes / n_runs


def adaptive_consistency_crp_stop(
    answers: Any,
    *,
    threshold: float = 0.95,
    horizon: int = 40,
    n_alpha: int = 100,
    n_simulations: int = 1000,
    seed: int | None = 0,
    return_prob: bool = False,
) -> Any:
    r"""Chinese-Restaurant-Process Adaptive-Consistency stopping criterion.

    This implements the CRP comparator in Appendix C.3 of Aggarwal et al.
    (2023): estimate unseen-answer mass with a Chinese restaurant process, draw
    its concentration parameter from West's Gamma approximation, simulate to a
    total generation ``horizon``, and estimate how often the current majority
    answer remains the fixed-tie-broken majority.

    The simulation uses the equivalent Dirichlet-process posterior predictive
    distribution: future draws are allocated jointly among observed answers and
    unseen mass, then the unseen draws are partitioned by a CRP. This is
    distributionally identical to adding all future draws one by one, while
    avoiding the staging implementation's quadratic full-matrix scans.

    References:
        Aggarwal, P., Madaan, A., Yang, Y., & Mausam. (2023). Let's Sample Step
        by Step: Adaptive-Consistency for Efficient Reasoning and Coding with
        LLMs. *EMNLP 2023*, *arXiv:2305.11860* (Sec. 5.3 and Appendix C.3).
        https://arxiv.org/abs/2305.11860

        West, M. (1992). Hyperparameter Estimation in Dirichlet Process Mixture
        Models. Duke University technical report.

    Args:
        answers: The one-dimensional sequence of extracted answers sampled so
            far. Unparsable entries are ignored; count ties are broken by
            earliest answer appearance.
        threshold: Monte Carlo stability threshold in ``(0, 1)``.
        horizon: Total number of valid generations at which the future majority
            is evaluated; the paper uses ``40``.
        n_alpha: Number of concentration-parameter draws; the paper uses ``100``.
        n_simulations: Continuations per concentration draw; the paper uses
            ``1000``.
        seed: Integer random seed for reproducible Monte Carlo draws (default
            ``0``), or ``None`` for nondeterministic draws.
        return_prob: If ``True``, return ``(stop, probability)``.

    Returns:
        Whether the estimated probability that the current leader remains the
        fixed-tie-broken leader at ``horizon`` reaches ``threshold``. With
        ``return_prob=True``, also returns that estimate. No valid answer yields
        ``(False, 0.0)``. A prefix already at or beyond ``horizon`` yields
        ``(True, 1.0)`` because its sampling budget is exhausted.

    Formula:
        With :math:`k` observed clusters and :math:`n` valid answers, the paper
        approximates the concentration posterior (with unit hyperparameters) as

        .. math::

            \alpha\mid k,n\sim\operatorname{Gamma}\!\left(
              k,\ 1+\gamma+\log n\right),

        where the second argument is a rate and :math:`\gamma` is Euler's
        constant. Given :math:`\alpha`, current cluster :math:`i` receives the
        next answer with probability :math:`n_i/(n+\alpha)` and a new cluster
        appears with probability :math:`\alpha/(n+\alpha)`.

    Notes:
        The paper specifies this comparator but its released repository does not
        include an executable CRP implementation. The result is a finite-horizon
        Monte Carlo estimate; report ``horizon``, ``n_alpha``,
        ``n_simulations``, and ``seed`` with experimental results.

    Examples:
        >>> result = adaptive_consistency_crp_stop(
        ...     ["A"] * 7 + ["B"], threshold=0.5, horizon=12,
        ...     n_alpha=12, n_simulations=100, seed=123,
        ...     return_prob=True,
        ... )
        >>> isinstance(result[0], bool), 0.0 <= result[1] <= 1.0
        (True, True)
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}.")
    for value, name in (
        (horizon, "horizon"),
        (n_alpha, "n_alpha"),
        (n_simulations, "n_simulations"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer >= 1; got {value!r}.")
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1; got {value}.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError(f"seed must be a non-negative integer or None; got {seed!r}.")
    try:
        np.random.default_rng(seed)
    except ValueError as exc:
        raise ValueError(
            f"seed must be a non-negative integer or None; got {seed!r}."
        ) from exc

    counts = _ordered_answer_counts(answers)
    if not counts:
        result = (False, 0.0)
        return result if return_prob else result[0]

    n = int(sum(counts))
    horizon = int(horizon)
    if n >= horizon:
        result = (True, 1.0)
        return result if return_prob else result[0]

    probability = _crp_leader_probability(
        counts,
        horizon=horizon,
        n_alpha=int(n_alpha),
        n_simulations=int(n_simulations),
        seed=None if seed is None else int(seed),
    )
    stop = probability >= threshold
    return (stop, probability) if return_prob else stop


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
