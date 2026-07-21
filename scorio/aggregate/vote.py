r"""Vote-based answer aggregation for test-time scaling.

Given a pool of candidate answers per question, these rules return a single
answer by *counting*: each candidate casts a vote for its answer and the answer
with the largest total wins. They differ only in how a candidate's vote is
weighted and which candidates are allowed to vote:

- plain **majority vote** -- unit weights (self-consistency);
- **weighted** vote -- weight equal to the raw score;
- **softmax** vote -- weight ``exp(score / T)`` (temperature ``T``);
- **rank-weighted** (Borda) vote -- weight from the score's ordinal rank;
- **logit-weighted** vote -- a threshold-shifted log-odds weight that can go
  negative;
- **filtered** vote -- only the top-scoring candidates are allowed to vote.

All but ``majority_vote`` consume per-candidate reward/verifier/confidence
scores (higher is better). Every rule groups candidates by answer equality and
returns the winning group's answer, with ties broken by earliest appearance.

Methods
-------
- ``majority_vote``: the most frequent answer (self-consistency; Wang et al.,
  2023).
- ``weighted_majority_vote``: the answer whose group maximizes the aggregated
  score (verifier-weighted voting; Li et al., 2023).
- ``softmax_weighted_vote``: temperature-softmax-weighted vote bridging majority
  vote and Best-of-N (CISC; Taubenfeld et al., 2025).
- ``rank_weighted_vote``: Borda / rank-weighted vote, invariant to monotone
  rescaling of the scores (Kang et al., 2025).
- ``logit_weighted_vote``: threshold-shifted log-odds weighted vote where
  low-scoring candidates vote against their answer (Kuang et al., 2025).
- ``filtered_vote``: keep the top-scoring candidates, then (weighted) vote
  (DeepConf; Fu et al., 2025; Cobbe et al., 2021).

All operate on ``(N,)`` (one question) or ``(M, N)`` (a batch of questions) and
return only the selected answer(s); measuring the accuracy of a selection is
left to :mod:`scorio.eval` (see the subpackage docstring).
"""

from __future__ import annotations

import math
from typing import Any

from ._base import (
    _finalize,
    _finalize_index,
    _is_valid,
    _keep_count,
    _normalize,
    _pack,
    _plurality,
    _valid_indices,
)

__all__ = [
    "majority_vote",
    "weighted_majority_vote",
    "softmax_weighted_vote",
    "rank_weighted_vote",
    "logit_weighted_vote",
    "filtered_vote",
]


def _row_majority(ans_row: Any) -> tuple[Any, int]:
    """Most frequent valid answer in a single row; ties broken by first appearance."""
    tally: dict[Any, list[int]] = {}  # answer -> [count, first_index]
    for j, a in enumerate(ans_row):
        if not _is_valid(a):
            continue
        if a not in tally:
            tally[a] = [0, j]
        tally[a][0] += 1
    if not tally:
        return None, -1
    best = min(tally, key=lambda a: (-tally[a][0], tally[a][1]))
    return best, tally[best][1]


def _row_weighted(ans_row: Any, score_row: Any, aggregate: str) -> tuple[Any, int]:
    """Answer whose group maximizes summed/mean score; ties broken by first appearance."""
    groups: dict[Any, dict[str, Any]] = {}
    for j, a in enumerate(ans_row):
        if not _is_valid(a):
            continue
        s = float(score_row[j])
        g = groups.get(a)
        if g is None:
            groups[a] = {"sum": s, "count": 1, "first": j, "best_s": s, "best_i": j}
        else:
            g["sum"] += s
            g["count"] += 1
            if s > g["best_s"]:
                g["best_s"] = s
                g["best_i"] = j
    if not groups:
        return None, -1

    def weight(a: Any) -> float:
        g = groups[a]
        return g["sum"] / g["count"] if aggregate == "mean" else g["sum"]

    best = min(groups, key=lambda a: (-weight(a), groups[a]["first"]))
    return best, groups[best]["best_i"]


def majority_vote(answers: Any, *, return_index: bool = False) -> Any:
    r"""
    Majority vote (self-consistency) over a candidate pool.

    Selects the answer that appears most frequently among the sampled
    candidates for each question. This is the *self-consistency* decoding rule:
    sample several chain-of-thought completions, marginalize over the reasoning
    paths, and keep the most common final answer.

    References:
        Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S.,
        Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of
        Thought Reasoning in Language Models. *ICLR 2023*, *arXiv:2203.11171*.
        https://arxiv.org/abs/2203.11171

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Entries
            that are ``None``, ``""`` or ``NaN`` are treated as unparsable and
            ignored. Answers are compared by equality, so any hashable label
            (string, int, canonicalized expression) works.
        return_index: If ``True``, also return a representative candidate index
            per question (the first occurrence of the winning answer).

    Returns:
        The selected answer for each question. This is a scalar for ``(N,)``
        input or an ``(M,)`` object array for ``(M, N)`` input. With
        ``return_index``, a pair ``(selected, index)`` is returned, where
        ``index`` is ``-1`` for a question whose candidates are all unparsable.

    Formula:
        Let :math:`Z_{\alpha\cdot}` be the answers for question :math:`\alpha`
        and :math:`c_\alpha(z) = \sum_i \mathbb{1}[Z_{\alpha i} = z]`. Then

        .. math::

            \hat z_\alpha = \operatorname*{arg\,max}_{z} c_\alpha(z),

        with ties broken by earliest appearance in the sample order.

    Examples:
        >>> answers = ["A", "B", "A", "C", "A"]
        >>> majority_vote(answers)
        'A'
        >>> majority_vote(answers, return_index=True)
        ('A', 0)
        >>> batch = [["A", "B", "A"], ["X", "X", "Y"]]
        >>> majority_vote(batch).tolist()
        ['A', 'X']
    """
    Z, _, single = _normalize(answers)
    selected: list[Any] = []
    indices: list[int] = []
    for row in Z:
        a, idx = _row_majority(row)
        selected.append(a)
        indices.append(idx)
    if return_index:
        return _finalize(selected, single), _finalize_index(indices, single)
    return _finalize(selected, single)


def weighted_majority_vote(
    answers: Any,
    scores: Any,
    *,
    aggregate: str = "sum",
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""
    Verifier-weighted majority vote over a candidate pool.

    Groups candidates by answer and selects the group with the largest
    aggregated score. With ``aggregate="sum"`` each answer is weighted by the
    total score of its candidates (so agreement *and* quality both help); with
    ``aggregate="mean"`` answers are ranked by average candidate quality. When
    the scores are a verifier's correctness probabilities this is the weighted
    self-consistency of Li et al. (2023). With ``aggregate="sum"``, unit weights
    recover plain :func:`majority_vote`. With ``aggregate="mean"``, unit weights
    give every nonempty answer group the same score, so the tie-break rule decides
    the result.

    References:
        Li, Y., Lin, Z., Zhang, S., Fu, Q., Chen, B., Lou, J.-G., & Chen, W.
        (2023). Making Language Models Better Reasoners with Step-Aware
        Verifier. *ACL 2023*, *arXiv:2206.02336*.
        https://arxiv.org/abs/2206.02336

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Unparsable
            entries (``None``, ``""``, ``NaN``) are ignored.
        scores: ``(N,)`` or ``(M, N)`` array-like of per-candidate scores
            (reward-model / verifier scores; higher is better), same shape as
            ``answers``.
        aggregate: ``"sum"`` (default) or ``"mean"`` group aggregation.
        return_index: If ``True``, also return the index of the highest-scoring
            member of the winning group per question.
        return_score: If ``True``, also return the score of that representative
            candidate -- the best reward within the winning group (``NaN`` if no
            candidate has a valid answer). Note this is the representative's own
            score, not the aggregated group weight that decided the vote.

    Returns:
        The selected answer per question (scalar or ``(M,)`` object array). When
        ``return_index`` and/or ``return_score`` are set, a tuple
        ``(selected[, index][, score])`` is returned in that order.

    Formula:
        With group score :math:`W_\alpha(z)` the answer is

        .. math::

            W_\alpha(z) = \sum_{i:\, Z_{\alpha i} = z} S_{\alpha i}
            \quad(\text{sum}), \qquad
            W_\alpha(z) = \frac{\sum_{i:\, Z_{\alpha i}=z} S_{\alpha i}}
                               {\sum_i \mathbb{1}[Z_{\alpha i}=z]}
            \quad(\text{mean}),

        and :math:`\hat z_\alpha = \arg\max_z W_\alpha(z)`, ties broken by
        earliest appearance.

    Examples:
        >>> answers = ["A", "A", "A", "B"]
        >>> scores = [0.3, 0.3, 0.3, 0.85]
        >>> weighted_majority_vote(answers, scores)          # sum: A=0.9 > B=0.85
        'A'
        >>> weighted_majority_vote(answers, scores, aggregate="mean")  # mean: A=0.3 < B=0.85
        'B'
        >>> weighted_majority_vote(answers, scores, return_score=True)  # best A-member reward
        ('A', 0.3)
    """
    if aggregate not in ("sum", "mean"):
        raise ValueError(f"aggregate must be 'sum' or 'mean'; got {aggregate!r}.")
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    selected: list[Any] = []
    indices: list[int] = []
    sel_scores: list[float] = []
    for row, srow in zip(Z, S, strict=True):
        a, idx = _row_weighted(row, srow, aggregate)
        selected.append(a)
        indices.append(idx)
        sel_scores.append(float(srow[idx]) if idx >= 0 else float("nan"))
    return _pack(
        selected,
        indices,
        sel_scores,
        single,
        return_index=return_index,
        return_score=return_score,
    )


def _row_softmax(ans_row: Any, score_row: Any, temperature: float) -> tuple[Any, int]:
    """Softmax-weighted plurality for one row; ties broken by first appearance."""
    part = _valid_indices(ans_row)
    if not part:
        return None, -1
    smax = max(float(score_row[j]) for j in part)
    w = {j: math.exp((float(score_row[j]) - smax) / temperature) for j in part}
    return _plurality(ans_row, part, lambda j: w[j], lambda j: float(score_row[j]))


def _row_rank(ans_row: Any, score_row: Any, p: float) -> tuple[Any, int]:
    """Rank-weighted (Borda) plurality for one row; ties broken by first appearance."""
    part = _valid_indices(ans_row)
    if not part:
        return None, -1
    n = len(part)
    order = sorted(part, key=lambda j: (-float(score_row[j]), j))
    if float(p).is_integer():
        # Exact arbitrary-precision integer weights (n - t) ** p: no overflow and
        # exact tie detection, matching the package's integer-weight convention.
        pe = int(p)
        w: dict[int, float] = {j: (n - t) ** pe for t, j in enumerate(order)}
    else:
        # Non-integer p: ((n - t) / n) ** p is the paper's (n - t) ** p divided by
        # the per-question constant n ** p (argmax-invariant), and its (0, 1] base
        # cannot overflow the way int ** float can for large exponents.
        w = {j: ((n - t) / n) ** p for t, j in enumerate(order)}
    return _plurality(ans_row, part, lambda j: w[j], lambda j: float(score_row[j]))


def _row_logit(
    ans_row: Any, score_row: Any, threshold: float, transform: str
) -> tuple[Any, int]:
    """Log-odds / linear threshold-shifted weighted plurality for one row."""
    part = _valid_indices(ans_row)
    if not part:
        return None, -1
    if transform == "logit":
        tb = math.log(threshold / (1.0 - threshold))

        def weight_of(j: int) -> float:
            s = float(score_row[j])
            return math.log(s / (1.0 - s)) - tb
    else:  # "linear"

        def weight_of(j: int) -> float:
            return float(score_row[j]) - threshold

    return _plurality(ans_row, part, weight_of, lambda j: float(score_row[j]))


def _row_filtered(
    ans_row: Any, score_row: Any, keep: Any, weighted: bool
) -> tuple[Any, int]:
    """Vote over the top-``keep`` scoring candidates for one row."""
    part = _valid_indices(ans_row)
    if not part:
        return None, -1
    k = _keep_count(keep, len(part))
    kept = sorted(part, key=lambda j: (-float(score_row[j]), j))[:k]

    def weight_of(j: int) -> float:
        return float(score_row[j]) if weighted else 1.0

    return _plurality(ans_row, kept, weight_of, lambda j: float(score_row[j]))


def _run_vote(
    row_fn: Any,
    Z: Any,
    S: Any,
    single: bool,
    *,
    return_index: bool,
    return_score: bool,
) -> Any:
    """Apply a per-row score-aware vote helper across all rows and pack the output."""
    selected: list[Any] = []
    indices: list[int] = []
    sel_scores: list[float] = []
    for row, srow in zip(Z, S, strict=True):
        a, idx = row_fn(row, srow)
        selected.append(a)
        indices.append(idx)
        sel_scores.append(float(srow[idx]) if idx >= 0 else float("nan"))
    return _pack(
        selected,
        indices,
        sel_scores,
        single,
        return_index=return_index,
        return_score=return_score,
    )


def softmax_weighted_vote(
    answers: Any,
    scores: Any,
    *,
    temperature: float = 1.0,
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""
    Temperature-softmax-weighted majority vote (CISC).

    A weighted vote in which each candidate's weight is a *softmax* of its
    score, :math:`\exp(S_i / T)`, rather than the raw score. The single
    temperature :math:`T > 0` continuously bridges the three basic rules:
    :math:`T \to \infty` flattens the weights to majority vote, while
    :math:`T \to 0` concentrates all weight on the highest-scoring candidate,
    recovering Best-of-N (exactly so when the top score is unique). This is the
    Confidence-Informed Self-Consistency
    (CISC) rule, where the score is the model's own confidence in each response.

    References:
        Taubenfeld, A., Sheffer, T., Ofek, E., Feder, A., Goldstein, A.,
        Gekhman, Z., & Yona, G. (2025). Confidence Improves Self-Consistency in
        LLMs. *Findings of ACL 2025*, *arXiv:2502.06233* (Definition 3.1).
        https://arxiv.org/abs/2502.06233

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Unparsable
            entries (``None``, ``""``, ``NaN``) are ignored.
        scores: ``(N,)`` or ``(M, N)`` array-like of per-candidate scores
            (higher is better), same shape as ``answers``.
        temperature: Softmax temperature :math:`T > 0`. Smaller sharpens toward
            Best-of-N; larger flattens toward majority vote (``float('inf')``
            gives exactly majority vote).
        return_index: If ``True``, also return the highest-scoring member of the
            winning answer group per question (``-1`` if none are valid).
        return_score: If ``True``, also return the score of that representative
            candidate (``NaN`` if none are valid).

    Returns:
        The selected answer per question (scalar or ``(M,)`` object array). When
        ``return_index`` and/or ``return_score`` are set, a tuple
        ``(selected[, index][, score])`` is returned in that order.

    Formula:
        The paper's softmax weights :math:`\tilde c_i = \exp(S_{\alpha i}/T) /
        \sum_j \exp(S_{\alpha j}/T)` share a per-question denominator that
        cancels in the ``argmax``, so the selection is equivalently

        .. math::

            \hat z_\alpha = \operatorname*{arg\,max}_z
            \sum_{i:\, Z_{\alpha i} = z} \exp\!\big(S_{\alpha i} / T\big),

        with ties broken by earliest appearance. (Implemented with the standard
        ``exp((S - max S) / T)`` shift for numerical stability, which leaves the
        ``argmax`` unchanged.)

    Examples:
        >>> answers = ["A", "A", "B"]
        >>> scores = [0.1, 0.2, 0.9]
        >>> softmax_weighted_vote(answers, scores, temperature=0.1)  # sharp -> Best-of-N
        'B'
        >>> softmax_weighted_vote(answers, scores, temperature=float("inf"))  # -> majority
        'A'
    """
    if not (temperature > 0.0):
        raise ValueError(f"temperature must be > 0; got {temperature}.")
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    return _run_vote(
        lambda row, srow: _row_softmax(row, srow, temperature),
        Z,
        S,
        single,
        return_index=return_index,
        return_score=return_score,
    )


def rank_weighted_vote(
    answers: Any,
    scores: Any,
    *,
    p: float = 1.0,
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""
    Rank-weighted (Borda) majority vote.

    A weighted vote that uses only the *ordinal rank* of the scores, not their
    magnitudes. Candidates are ranked by score (best first) and the rank-``t``
    candidate (``t = 0`` is the best) casts a vote of weight :math:`(n - t)^p`.
    Because it depends only on the score ordering, the rule is invariant to any
    monotone rescaling of the scores -- a robustness the magnitude-based
    :func:`weighted_majority_vote` does not have. The exponent :math:`p \ge 0`
    interpolates between majority vote (``p = 0``, all weights equal) and
    Best-of-N (``p \to \infty``, the top candidate dominates); ``p = 1`` is the
    standard Borda count. Introduced as *self-certainty Borda voting*.

    References:
        Kang, Z., Zhao, X., & Song, D. (2025). Scalable Best-of-N Selection for
        Large Language Models via Self-Certainty. *NeurIPS 2025*,
        *arXiv:2502.18581*. https://arxiv.org/abs/2502.18581

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Unparsable
            entries (``None``, ``""``, ``NaN``) are ignored.
        scores: ``(N,)`` or ``(M, N)`` array-like of per-candidate scores
            (higher is better), same shape as ``answers``.
        p: Non-negative Borda exponent. ``0`` recovers majority vote; ``1`` is
            standard Borda; larger values sharpen toward Best-of-N.
        return_index: If ``True``, also return the highest-scoring member of the
            winning answer group per question (``-1`` if none are valid).
        return_score: If ``True``, also return the score of that representative
            candidate (``NaN`` if none are valid).

    Returns:
        The selected answer per question (scalar or ``(M,)`` object array). When
        ``return_index`` and/or ``return_score`` are set, a tuple
        ``(selected[, index][, score])`` is returned in that order.

    Formula:
        Ranking the :math:`n` valid candidates for question :math:`\alpha` by
        descending score (ties by lowest index) and letting :math:`r_{\alpha i}
        \in \{1,\ldots,n\}` be candidate :math:`i`'s rank,

        .. math::

            \hat z_\alpha = \operatorname*{arg\,max}_z
            \sum_{i:\, Z_{\alpha i} = z} (n - r_{\alpha i} + 1)^{p},

        with ties broken by earliest appearance.

    Examples:
        >>> answers = ["A", "A", "B"]
        >>> scores = [0.30, 0.31, 0.99]
        >>> rank_weighted_vote(answers, scores, p=1.0)   # Borda: A gets 2+1, B gets 3
        'A'
        >>> rank_weighted_vote(answers, scores, p=0.0)   # -> majority vote
        'A'
    """
    if not (p >= 0.0) or p == float("inf"):
        raise ValueError(f"p must be a finite non-negative number; got {p}.")
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    return _run_vote(
        lambda row, srow: _row_rank(row, srow, p),
        Z,
        S,
        single,
        return_index=return_index,
        return_score=return_score,
    )


def logit_weighted_vote(
    answers: Any,
    scores: Any,
    *,
    threshold: float = 0.5,
    transform: str = "logit",
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""
    Threshold-shifted log-odds weighted majority vote.

    The Bayes-optimal aggregation of a generator's samples with a verifier's
    scores casts each candidate's vote with a weight that is a *log-likelihood
    ratio* of its score, approximated in closed form by a threshold-shifted
    transform. With ``transform="logit"`` the weight is
    :math:`\operatorname{logit}(S_i) - \operatorname{logit}(b)` (scores must be
    probabilities in :math:`(0, 1)`); with ``transform="linear"`` it is the
    simpler :math:`S_i - b`. Unlike :func:`weighted_majority_vote`, weights can
    be **negative**: candidates scoring below the threshold :math:`b` vote
    *against* their own answer, so an answer supported only by low-quality
    candidates can be beaten by a rarer but higher-quality one.

    References:
        Kuang, P., Wang, Y., Han, X., Liu, Y., Xu, K., & Wang, H. (2025).
        Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time
        Scaling. *ICLR 2026*, *arXiv:2510.13918* (Theorem 3.2).
        https://arxiv.org/abs/2510.13918

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Unparsable
            entries (``None``, ``""``, ``NaN``) are ignored.
        scores: ``(N,)`` or ``(M, N)`` array-like of per-candidate scores. For
            ``transform="logit"`` every valid score must lie strictly in
            ``(0, 1)`` (a verifier probability); for ``transform="linear"`` any
            real score is allowed.
        threshold: The calibrated decision threshold :math:`b`. For
            ``transform="logit"`` it must lie in ``(0, 1)`` (default ``0.5``,
            i.e. a zero log-odds shift); for ``transform="linear"`` any real
            value (``0.0`` recovers :func:`weighted_majority_vote` with
            ``aggregate="sum"``).
        transform: ``"logit"`` (log-odds; default) or ``"linear"``.
        return_index: If ``True``, also return the highest-scoring member of the
            winning answer group per question (``-1`` if none are valid). Note the
            winning group may have a negative total weight.
        return_score: If ``True``, also return the score of that representative
            candidate (``NaN`` if none are valid).

    Returns:
        The selected answer per question (scalar or ``(M,)`` object array). When
        ``return_index`` and/or ``return_score`` are set, a tuple
        ``(selected[, index][, score])`` is returned in that order.

    Formula:
        With per-candidate weight :math:`w_{\alpha i} =
        \operatorname{logit}(S_{\alpha i}) - \operatorname{logit}(b)` (or
        :math:`S_{\alpha i} - b` for the linear transform),

        .. math::

            \hat z_\alpha = \operatorname*{arg\,max}_z
            \sum_{i:\, Z_{\alpha i} = z} w_{\alpha i},

        with ties broken by earliest appearance. This is the paper's *practical
        parametric* instantiation: the per-response LLM-reliability offset of the
        full Theorem 3.2 weight is folded into the calibrated threshold ``b``
        (the zero-crossing of the log-odds weight), so only ``b`` is exposed
        here. The nonparametric log-density-ratio weight is left to a caller who
        has a calibration set.

    Examples:
        >>> answers = ["A", "A", "B"]
        >>> scores = [0.4, 0.4, 0.9]     # both A-candidates below b=0.5
        >>> logit_weighted_vote(answers, scores)          # A votes negative -> B wins
        'B'
        >>> from scorio.aggregate import weighted_majority_vote
        >>> a, s = ["A", "A", "B"], [0.4, 0.4, 0.9]
        >>> logit_weighted_vote(a, s, transform="linear", threshold=0.0) == \
        ...     weighted_majority_vote(a, s)              # linear, b=0 -> weighted sum
        True
    """
    if transform not in ("logit", "linear"):
        raise ValueError(f"transform must be 'logit' or 'linear'; got {transform!r}.")
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    if transform == "logit":
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"threshold must be in (0, 1) for transform='logit'; got {threshold}."
            )
        # Validate scores lie strictly in (0, 1) for the log-odds transform.
        for row, srow in zip(Z, S, strict=True):
            for j in _valid_indices(row):
                s = float(srow[j])
                if not (0.0 < s < 1.0):
                    raise ValueError(
                        "transform='logit' requires every valid score in (0, 1); "
                        f"got {s}. Use transform='linear' for unbounded scores."
                    )
    return _run_vote(
        lambda row, srow: _row_logit(row, srow, threshold, transform),
        Z,
        S,
        single,
        return_index=return_index,
        return_score=return_score,
    )


def filtered_vote(
    answers: Any,
    scores: Any,
    *,
    keep: float = 0.5,
    weighted: bool = True,
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""
    Confidence-filtered majority vote (DeepConf / top-K verifier-ranked vote).

    Drops the lowest-scoring candidates *before* voting, keeping only the
    top-``keep`` by score, then takes a (weighted) majority vote over the
    survivors. Filtering out low-quality candidates concentrates the vote on the
    model's most confident responses; the retained set then votes with weight
    equal to the score (``weighted=True``, DeepConf's confidence-weighted vote)
    or with unit weights (``weighted=False``, the classic top-K
    verifier-ranked majority vote of Cobbe et al.).

    References:
        Fu, Y., Wang, X., Tian, Y., & Zhao, J. (2025). Deep Think with
        Confidence. *arXiv:2508.15260*. https://arxiv.org/abs/2508.15260

        Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L.,
        Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman,
        J. (2021). Training Verifiers to Solve Math Word Problems (Sec. 5.1,
        top-K verifier voting). *arXiv:2110.14168*.

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Unparsable
            entries (``None``, ``""``, ``NaN``) are ignored.
        scores: ``(N,)`` or ``(M, N)`` array-like of per-candidate scores
            (higher is better), same shape as ``answers``.
        keep: How many top-scoring candidates to retain per question. A float in
            ``(0, 1]`` is a *fraction* of the valid candidates (default ``0.5``,
            at least one is always kept); an integer ``>= 1`` is an explicit
            *count*. ``keep=1.0`` keeps everyone (no filtering); ``keep=1``
            (integer) keeps only the single best, recovering Best-of-N.
        weighted: If ``True`` (default), survivors vote with weight equal to
            their score (DeepConf); if ``False``, with unit weights (top-K
            majority vote).
        return_index: If ``True``, also return the highest-scoring member of the
            winning answer group per question (``-1`` if none are valid).
        return_score: If ``True``, also return the score of that representative
            candidate (``NaN`` if none are valid).

    Returns:
        The selected answer per question (scalar or ``(M,)`` object array). When
        ``return_index`` and/or ``return_score`` are set, a tuple
        ``(selected[, index][, score])`` is returned in that order.

    Formula:
        Let :math:`\mathcal T_\alpha` be the ``k`` highest-scoring valid
        candidates for question :math:`\alpha` (``k`` from ``keep``; score ties
        broken by lowest index). Then

        .. math::

            \hat z_\alpha = \operatorname*{arg\,max}_z
            \sum_{i \in \mathcal T_\alpha:\, Z_{\alpha i} = z} w_{\alpha i},
            \qquad
            w_{\alpha i} = \begin{cases} S_{\alpha i} & \text{weighted} \\
            1 & \text{unweighted,} \end{cases}

        with ties broken by earliest appearance.

    Examples:
        >>> answers = ["A", "A", "A", "B", "B", "B"]
        >>> scores = [0.1, 0.2, 0.3, 0.8, 0.85, 0.9]
        >>> filtered_vote(answers, scores, keep=0.5)   # top 3 are all B
        'B'
        >>> filtered_vote(answers, scores, keep=1.0, weighted=False)  # keep all -> majority tie, first
        'A'
    """
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    # Validate `keep` once up front with a representative row length.
    _keep_count(keep, Z.shape[1])
    return _run_vote(
        lambda row, srow: _row_filtered(row, srow, keep, weighted),
        Z,
        S,
        single,
        return_index=return_index,
        return_score=return_score,
    )
