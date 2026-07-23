r"""Reward-based answer selection for test-time scaling.

Given a pool of candidate answers per question, each carrying a reward-model /
verifier score, these rules pick a final answer using the scores. ``best_of_n``
takes the single highest-scoring candidate; ``majority_of_the_bests`` replaces
that noisy argmax with the *mode of the bootstrapped Best-of-N distribution*,
which is more robust when the reward model is imperfect; ``best_of_majority``
first discards rarely-produced answers, then takes the highest-reward answer
among the survivors.

Methods
-------
- ``best_of_n``: the answer of the highest-scoring candidate (Cobbe et al.,
  2021).
- ``majority_of_the_bests`` (alias ``mob``): the most likely Best-of-N outcome
  under bootstrap resampling (Rakhsha et al., 2025).
- ``best_of_majority``: the highest-reward answer among those produced at least
  a frequency ``alpha`` of the time (Di et al., 2025).

All operate on ``(N,)`` (one question) or ``(M, N)`` (a batch) and return only
the selected answer(s); accuracy of a selection is measured downstream with
:mod:`scorio.eval` (see the subpackage docstring).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import _default_m, _is_valid, _normalize, _pack, _valid_indices

__all__ = [
    "best_of_n",
    "majority_of_the_bests",
    "mob",
    "best_of_majority",
]


def _row_best_of_n(ans_row: Any, score_row: Any) -> tuple[Any, int]:
    """Answer of the highest-scoring candidate with a valid answer (ties: lowest index)."""
    best_i = -1
    best_s = float("-inf")
    for j, a in enumerate(ans_row):
        if not _is_valid(a):
            continue
        s = float(score_row[j])
        if s > best_s:
            best_s = s
            best_i = j
    if best_i < 0:
        return None, -1
    return ans_row[best_i], best_i


def _row_mob(ans_row: Any, score_row: Any, m: int | None) -> tuple[Any, int]:
    r"""Mode of the bootstrapped Best-of-N distribution (closed form, :math:`B\to\infty`).

    Sorting the valid candidates by descending score, the candidate at
    reward-rank ``t`` (0-indexed) wins Best-of-N on a size-``m`` with-replacement
    resample with probability ``((n - t)/n)**m - ((n - t - 1)/n)**m``. Summing
    those weights over each answer group yields the exact bootstrap distribution
    of the Best-of-N answer; its mode is returned. Weights are accumulated as
    exact integers scaled by ``n**m`` (``(n - t)**m - (n - t - 1)**m``) so that
    genuine ties between answer groups are detected exactly rather than decided
    by floating-point rounding, and broken by earliest appearance.
    """
    idx = [j for j, a in enumerate(ans_row) if _is_valid(a)]
    if not idx:
        return None, -1
    n = len(idx)
    if n == 1:
        return ans_row[idx[0]], idx[0]

    mm = _default_m(n) if m is None else m
    mm = min(max(int(mm), 1), n)

    # Strict priority order: higher score first, ties broken by lower index.
    order = sorted(idx, key=lambda j: (-float(score_row[j]), j))

    # Exact integer weight per group, scaled by n**mm; avoids float-tie noise.
    weight: dict[Any, int] = {}
    rep: dict[Any, int] = {}  # highest-reward member of each answer group
    for t, j in enumerate(order):
        w = (n - t) ** mm - (n - t - 1) ** mm
        a = ans_row[j]
        if a not in weight:
            weight[a] = 0
            rep[a] = j
        weight[a] += w

    first_occ: dict[Any, int] = {}
    for j in idx:
        a = ans_row[j]
        if a not in first_occ:
            first_occ[a] = j

    best = min(weight, key=lambda a: (-weight[a], first_occ[a]))
    return best, rep[best]


def best_of_n(
    answers: Any,
    scores: Any,
    *,
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""
    Best-of-N: return the answer of the highest-scoring candidate.

    Samples ``N`` candidates per question, scores each with a reward model or
    verifier, and keeps the answer of the single highest-scoring one. This is
    the classic verifier reranking / best-of-:math:`n` selection used to trade
    inference compute for accuracy.

    References:
        Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L.,
        Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman,
        J. (2021). Training Verifiers to Solve Math Word Problems.
        *arXiv:2110.14168*. https://arxiv.org/abs/2110.14168

        Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C.,
        Radford, A., Amodei, D., & Christiano, P. (2020). Learning to Summarize
        from Human Feedback. *NeurIPS 2020*, *arXiv:2009.01325*.

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Candidates
            whose answer is unparsable (``None``, ``""``, ``NaN``) are not
            eligible to be selected.
        scores: ``(N,)`` or ``(M, N)`` array-like of per-candidate scores
            (higher is better), same shape as ``answers``.
        return_index: If ``True``, also return the selected candidate index per
            question (``-1`` if no candidate has a valid answer).
        return_score: If ``True``, also return the score of the selected
            candidate -- for Best-of-N, the maximizing (``argmax``) score
            (``NaN`` if no candidate has a valid answer).

    Returns:
        The selected answer per question (scalar or ``(M,)`` object array). When
        ``return_index`` and/or ``return_score`` are set, a tuple
        ``(selected[, index][, score])`` is returned in that order.

    Formula:
        .. math::

            i^\star_\alpha = \operatorname*{arg\,max}_i S_{\alpha i}, \qquad
            \hat z_\alpha = Z_{\alpha, i^\star_\alpha},

        with ties in the score broken by lowest index.

    Examples:
        >>> answers = ["A", "B", "A", "C", "A"]
        >>> scores = [0.1, 0.9, 0.2, 0.3, 0.15]
        >>> best_of_n(answers, scores)
        'B'
        >>> best_of_n(answers, scores, return_index=True)
        ('B', 1)
        >>> best_of_n(answers, scores, return_score=True)
        ('B', 0.9)
        >>> best_of_n(answers, scores, return_index=True, return_score=True)
        ('B', 1, 0.9)
    """
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    selected: list[Any] = []
    indices: list[int] = []
    sel_scores: list[float] = []
    for row, srow in zip(Z, S, strict=True):
        a, idx = _row_best_of_n(row, srow)
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


def majority_of_the_bests(
    answers: Any,
    scores: Any,
    *,
    m: int | None = None,
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""
    Majority-of-the-Bests (MoB): the mode of the bootstrapped Best-of-N answer.

    Best-of-N returns a single draw from a noisy answer distribution induced by
    an imperfect reward model. MoB instead *estimates that distribution by
    bootstrapping* -- repeatedly resample :math:`m` candidates with replacement,
    apply Best-of-N to each resample, and record the resulting answer -- and
    returns its **mode**. Because the correct answer, while rarely near
    certainty under a flawed reward, is often the single most likely Best-of-N
    outcome, taking the mode improves over vanilla Best-of-N.

    This implementation uses the closed-form (:math:`B\to\infty`) bootstrap
    distribution from the paper rather than Monte-Carlo resampling, so the result
    is exact and deterministic. Sorting candidates by descending score, the
    candidate at reward-rank :math:`t` (0-indexed) is the Best-of-N winner of a
    size-:math:`m` with-replacement resample with probability
    :math:`\left(\tfrac{n-t}{n}\right)^m - \left(\tfrac{n-t-1}{n}\right)^m`;
    these weights are summed within each answer group and the largest group is
    returned.

    References:
        Rakhsha, A., Madan, K., Zhang, T., Farahmand, A.-m., & Khasahmadi, A.
        (2025). Majority of the Bests: Improving Best-of-N via Bootstrapping.
        *arXiv:2511.18630*. https://arxiv.org/abs/2511.18630

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Unparsable
            entries (``None``, ``""``, ``NaN``) are excluded from resampling.
        scores: ``(N,)`` or ``(M, N)`` array-like of per-candidate scores
            (higher is better), same shape as ``answers``.
        m: Bootstrap resample size. Defaults to :math:`\lfloor\sqrt{n}\rfloor`
            over the ``n`` valid candidates, as recommended by the paper;
            clamped to ``[1, n]``.
        return_index: If ``True``, also return the highest-scoring member of the
            winning answer group per question (``-1`` if none are valid).
        return_score: If ``True``, also return the score of that representative
            candidate -- the best reward within the winning answer group
            (``NaN`` if none are valid).

    Returns:
        The selected answer per question (scalar or ``(M,)`` object array). When
        ``return_index`` and/or ``return_score`` are set, a tuple
        ``(selected[, index][, score])`` is returned in that order.

    Formula:
        Let candidates be ranked by score, :math:`z_{(t)}` the answer at rank
        :math:`t`, and

        .. math::

            w_t = \Big(\tfrac{n-t}{n}\Big)^{m}
                 - \Big(\tfrac{n-t-1}{n}\Big)^{m}, \qquad t = 0,\ldots,n-1,

        the probability that rank :math:`t` wins Best-of-N on a size-:math:`m`
        with-replacement resample (:math:`\sum_t w_t = 1`). MoB returns

        .. math::

            \hat z_\alpha = \operatorname*{arg\,max}_z
            \sum_{t:\, z_{(t)} = z} w_t .

    Examples:
        >>> answers = ["A", "A", "A", "B", "B", "C"]
        >>> scores = [0.5, 0.55, 0.6, 0.9, 0.4, 0.3]
        >>> best_of_n(answers, scores)             # fooled by the lone high-reward B
        'B'
        >>> majority_of_the_bests(answers, scores)  # robust: bootstrap mode is A
        'A'
    """
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    if m is not None:
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
            raise ValueError(f"m must be a positive integer or None; got {m!r}.")
        if int(m) < 1:
            raise ValueError(f"m must be a positive integer or None; got {m!r}.")
    selected: list[Any] = []
    indices: list[int] = []
    sel_scores: list[float] = []
    for row, srow in zip(Z, S, strict=True):
        a, idx = _row_mob(row, srow, m)
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


def _row_best_of_majority(
    ans_row: Any, score_row: Any, alpha: float, aggregate: str
) -> tuple[Any, int]:
    """Highest-reward answer among the frequency-gated groups for one row."""
    part = _valid_indices(ans_row)
    if not part:
        return None, -1
    n = len(part)
    groups: dict[Any, dict[str, Any]] = {}
    for j in part:
        a = ans_row[j]
        g = groups.get(a)
        if g is None:
            groups[a] = {"idx": [j], "first": j, "rep": j}
        else:
            g["idx"].append(j)
            if float(score_row[j]) > float(score_row[g["rep"]]):
                g["rep"] = j

    gated = [a for a in groups if len(groups[a]["idx"]) / n >= alpha]
    if not gated:  # alpha too high -> relax the gate to keep the pool non-empty
        gated = list(groups)

    def reward(a: Any) -> float:
        ss = [float(score_row[j]) for j in groups[a]["idx"]]
        if aggregate == "mean":
            return sum(ss) / len(ss)
        if aggregate == "sum":
            return sum(ss)
        return max(ss)  # "max"

    best = min(gated, key=lambda a: (-reward(a), groups[a]["first"]))
    return best, groups[best]["rep"]


def best_of_majority(
    answers: Any,
    scores: Any,
    *,
    alpha: float = 0.0,
    aggregate: str = "mean",
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""
    Best-of-Majority (BoM): highest-reward answer among the frequent ones.

    Best-of-N is fooled by a single high-reward outlier; majority vote ignores
    reward quality. Best-of-Majority interpolates: it first keeps only answers
    whose empirical frequency is at least ``alpha`` (a *frequency gate* that
    discards rarely-produced answers a reward model is most likely to
    over-score), then, among those survivors, returns the one with the highest
    aggregated reward. This frequency-gated reward selection is minimax-optimal
    for Pass@k inference scaling; here it is used at ``k = 1`` to return a single
    answer.

    References:
        Di, Q., Ji, K., Li, X., Zhao, H., & Gu, Q. (2025). Best-of-Majority:
        Minimax-Optimal Strategy for Pass@k Inference Scaling.
        *arXiv:2510.03199* (Algorithm 3). https://arxiv.org/abs/2510.03199

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of answer labels. Unparsable
            entries (``None``, ``""``, ``NaN``) are ignored.
        scores: ``(N,)`` or ``(M, N)`` array-like of per-candidate scores
            (higher is better), same shape as ``answers``.
        alpha: Minimum empirical frequency :math:`\alpha \in [0, 1]` an answer
            must reach to be eligible. ``0`` (default) keeps every answer,
            matching :func:`weighted_majority_vote` when ``aggregate`` is
            ``"sum"`` or ``"mean"``. With ``aggregate="max"``, it selects the
            answer group whose best member has the highest score. A small
            positive value (e.g. enough to drop single-occurrence answers)
            enables the intended frequency gate. If the gate would empty the
            pool it is relaxed to keep all valid answers.
        aggregate: Per-answer reward pooling: ``"mean"`` (default, the paper's
            instantiation), ``"sum"``, or ``"max"``.
        return_index: If ``True``, also return the highest-scoring member of the
            winning answer group per question (``-1`` if none are valid).
        return_score: If ``True``, also return the score of that representative
            candidate (``NaN`` if none are valid).

    Returns:
        The selected answer per question (scalar or ``(M,)`` object array). When
        ``return_index`` and/or ``return_score`` are set, a tuple
        ``(selected[, index][, score])`` is returned in that order.

    Formula:
        For question :math:`q`, let the empirical frequency be
        :math:`\pi_q(z) = \tfrac{1}{n}\sum_i \mathbb 1[Z_{q i} = z]`, the gated
        set be :math:`\mathcal E_q(\alpha) = \{z : \pi_q(z) \ge \alpha\}`, and
        :math:`\hat r_q(z)` be the mean, sum, or maximum of the group's scores.

        .. math::

            \hat z_q = \operatorname*{arg\,max}_{z \in \mathcal E_q(\alpha)}
            \hat r_q(z),

        with ties broken by earliest appearance.

    Examples:
        >>> answers = ["A", "A", "A", "B"]
        >>> scores = [0.5, 0.55, 0.6, 0.99]
        >>> best_of_n(answers, scores)                       # lone high-reward B
        'B'
        >>> best_of_majority(answers, scores, alpha=0.5)     # B is too rare -> A
        'A'
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
    if aggregate not in ("mean", "sum", "max"):
        raise ValueError(
            f"aggregate must be 'mean', 'sum', or 'max'; got {aggregate!r}."
        )
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None
    selected: list[Any] = []
    indices: list[int] = []
    sel_scores: list[float] = []
    for row, srow in zip(Z, S, strict=True):
        a, idx = _row_best_of_majority(row, srow, alpha, aggregate)
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


mob = majority_of_the_bests
