r"""Confidence-Guided Early Stopping (CGES).

CGES scores each observed answer and an ``OTHER`` bucket for a correct answer
that has not appeared yet. The scores can be used to select a final answer or
to stop sampling once one hypothesis crosses a threshold.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import logsumexp

from ._base import _normalize, _pack, _valid_indices

__all__ = [
    "CGES_OTHER",
    "cges_vote",
    "cges_stop",
]


class _CGESOther:
    """Identity sentinel for the unseen-answer hypothesis."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "CGES_OTHER"

    def __reduce__(self) -> tuple[Any, tuple[()]]:
        return _restore_cges_other, ()


def _restore_cges_other() -> _CGESOther:
    return CGES_OTHER


CGES_OTHER = _CGESOther()
"""Bucket for a correct answer that has not been observed."""


def _validate_bool(value: Any, name: str) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a bool; got {value!r}.")


def _row_cges_posterior(ans_row: Any, score_row: Any) -> dict[Any, float]:
    """Compute CGES scores for one question."""
    part = _valid_indices(ans_row)
    if not part:
        return {CGES_OTHER: 1.0}

    concrete: list[Any] = []
    seen: set[Any] = set()
    for j in part:
        answer = ans_row[j]
        if answer is CGES_OTHER:
            raise ValueError("CGES_OTHER is reserved and cannot be an observed answer.")
        if answer not in seen:
            seen.add(answer)
            concrete.append(answer)

    # The support consists of the observed answers and OTHER.
    support_size = len(concrete) + 1
    mismatch_log_likelihood: dict[int, float] = {}
    base = 0.0
    for j in part:
        confidence = float(score_row[j])
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError(
                "CGES requires every valid candidate score to be finite and "
                f"strictly in (0, 1); got {confidence}."
            )
        mismatch = math.log1p(-confidence) - math.log(support_size - 1)
        mismatch_log_likelihood[j] = mismatch
        base += mismatch

    log_scores: dict[Any, float] = {answer: base for answer in concrete}
    log_scores[CGES_OTHER] = base
    for j in part:
        answer = ans_row[j]
        confidence = float(score_row[j])
        log_scores[answer] += math.log(confidence) - mismatch_log_likelihood[j]

    normalizer = float(logsumexp(list(log_scores.values())))
    return {
        answer: math.exp(log_score - normalizer)
        for answer, log_score in log_scores.items()
    }


def cges_vote(
    answers: Any,
    scores: Any,
    *,
    allow_other: bool = False,
    return_index: bool = False,
    return_score: bool = False,
) -> Any:
    r"""Select the answer with the largest CGES score.

    The default selects among answers that have been observed. Set
    ``allow_other=True`` to include :data:`CGES_OTHER` in the argmax, as in
    Algorithm 1 of the paper. Ties follow first observation order.

    References:
        Aghazadeh, E., Ghasemi, A., Beyhaghi, H., & Pishro-Nik, H. (2026).
        CGES: Confidence-Guided Early Stopping for Efficient and Accurate
        Self-Consistency. *arXiv:2511.02603v2* (Algorithm 1 and Remark D.4).
        https://arxiv.org/abs/2511.02603

    Args:
        answers: ``(N,)`` or ``(M, N)`` sampled answer labels. Unparsable
            entries are ignored.
        scores: Aligned confidence values, finite and strictly in ``(0, 1)``
            for every valid answer.
        allow_other: Include :data:`CGES_OTHER` in the final selection.
        return_index: Also return the highest-scoring candidate index for the
            selected observed answer. ``CGES_OTHER`` and empty rows use ``-1``.
        return_score: Also return that candidate's input score.
            ``CGES_OTHER`` and empty rows use ``NaN``.

    Returns:
        The selected answer, or an ``(M,)`` object array for batched input.
        Requested extras follow as ``(selected[, index][, score])``. A row
        with no valid answer returns ``None``.

    Formula:
        For support size :math:`K` and observations :math:`(R_t,C_t)`, CGES
        assigns hypothesis :math:`a` the unnormalized score

        .. math::

            A(a) = \prod_t C_t^{\mathbb{1}[R_t=a]}
                   \left(\frac{1-C_t}{K-1}\right)^{\mathbb{1}[R_t\ne a]}.

        The scores are normalized over the observed answers and
        :data:`CGES_OTHER`. The function returns their argmax, restricted to
        observed answers unless ``allow_other=True``.

    Examples:
        >>> cges_vote(["A", "A", "B"], [0.8, 0.7, 0.6])
        'A'
        >>> cges_vote(["A"], [0.1], allow_other=True) is CGES_OTHER
        True
    """
    _validate_bool(allow_other, "allow_other")
    Z, S, single = _normalize(answers, scores, require_scores=True)
    assert S is not None

    selected: list[Any] = []
    indices: list[int] = []
    selected_scores: list[float] = []
    for row, score_row in zip(Z, S, strict=True):
        part = _valid_indices(row)
        if not part:
            selected.append(None)
            indices.append(-1)
            selected_scores.append(float("nan"))
            continue

        posterior = _row_cges_posterior(row, score_row)
        hypotheses = list(posterior)
        if not allow_other:
            hypotheses = [answer for answer in hypotheses if answer is not CGES_OTHER]
        winner = max(hypotheses, key=posterior.__getitem__)
        selected.append(winner)

        if winner is CGES_OTHER:
            indices.append(-1)
            selected_scores.append(float("nan"))
            continue

        members = [j for j in part if row[j] == winner]
        index = min(members, key=lambda j: (-float(score_row[j]), j))
        indices.append(index)
        selected_scores.append(float(score_row[index]))

    return _pack(
        selected,
        indices,
        selected_scores,
        single,
        return_index=return_index,
        return_score=return_score,
    )


def cges_stop(
    answers: Any,
    scores: Any,
    *,
    threshold: float = 0.95,
    include_other: bool = False,
    min_samples: int = 1,
    return_prob: bool = False,
) -> Any:
    r"""Check whether a CGES sampling stream has reached its threshold.

    The input is the sequence sampled so far. By default, only observed answers
    can trigger stopping. Set ``include_other=True`` for the stopping rule in
    Algorithm 1 of the paper.

    References:
        Aghazadeh, E., Ghasemi, A., Beyhaghi, H., & Pishro-Nik, H. (2026).
        CGES: Confidence-Guided Early Stopping for Efficient and Accurate
        Self-Consistency. *arXiv:2511.02603v2* (Algorithm 1 and Remark D.4).
        https://arxiv.org/abs/2511.02603

    Args:
        answers: One ``(N,)`` answer sequence in sampling order.
        scores: Aligned confidence values, finite and strictly in ``(0, 1)``
            for every valid answer.
        threshold: Stopping threshold in ``(0, 1)``.
        include_other: Include :data:`CGES_OTHER` when checking the threshold.
        min_samples: Minimum number of valid samples required to stop.
        return_prob: Also return the largest score checked by the rule.

    Returns:
        A boolean, or ``(stop, probability)`` when ``return_prob=True``. A
        sequence with no valid answers returns ``False`` and probability
        ``0.0``.

    Examples:
        >>> cges_stop(["A"], [0.9], threshold=0.8)
        True
        >>> cges_stop(["A"], [0.9], threshold=0.8, min_samples=2)
        False
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1); got {threshold}.")
    _validate_bool(include_other, "include_other")
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)):
        raise ValueError(f"min_samples must be an integer >= 1; got {min_samples!r}.")
    if int(min_samples) < 1:
        raise ValueError(f"min_samples must be >= 1; got {min_samples}.")

    Z, S, single = _normalize(answers, scores, require_scores=True)
    if not single:
        raise ValueError("cges_stop expects one 1D sampling stream, not a batch.")
    assert S is not None
    part = _valid_indices(Z[0])
    if not part:
        result = (False, 0.0)
        return result if return_prob else result[0]

    posterior = _row_cges_posterior(Z[0], S[0])
    hypotheses = list(posterior)
    if not include_other:
        hypotheses = [answer for answer in hypotheses if answer is not CGES_OTHER]
    probability = max(posterior[answer] for answer in hypotheses)
    stop = len(part) >= int(min_samples) and probability >= threshold
    return (stop, probability) if return_prob else stop
