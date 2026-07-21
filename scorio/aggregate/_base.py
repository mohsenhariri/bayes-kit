r"""Shared input handling for :mod:`scorio.aggregate` selection rules.

All aggregation rules consume a candidate pool for each question: an answer
matrix ``Z`` and (for reward-aware rules) a score matrix ``S`` of the same
shape. This module normalizes those inputs to a common ``(M, N)`` layout,
decides which entries are *valid* answers, and finalizes the per-question
selections back into the caller-facing shape (a scalar for a single question,
an object array for a batch).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "_is_valid",
    "_normalize",
    "_finalize",
    "_finalize_index",
    "_finalize_score",
    "_pack",
    "_default_m",
    "_valid_indices",
    "_plurality",
    "_keep_count",
]


def _is_valid(a: Any) -> bool:
    """Whether ``a`` is a usable answer label.

    ``None``, the empty string, and ``NaN`` denote an unparsable / missing
    answer and are excluded from every selection rule (they can neither be
    voted for nor submitted as a final answer).
    """
    if a is None:
        return False
    if isinstance(a, str):
        return a != ""
    return not (isinstance(a, float) and math.isnan(a))


def _normalize(
    answers: Any,
    scores: Any = None,
    *,
    require_scores: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    r"""Coerce ``answers``/``scores`` to ``(M, N)`` and flag single-question input.

    Args:
        answers: ``(N,)`` or ``(M, N)`` array-like of hashable answer labels.
        scores: ``(N,)`` or ``(M, N)`` array-like of floats, or ``None``.
        require_scores: If ``True``, raise when ``scores`` is ``None``.

    Returns:
        ``(Z, S, single)`` where ``Z`` is an ``(M, N)`` object array, ``S`` is an
        ``(M, N)`` float array (or ``None``), and ``single`` is ``True`` when the
        input was one-dimensional (one question).
    """
    Z = np.asarray(answers, dtype=object)
    if Z.ndim == 1:
        Z = Z.reshape(1, -1)
        single = True
    elif Z.ndim == 2:
        single = False
    else:
        raise ValueError("answers must be a 1D (N,) or 2D (M, N) array.")

    _, N = Z.shape
    if N == 0:
        raise ValueError("need at least one candidate per question (N >= 1).")

    if scores is None:
        if require_scores:
            raise ValueError("scores are required for this selection rule.")
        return Z, None, single

    S = np.asarray(scores, dtype=float)
    if S.ndim == 1:
        S = S.reshape(1, -1)
    if S.shape != Z.shape:
        raise ValueError(
            f"answers and scores must have the same shape; got {Z.shape} and {S.shape}."
        )
    return Z, S, single


def _finalize(selected: list[Any], single: bool) -> Any:
    """Return a scalar for a single question, else an ``(M,)`` object array."""
    if single:
        return selected[0]
    return np.array(selected, dtype=object)


def _finalize_index(indices: list[int], single: bool) -> Any:
    """Return an int for a single question, else an ``(M,)`` int array.

    Rows with no valid answer use the sentinel index ``-1``.
    """
    if single:
        return int(indices[0])
    return np.array(indices, dtype=int)


def _finalize_score(scores: list[float], single: bool) -> Any:
    """Return a float for a single question, else an ``(M,)`` float array.

    Rows with no valid answer carry ``NaN`` (matching the ``-1`` index sentinel).
    """
    if single:
        return float(scores[0])
    return np.array(scores, dtype=float)


def _pack(
    selected: list[Any],
    indices: list[int],
    sel_scores: list[float],
    single: bool,
    *,
    return_index: bool,
    return_score: bool,
) -> Any:
    """Assemble the caller-facing return value from the requested pieces.

    Always yields the selection first, then the index (if ``return_index``) and
    the associated score (if ``return_score``), in that fixed order. A bare
    selection is returned when no extras are requested.
    """
    out: list[Any] = [_finalize(selected, single)]
    if return_index:
        out.append(_finalize_index(indices, single))
    if return_score:
        out.append(_finalize_score(sel_scores, single))
    return out[0] if len(out) == 1 else tuple(out)


def _default_m(n: int) -> int:
    r"""Default Majority-of-the-Bests resample size :math:`m = \lfloor\sqrt{n}\rfloor`."""
    return max(1, math.isqrt(n))


def _valid_indices(ans_row: Any) -> list[int]:
    """Indices of the candidates in ``ans_row`` whose answer is usable."""
    return [j for j, a in enumerate(ans_row) if _is_valid(a)]


def _plurality(
    ans_row: Any,
    part: list[int],
    weight_of: Any,
    score_of: Any,
) -> tuple[Any, int]:
    """Answer group with the largest total vote weight over candidate indices ``part``.

    This is the shared kernel of the score-aware voting rules: each candidate
    ``j`` in ``part`` casts a vote of weight ``weight_of(j)`` (which may be
    negative) for its answer ``ans_row[j]``; the answer whose votes sum highest
    wins. ``score_of(j)`` returns candidate ``j``'s raw score and is used only to
    pick the winning group's *representative* -- its highest-scoring member, ties
    broken by lowest index. Group-weight ties are broken by earliest appearance,
    matching the rest of :mod:`scorio.aggregate`.

    Args:
        ans_row: sequence of answer labels for one question.
        part: candidate indices that participate in the vote (already filtered to
            valid, in-window candidates). Empty yields the ``(None, -1)`` sentinel.
        weight_of: callable ``j -> float`` giving candidate ``j``'s vote weight.
        score_of: callable ``j -> float`` giving candidate ``j``'s raw score.

    Returns:
        ``(winning_answer, representative_index)``, or ``(None, -1)`` if ``part``
        is empty.
    """
    if not part:
        return None, -1
    # Accumulate from an integer zero so that integer weights (e.g. exact Borda
    # weights) sum exactly -- Python ints do not overflow and detect genuine
    # ties without float noise; float weights promote the running sum as usual.
    total: dict[Any, float] = {}
    first: dict[Any, int] = {}
    rep: dict[Any, int] = {}
    for j in part:
        a = ans_row[j]
        if a not in total:
            total[a] = 0
            first[a] = j
            rep[a] = j
        total[a] += weight_of(j)
        if score_of(j) > score_of(rep[a]):
            rep[a] = j
    best = min(total, key=lambda a: (-total[a], first[a]))
    return best, rep[best]


def _keep_count(keep: Any, n: int) -> int:
    """Resolve the ``keep`` filter setting to a candidate count in ``[1, n]``.

    A float in ``(0, 1]`` is a *fraction* -- keep the top
    :math:`\\lceil \\mathrm{keep}\\cdot n \\rceil` candidates (at least one). An
    integer ``>= 1`` is an explicit *count* -- keep the top ``min(keep, n)``.
    """
    if isinstance(keep, bool):
        raise ValueError("keep must be a float in (0, 1] or an int >= 1; got a bool.")
    if isinstance(keep, (int, np.integer)) and keep >= 1:
        return min(int(keep), n)
    if isinstance(keep, (float, np.floating)) and 0.0 < float(keep) <= 1.0:
        # Nudge down by a tiny tolerance so float error (e.g. 0.07 * 100 ==
        # 7.000000000000001) does not round a whole fraction up an extra step.
        return max(1, math.ceil(float(keep) * n - 1e-9))
    raise ValueError(
        f"keep must be a float fraction in (0, 1] or an int count >= 1; got {keep!r}."
    )
