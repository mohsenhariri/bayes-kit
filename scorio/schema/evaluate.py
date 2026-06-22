"""Orchestration layer: io → thresholds → schemas → eval.bayes.

Pipeline
--------
1. Load JSONL signals into a DataFrame (:func:`scorio.schema.io.load_records`).
2. Compute per-signal thresholds from the pooled corpus
   (:class:`scorio.schema.thresholds.Thresholds`).
3. For each criterion in :data:`scorio.schema.schemas._CRITERION_REGISTRY`:
   a. Classify every row's signals into levels and apply the criterion's
      ``classify_fn`` to obtain a float score per completion.
   b. Pivot (problem × trial) into an integer R matrix; derive weight
      vector *w* from the unique score values.
   c. Call :func:`scorio.eval.bayes` → ``(mu, sigma)``.
4. Return results grouped by model (or any other column).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scorio import eval as scorio_eval
from scorio.schema.io import load_records
from scorio.schema.schemas import _CRITERION_REGISTRY
from scorio.schema.thresholds import Thresholds, _classify_signal, _get_signal_value

logger = logging.getLogger(__name__)

# ── internal helpers ─────────────────────────────────────────────────


def _score_rows(
    df: pd.DataFrame,
    crit_entry: dict,
    thresholds: Thresholds,
) -> pd.DataFrame:
    """Apply a criterion's classify_fn to every row in *df*.

    Args:
        df: Subset of the signals DataFrame (e.g. one model's rows).
        crit_entry: Entry from ``_CRITERION_REGISTRY``.
        thresholds: Pre-computed thresholds for level classification.

    Returns:
        DataFrame with columns ``[model, problem, trial, score]``
        where *score* is the float returned by the criterion's classify_fn.
    """
    signals = crit_entry["signals"]
    classify_fn = crit_entry["classify"]

    records = []
    for _, row in df.iterrows():
        lvl: dict[str, str] = {}
        val: dict[str, float | None] = {}
        for sig_id in signals:
            v = _get_signal_value(row, sig_id)
            val[sig_id] = v
            if v is not None:
                lvl[sig_id] = _classify_signal(sig_id, v, thresholds)

        *_, score = classify_fn(lvl, val, thresholds)
        records.append({
            "model":   row.get("model"),
            "problem": row.get("problem"),
            "trial":   row.get("trial", 0),
            "score":   float(score),
        })

    return pd.DataFrame(records)


def _scores_to_R(scored_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pivot a scored DataFrame to an integer R matrix.

    Args:
        scored_df: DataFrame with columns ``[problem, trial, score]``.

    Returns:
        ``(R_int, w)`` where:

        * ``R_int`` is an M × N integer array (rows = problems, cols = trials)
          with each entry mapping to a category index.
        * ``w`` is the weight vector — the sorted unique float scores — for use
          with :func:`scorio.eval.bayes`.
    """
    pivot = scored_df.pivot_table(
        index="problem",
        columns="trial",
        values="score",
        aggfunc="first",
    )
    # Fill missing (problem, trial) cells with the lowest possible score (0.0
    # fallback → category 0) rather than dropping problems with uneven trial counts.
    pivot = pivot.fillna(0.0)

    flat = pivot.values.flatten()
    unique_scores = sorted(set(flat))
    score_to_cat = {s: i for i, s in enumerate(unique_scores)}
    w = np.array(unique_scores, dtype=float)

    cat_fn = np.vectorize(lambda x: score_to_cat[x])
    R_int = cat_fn(pivot.values).astype(int)

    return R_int, w


# ── public API ───────────────────────────────────────────────────────


def evaluate_criterion(
    signals: pd.DataFrame | str | Path,
    criterion_id: str,
    thresholds: Thresholds | None = None,
    group_key: str = "model",
) -> dict[str, tuple[float, float]]:
    """Evaluate one criterion and return Bayes@N results per group.

    Args:
        signals: DataFrame from :func:`~scorio.schema.io.load_records`, or a
                 path to a directory of ``.jsonl`` files.
        criterion_id: Key in ``_CRITERION_REGISTRY`` (e.g. ``"2.5"``).
        thresholds: Pre-computed thresholds. If ``None``, computed from *signals*.
        group_key: Column to group by (default ``"model"``).

    Returns:
        ``{group_value: (mu, sigma)}`` from :func:`scorio.eval.bayes`.

    Raises:
        KeyError: If *criterion_id* is not registered.
        ValueError: If *signals* contains no rows or no ``.jsonl`` files are found.
    """
    if criterion_id not in _CRITERION_REGISTRY:
        raise KeyError(
            f"Unknown criterion {criterion_id!r}. "
            f"Registered: {sorted(_CRITERION_REGISTRY)}"
        )
    crit_entry = _CRITERION_REGISTRY[criterion_id]

    if not isinstance(signals, pd.DataFrame):
        signals = load_records(signals)
    df = signals

    if thresholds is None:
        logger.info(
            "criterion=%s: computing thresholds from %d rows",
            criterion_id, len(df),
        )
        thresholds = Thresholds.from_dataframe(df)

    results: dict[str, tuple[float, float]] = {}

    for group_val, group_df in df.groupby(group_key):
        label = str(group_val)
        try:
            scored = _score_rows(group_df, crit_entry, thresholds)
            R_int, w = _scores_to_R(scored)

            if R_int.shape[0] == 0:
                logger.warning(
                    "criterion=%s %s=%s: empty R matrix — skipping",
                    criterion_id, group_key, label,
                )
                continue

            mu, sigma = scorio_eval.bayes(R_int, w)
            results[label] = (mu, sigma)
            logger.debug(
                "criterion=%s %s=%s: R%s w=%s → bayes=(%.4f ± %.4f)",
                criterion_id, group_key, label,
                R_int.shape, w.tolist(), mu, sigma,
            )

        except Exception as exc:
            logger.error(
                "criterion=%s %s=%s: failed — %s",
                criterion_id, group_key, label, exc, exc_info=True,
            )

    logger.info(
        "criterion=%s: evaluated %d group(s)",
        criterion_id, len(results),
    )
    return results


def evaluate_all(
    signals: pd.DataFrame | str | Path,
    criterion_ids: list[str] | None = None,
    thresholds: Thresholds | None = None,
    group_key: str = "model",
) -> dict[str, dict[str, tuple[float, float]]]:
    """Evaluate multiple criteria and return results for all.

    Thresholds are computed once from the pooled data and reused across
    every criterion, so calling this is more efficient than calling
    :func:`evaluate_criterion` in a loop.

    Args:
        signals: DataFrame or path to ``.jsonl`` directory.
        criterion_ids: Subset of ``_CRITERION_REGISTRY`` keys to evaluate.
                       ``None`` → evaluate all registered criteria.
        thresholds: Pre-computed thresholds. If ``None``, computed once here.
        group_key: Column to group by (default ``"model"``).

    Returns:
        ``{criterion_id: {model: (mu, sigma)}}``
    """
    if not isinstance(signals, pd.DataFrame):
        signals = load_records(signals)
    df = signals

    ids = criterion_ids if criterion_ids is not None else list(_CRITERION_REGISTRY)

    if thresholds is None:
        logger.info(
            "evaluate_all: computing thresholds from %d rows (reused for %d criterion/a)",
            len(df), len(ids),
        )
        thresholds = Thresholds.from_dataframe(df)

    return {
        cid: evaluate_criterion(df, cid, thresholds=thresholds, group_key=group_key)
        for cid in ids
    }


__all__ = ["evaluate_criterion", "evaluate_all"]
