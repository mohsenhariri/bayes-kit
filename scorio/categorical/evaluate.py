"""Orchestration layer: io → thresholds → schemas → eval.bayes.

Pipeline
--------
1. Load JSONL signals into a columns dict (:func:`scorio.categorical.io.load_records`).
2. Compute per-signal thresholds from the pooled corpus
   (:class:`scorio.categorical.thresholds.Thresholds`).
3. For each schema in :data:`scorio.categorical.schemas._SCHEMA_REGISTRY`:
   a. Classify every row's signals into levels and apply the schema's
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

from scorio import eval as scorio_eval
from scorio.categorical._util import Columns, num_rows, select_rows
from scorio.categorical.io import load_records
from scorio.categorical.schemas import _SCHEMA_REGISTRY
from scorio.categorical.thresholds import (
    Thresholds,
    _classify_signal,
    _get_signal_value,
)

logger = logging.getLogger(__name__)

# ── internal helpers ─────────────────────────────────────────────────


def _group_by(columns: Columns, key: str) -> list[tuple]:
    """Group rows by the value of column *key*.

    Returns a list of ``(group_value, group_columns)`` pairs sorted by group
    value, mirroring the deterministic ordering of ``DataFrame.groupby``.
    """
    key_col = columns[key]
    groups: dict = {}
    for i, v in enumerate(key_col):
        groups.setdefault(v, []).append(i)
    return [(v, select_rows(columns, groups[v])) for v in sorted(groups, key=str)]


def _score_rows(
    columns: Columns,
    schema_entry: dict,
    thresholds: Thresholds,
) -> list[tuple]:
    """Apply a schema's classify_fn to every row in *columns*.

    Args:
        columns: Subset of the signals columns dict (e.g. one model's rows).
        schema_entry: Entry from ``_SCHEMA_REGISTRY``.
        thresholds: Pre-computed thresholds for level classification.

    Returns:
        A list of ``(problem, trial, score)`` tuples where *score* is the float
        returned by the schema's classify_fn.
    """
    signals = schema_entry["signals"]
    classify_fn = schema_entry["classify"]

    problem_col = columns.get("problem")
    trial_col = columns.get("trial")

    scored: list[tuple] = []
    for i in range(num_rows(columns)):
        lvl: dict[str, str] = {}
        val: dict[str, float | None] = {}
        for sig_id in signals:
            v = _get_signal_value(columns, i, sig_id)
            val[sig_id] = v
            if v is not None:
                lvl[sig_id] = _classify_signal(sig_id, v, thresholds)

        *_, score = classify_fn(lvl, val, thresholds)
        problem = problem_col[i] if problem_col is not None else None
        trial = trial_col[i] if trial_col is not None else 0
        scored.append((problem, trial, float(score)))

    return scored


def _scores_to_R(scored: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Pivot scored ``(problem, trial, score)`` tuples to an integer R matrix.

    Returns:
        ``(R_int, w)`` where:

        * ``R_int`` is an M × N integer array (rows = problems, cols = trials)
          with each entry mapping to a category index.
        * ``w`` is the weight vector — the sorted unique float scores — for use
          with :func:`scorio.eval.bayes`.
    """
    problems = sorted({p for p, _, _ in scored})
    trials = sorted({t for _, t, _ in scored})
    p_idx = {p: i for i, p in enumerate(problems)}
    t_idx = {t: j for j, t in enumerate(trials)}

    # Fill missing (problem, trial) cells with the lowest possible score (0.0
    # fallback → category 0) rather than dropping problems with uneven trial counts.
    mat = np.zeros((len(problems), len(trials)), dtype=float)
    filled = np.zeros(mat.shape, dtype=bool)
    for p, t, s in scored:
        i, j = p_idx[p], t_idx[t]
        if not filled[i, j]:  # "first" aggregation: keep the earliest value
            mat[i, j] = s
            filled[i, j] = True

    unique_scores = sorted(set(mat.flatten().tolist()))
    score_to_cat = {s: i for i, s in enumerate(unique_scores)}
    w = np.array(unique_scores, dtype=float)

    R_int = np.array(
        [[score_to_cat[v] for v in row] for row in mat], dtype=int
    ).reshape(mat.shape)

    return R_int, w


# ── public API ───────────────────────────────────────────────────────


def evaluate_schema(
    signals: Columns | str | Path,
    schema_id: str,
    thresholds: Thresholds | None = None,
    group_key: str = "model",
) -> dict[str, tuple[float, float]]:
    """Evaluate one schema and return Bayes@N results per group.

    Args:
        signals: Columns dict from :func:`~scorio.categorical.io.load_records`, or
                 a path to a directory of ``.jsonl`` files.
        schema_id: Key in ``_SCHEMA_REGISTRY`` (e.g. ``"2.5"``).
        thresholds: Pre-computed thresholds. If ``None``, computed from *signals*.
        group_key: Column to group by (default ``"model"``).

    Returns:
        ``{group_value: (mu, sigma)}`` from :func:`scorio.eval.bayes`.

    Raises:
        KeyError: If *schema_id* is not registered.
        ValueError: If *signals* contains no rows or no ``.jsonl`` files are found.
    """
    if schema_id not in _SCHEMA_REGISTRY:
        raise KeyError(
            f"Unknown schema {schema_id!r}. Registered: {sorted(_SCHEMA_REGISTRY)}"
        )
    schema_entry = _SCHEMA_REGISTRY[schema_id]

    if not isinstance(signals, dict):
        signals = load_records(signals)
    columns = signals

    if thresholds is None:
        logger.info(
            "schema=%s: computing thresholds from %d rows",
            schema_id,
            num_rows(columns),
        )
        thresholds = Thresholds.from_columns(columns)

    results: dict[str, tuple[float, float]] = {}

    for group_val, group_columns in _group_by(columns, group_key):
        label = str(group_val)
        try:
            scored = _score_rows(group_columns, schema_entry, thresholds)
            R_int, w = _scores_to_R(scored)

            if R_int.shape[0] == 0:
                logger.warning(
                    "schema=%s %s=%s: empty R matrix — skipping",
                    schema_id,
                    group_key,
                    label,
                )
                continue

            mu, sigma = scorio_eval.bayes(R_int, w)
            results[label] = (mu, sigma)
            logger.debug(
                "schema=%s %s=%s: R%s w=%s → bayes=(%.4f ± %.4f)",
                schema_id,
                group_key,
                label,
                R_int.shape,
                w.tolist(),
                mu,
                sigma,
            )

        except Exception as exc:
            logger.error(
                "schema=%s %s=%s: failed — %s",
                schema_id,
                group_key,
                label,
                exc,
                exc_info=True,
            )

    logger.info(
        "schema=%s: evaluated %d group(s)",
        schema_id,
        len(results),
    )
    return results


def evaluate_all(
    signals: Columns | str | Path,
    schema_ids: list[str] | None = None,
    thresholds: Thresholds | None = None,
    group_key: str = "model",
) -> dict[str, dict[str, tuple[float, float]]]:
    """Evaluate multiple schemas and return results for all.

    Thresholds are computed once from the pooled data and reused across
    every schema, so calling this is more efficient than calling
    :func:`evaluate_schema` in a loop.

    Args:
        signals: Columns dict or path to ``.jsonl`` directory.
        schema_ids: Subset of ``_SCHEMA_REGISTRY`` keys to evaluate.
                    ``None`` → evaluate all registered schemas.
        thresholds: Pre-computed thresholds. If ``None``, computed once here.
        group_key: Column to group by (default ``"model"``).

    Returns:
        ``{schema_id: {model: (mu, sigma)}}``
    """
    if not isinstance(signals, dict):
        signals = load_records(signals)
    columns = signals

    ids = schema_ids if schema_ids is not None else list(_SCHEMA_REGISTRY)

    if thresholds is None:
        logger.info(
            "evaluate_all: computing thresholds from %d rows (reused for %d schema/s)",
            num_rows(columns),
            len(ids),
        )
        thresholds = Thresholds.from_columns(columns)

    return {
        sid: evaluate_schema(columns, sid, thresholds=thresholds, group_key=group_key)
        for sid in ids
    }


__all__ = ["evaluate_schema", "evaluate_all"]
