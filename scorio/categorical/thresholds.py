"""Per-signal thresholds computed from a pooled reference corpus.

Thresholds are computed once on the full corpus (via :meth:`Thresholds.from_columns`)
and then used by :mod:`scorio.categorical.schemas` to discretise continuous signal
values into levels before category assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scorio.categorical._util import Columns, is_missing, to_float

# ── Signal catalogue ─────────────────────────────────────────────────
#
# Maps symbolic signal IDs (used throughout schemas.py) to the exact
# column names produced by io.load_records().

SIGNAL_TO_COLUMN: dict[str, str] = {
    # ── binary outcome signals ────────────────────────────────────
    "R1": "is_correct",
    "R2": "has_box",
    "R3": "hit_max_len",
    # ── generation stats ──────────────────────────────────────────
    "C_len": "completion_length",
    "C_ppl": "completion_perplexity",
    "P_ppl": "prompt_perplexity",
    # ── token-level log-prob signals ──────────────────────────────
    "Lp_min": "logprob_min",
    "Lp_iqr": "logprob_iqr",
    # ── schemas.py (COLM-style) aliases ───────────────────────────
    "C1": "completion_avg_logprob",  # mean log P(token) over all completion tokens
    "T_lp_min": "logprob_min",  # same concept as Lp_min
    "P2": "prompt_perplexity",  # prompt ppl ≈ prompt perplexity
    "P3": "prompt_sum_logprob",  # Σ log P(token) over all prompt tokens
    "C3": "completion_sum_logprob",  # Σ log P(token) over all completion tokens
    "Lp_tail": "tail64_avg_logprob",
    # ── optional outcome reward models ────────────────────────────
    "O1": "acemath_orm",
    "O2": "skywork_orm",
    "W1": "verifier_pA",
    # ── optional PRM step-score derived features ──────────────────
    "V1_mean": "prm1_steps_mean",
    "V1_min": "prm1_steps_min",
    "V1_max": "prm1_steps_max",
    "V1_std": "prm1_steps_std",
    "V1_last": "prm1_steps_last",
    "V1_n": "prm1_steps_n_steps",
}

# Signals whose values are 0/1 flags; classified as "1"/"0" instead of
# "high"/"low", and whose threshold is fixed at 0.5 regardless of data.
BINARY_SIGNALS: frozenset[str] = frozenset({"R1", "R2", "R3"})


# ── Thresholds ───────────────────────────────────────────────────────


@dataclass
class Thresholds:
    """Per-signal thresholds computed from a reference columns dict.

    For continuous signals the median (and optionally quartiles, mean, std)
    are derived from the pooled corpus.  For binary signals the threshold
    is fixed at 0.5.
    """

    medians: dict[str, float] = field(default_factory=dict)
    q25: dict[str, float] = field(default_factory=dict)
    q75: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_columns(cls, columns: Columns) -> Thresholds:
        """Compute thresholds from a pooled corpus columns dict.

        Args:
            columns: ``dict[str, np.ndarray]`` returned by
                :func:`scorio.categorical.io.load_records`.

        Returns:
            A :class:`Thresholds` instance populated for every signal whose
            column is present in *columns*.  Missing columns are silently skipped.
        """
        t = cls()
        for sig_id, col in SIGNAL_TO_COLUMN.items():
            if col not in columns:
                continue
            if sig_id in BINARY_SIGNALS:
                t.medians[sig_id] = 0.5
                continue
            arr = np.asarray(columns[col])
            if arr.dtype != np.float64:
                arr = np.array([to_float(v) for v in arr], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            t.medians[sig_id] = float(np.median(arr))
            t.q25[sig_id] = float(np.percentile(arr, 25))
            t.q75[sig_id] = float(np.percentile(arr, 75))
            t.means[sig_id] = float(np.mean(arr))
            # ddof=1 to match pandas Series.std() (sample standard deviation).
            t.stds[sig_id] = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
        return t


# ── Signal extraction helpers ────────────────────────────────────────


def _get_signal_value(columns: Columns, idx: int, signal_id: str) -> float | None:
    """Extract the numeric value for a signal from row *idx* of a columns dict."""
    col = SIGNAL_TO_COLUMN.get(signal_id)
    if col is None or col not in columns:
        return None
    v = to_float(columns[col][idx])
    if is_missing(v):
        return None
    return v


def _classify_signal(signal_id: str, value: float, thresholds: Thresholds) -> str:
    """Classify a signal value as ``'high'``/``'low'`` or ``'1'``/``'0'``."""
    if signal_id in BINARY_SIGNALS:
        return "1" if value >= 0.5 else "0"
    med = thresholds.medians.get(signal_id)
    if med is None:
        return "high" if value >= 0.0 else "low"
    return "high" if value >= med else "low"


def _classify_signal_tertile(
    signal_id: str, value: float, thresholds: Thresholds
) -> str:
    """Classify into ``'high'``/``'mid'``/``'low'`` using quartile boundaries."""
    if signal_id in BINARY_SIGNALS:
        return "1" if value >= 0.5 else "0"
    q25 = thresholds.q25.get(signal_id)
    q75 = thresholds.q75.get(signal_id)
    if q25 is None or q75 is None:
        return _classify_signal(signal_id, value, thresholds)
    if value >= q75:
        return "high"
    elif value <= q25:
        return "low"
    return "mid"


__all__ = [
    "SIGNAL_TO_COLUMN",
    "BINARY_SIGNALS",
    "Thresholds",
    "_get_signal_value",
    "_classify_signal",
    "_classify_signal_tertile",
]
