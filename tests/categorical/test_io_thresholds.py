"""Tests for C1/C3/P3 column extraction in io.py and signal mapping in thresholds.py.

The actual .jsonl.gz format stores pre-computed logprob statistics directly in
the `tokens` dict (completion_avg_logprob, completion_sum_logprob, prompt_sum_logprob),
so io._extract_row reads them directly rather than deriving them from the raw lists.
"""

import json
import math

import numpy as np
import pytest

from scorio.categorical import load_records
from scorio.categorical.thresholds import Thresholds

# ── known values (match what the inference framework pre-computes) ────

_COMPLETION_AVG = -0.3     # C1: mean log P over all completion tokens
_COMPLETION_SUM = -1.5     # C3: sum  log P over all completion tokens
_PROMPT_SUM     = -4.0     # P3: sum  log P over all prompt tokens
_COMPLETION_LP  = [-0.5, -0.2, -0.1, -0.3, -0.4]   # raw list (for existing stats)

_FULL_RECORD = {
    "seed": 1,
    "data_id": 1,
    "model": "test/m",
    "output": {"finish_reason": "stop", "num_completion_tokens": 5},
    "tokens": {
        "completion_avg_logprob": _COMPLETION_AVG,
        "completion_sum_logprob": _COMPLETION_SUM,
        "completion_ppl": 1.09,
        "prompt_sum_logprob": _PROMPT_SUM,
        "prompt_ppl": 70.0,
        "completion_logprob_list": _COMPLETION_LP,
    },
    "processed_results": {"is_correct": 1, "has_box": 0},
}

_NO_PROMPT_SUM_RECORD = {
    "seed": 2,
    "data_id": 2,
    "model": "test/m",
    "output": {"finish_reason": "stop", "num_completion_tokens": 5},
    "tokens": {
        "completion_avg_logprob": _COMPLETION_AVG,
        "completion_sum_logprob": _COMPLETION_SUM,
        "completion_ppl": 1.09,
        # prompt_sum_logprob intentionally absent
        "prompt_ppl": 70.0,
        "completion_logprob_list": _COMPLETION_LP,
    },
    "processed_results": {"is_correct": 0, "has_box": 1},
}


# ── shared fixtures ───────────────────────────────────────────────────


def _first_row(columns) -> dict:
    """Materialise row 0 of a columns dict as a plain ``{col: value}`` dict."""
    return {k: v[0] for k, v in columns.items()}


@pytest.fixture
def loaded_row(tmp_path):
    (tmp_path / "out.jsonl").write_text(json.dumps(_FULL_RECORD) + "\n")
    return _first_row(load_records(tmp_path))


@pytest.fixture
def loaded_row_no_prompt(tmp_path):
    (tmp_path / "out.jsonl").write_text(json.dumps(_NO_PROMPT_SUM_RECORD) + "\n")
    return _first_row(load_records(tmp_path))


# ── io column value tests ─────────────────────────────────────────────


def test_completion_avg_logprob_value(loaded_row):
    assert abs(loaded_row["completion_avg_logprob"] - _COMPLETION_AVG) < 1e-9


def test_completion_sum_logprob_value(loaded_row):
    assert abs(loaded_row["completion_sum_logprob"] - _COMPLETION_SUM) < 1e-9


def test_prompt_sum_logprob_value(loaded_row):
    assert abs(loaded_row["prompt_sum_logprob"] - _PROMPT_SUM) < 1e-9


def test_avg_equals_sum_over_n(loaded_row):
    n = len(_COMPLETION_LP)
    assert abs(
        loaded_row["completion_avg_logprob"]
        - loaded_row["completion_sum_logprob"] / n
    ) < 1e-9


def test_prompt_sum_nan_when_absent(loaded_row_no_prompt):
    val = loaded_row_no_prompt["prompt_sum_logprob"]
    assert val is None or (isinstance(val, float) and math.isnan(val))


# ── Thresholds signal-mapping tests ──────────────────────────────────


def _make_threshold_columns(n: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "completion_avg_logprob": rng.uniform(-2.0, -0.1, n),
        "completion_sum_logprob": rng.uniform(-500.0, -10.0, n),
        "prompt_sum_logprob":     rng.uniform(-300.0, -5.0, n),
    }


def test_thresholds_c1_median_populated():
    cols = _make_threshold_columns()
    t = Thresholds.from_columns(cols)
    assert "C1" in t.medians
    assert abs(t.medians["C1"] - float(np.median(cols["completion_avg_logprob"]))) < 1e-9


def test_thresholds_c3_median_populated():
    cols = _make_threshold_columns()
    t = Thresholds.from_columns(cols)
    assert "C3" in t.medians
    assert abs(t.medians["C3"] - float(np.median(cols["completion_sum_logprob"]))) < 1e-9


def test_thresholds_p3_median_populated():
    cols = _make_threshold_columns()
    t = Thresholds.from_columns(cols)
    assert "P3" in t.medians
    assert abs(t.medians["P3"] - float(np.median(cols["prompt_sum_logprob"]))) < 1e-9


def test_thresholds_c1_not_populated_when_column_absent():
    cols = {"completion_sum_logprob": np.array([-100.0, -200.0])}
    t = Thresholds.from_columns(cols)
    assert "C1" not in t.medians


def test_thresholds_c3_not_populated_when_column_absent():
    cols = {"completion_avg_logprob": np.array([-1.0, -2.0])}
    t = Thresholds.from_columns(cols)
    assert "C3" not in t.medians


def test_thresholds_p3_not_populated_when_column_absent():
    cols = {"completion_avg_logprob": np.array([-1.0, -2.0])}
    t = Thresholds.from_columns(cols)
    assert "P3" not in t.medians
