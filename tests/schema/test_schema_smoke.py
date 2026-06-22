import json

import numpy as np
import pandas as pd
import pytest

from scorio.schema import evaluate_all, load_records
from scorio.schema.schemas import _CRITERION_REGISTRY

# ── raw-format fixture ────────────────────────────────────────────────

_RAW_RECORD = {
    "seed": 1261,
    "data_id": 6,
    "model": "test/model",
    "task": "aime24",
    "output": {
        "finish_reason": "stop",
        "num_completion_tokens": 22818,
    },
    "tokens": {
        "completion_ppl": 1.094944594480259,
        "prompt_ppl": 70.38144192748423,
        "completion_logprob_list": [-0.5, -0.2, -0.1, -0.3, -0.4],
    },
    "processed_results": {
        "is_correct": 0,
        "has_box": 1,
    },
}


def test_load_records_raw_format(tmp_path):
    (tmp_path / "out.jsonl").write_text(json.dumps(_RAW_RECORD) + "\n")
    df = load_records(tmp_path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["problem"] == 6
    assert row["trial"] == 1261
    assert row["model"] == "test/model"
    assert row["is_correct"] == 0
    assert row["has_box"] == 1
    assert row["hit_max_len"] == 0
    assert row["completion_length"] == 22818
    assert abs(row["completion_perplexity"] - 1.094944594480259) < 1e-6
    assert abs(row["prompt_perplexity"] - 70.38144192748423) < 1e-6
    assert np.isfinite(row["logprob_min"])
    assert np.isfinite(row["logprob_iqr"])
    assert np.isfinite(row["tail64_avg_logprob"])


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(42)
    rows = []
    for model in ["model_A", "model_B"]:
        for problem in range(5):
            for trial in range(4):
                rows.append({
                    "model": model,
                    "problem": problem,
                    "trial": trial,
                    "is_correct": int(rng.integers(0, 2)),
                    "has_box": int(rng.integers(0, 2)),
                    "hit_max_len": 0,
                    "completion_length": int(rng.integers(100, 1000)),
                    "completion_perplexity": float(rng.uniform(1.0, 5.0)),
                    "prompt_perplexity": float(rng.uniform(5.0, 20.0)),
                    "logprob_min": float(rng.uniform(-5.0, -0.5)),
                    "logprob_iqr": float(rng.uniform(0.0, 1.0)),
                    "tail64_avg_logprob": float(rng.uniform(-2.0, -0.1)),
                })
    return pd.DataFrame(rows)


def test_evaluate_all_returns_expected_structure(synthetic_df):
    results = evaluate_all(synthetic_df)
    assert isinstance(results, dict)
    for cid, model_map in results.items():
        assert isinstance(model_map, dict)
        for model, (mu, sigma) in model_map.items():
            assert isinstance(mu, float)
            assert isinstance(sigma, float)
            assert sigma >= 0.0


def test_evaluate_all_covers_all_criteria(synthetic_df):
    results = evaluate_all(synthetic_df)
    assert set(results.keys()) == set(_CRITERION_REGISTRY.keys())


def test_evaluate_all_covers_both_models(synthetic_df):
    results = evaluate_all(synthetic_df)
    for cid, model_map in results.items():
        assert "model_A" in model_map
        assert "model_B" in model_map
