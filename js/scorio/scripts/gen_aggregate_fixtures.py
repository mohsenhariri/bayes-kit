"""Generate Python-reference fixtures for the TypeScript aggregate port.

Run from the repository root with::

    python js/scorio/scripts/gen_aggregate_fixtures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scorio import aggregate as agg  # noqa: E402, I001


OUT = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "aggregate.json"


def serializable(value: Any) -> Any:
    if value is agg.CGES_OTHER:
        return "CGES_OTHER"
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == math.inf:
            return "inf"
        if value == -math.inf:
            return "-inf"
    return value


def build_fixture() -> dict[str, Any]:
    logprobs = [-0.13, -0.44, -0.07, -0.81, -0.22]
    topk = [
        [-0.05, -1.7, -3.2, -4.1],
        [-0.9, -1.0, -1.4, -2.8],
        [-0.2, -0.7, -2.1, -3.0],
        [-1.1, -1.2, -1.25, -1.8],
        [-0.01, -2.4, -4.0, -5.2],
        [-0.5, -0.8, -1.9, -2.2],
    ]
    ragged = [[-0.1, -1.3], [-0.2, -0.7, -2.9], [-0.4]]
    reducers = ("mean", "min", "max")
    confidence = {
        "logprobs": logprobs,
        "topk": topk,
        "ragged": ragged,
        "mean_logprob": agg.mean_logprob(logprobs),
        "sequence_logprob": agg.sequence_logprob(logprobs),
        "perplexity": agg.perplexity(logprobs),
        "picsar": agg.picsar(logprobs),
        "picsar_split": agg.picsar(logprobs, answer_start=3),
        "picsar_normalized": agg.picsar(
            logprobs, answer_start=3, normalize_reasoning=True
        ),
        "picsar_zero": agg.picsar(logprobs, answer_start=0),
        "picsar_end": agg.picsar(logprobs, answer_start=len(logprobs)),
        "self_certainty": {
            reducer: agg.self_certainty(topk, aggregate=reducer) for reducer in reducers
        },
        "token_entropy": {
            reducer: agg.token_entropy(topk, aggregate=reducer) for reducer in reducers
        },
        "varentropy": {
            reducer: agg.varentropy(topk, aggregate=reducer) for reducer in reducers
        },
        "max_softmax_probability": {
            reducer: agg.max_softmax_probability(topk, aggregate=reducer)
            for reducer in reducers
        },
        "logprob_margin": {
            reducer: agg.logprob_margin(topk, aggregate=reducer) for reducer in reducers
        },
        "prob_margin": agg.logprob_margin(topk, use_prob=True),
        "token_confidence": agg.token_confidence(topk),
        "deepconf": {
            "mean": agg.deepconf_confidence(topk),
            "tail": agg.deepconf_confidence(topk, mode="tail", tail_tokens=3),
            "lowest_group": agg.deepconf_confidence(
                topk, mode="lowest_group", window=3
            ),
            "bottom_group": agg.deepconf_confidence(
                topk, mode="bottom_group", window=3, bottom_quantile=0.4
            ),
        },
        "ragged_entropy": agg.token_entropy(ragged),
        "ragged_self_certainty": agg.self_certainty(ragged),
        "bare_row_entropy": agg.token_entropy(topk[0]),
        "single_candidate_margin": agg.logprob_margin([[-0.4]]),
        "extreme": {
            "topk": [[0.0, -1000.0]],
            "token_entropy": agg.token_entropy([[0.0, -1000.0]]),
            "varentropy": agg.varentropy([[0.0, -1000.0]]),
            "self_certainty": agg.self_certainty([[0.0, -1000.0]]),
        },
    }

    steps = [0.91, 0.42, 0.88, 0.73]
    prm = {
        "steps": steps,
        "outputs": {
            method: agg.prm_aggregate(steps, method=method)
            for method in ("last", "min", "mean", "prod", "max")
        },
    }

    answers = ["A", "A", "A", "B", "B", "C", None, ""]
    scores = [0.32, 0.61, 0.55, 0.94, 0.25, 0.15, 99.0, 98.0]
    metadata = {"return_index": True, "return_score": True}
    selection = {
        "answers": answers,
        "scores": scores,
        "majority_vote": agg.majority_vote(answers, return_index=True),
        "best_of_n": agg.best_of_n(answers, scores, **metadata),
        "weighted_sum": agg.weighted_majority_vote(
            answers, scores, aggregate="sum", **metadata
        ),
        "weighted_mean": agg.weighted_majority_vote(
            answers, scores, aggregate="mean", **metadata
        ),
        "mob_default": agg.majority_of_the_bests(answers, scores, **metadata),
        "mob_m3": agg.majority_of_the_bests(answers, scores, m=3, **metadata),
        "best_of_majority": agg.best_of_majority(
            answers, scores, alpha=0.4, aggregate="mean", **metadata
        ),
        "best_of_majority_sum": agg.best_of_majority(
            answers, scores, aggregate="sum", **metadata
        ),
        "best_of_majority_max": agg.best_of_majority(
            answers, scores, aggregate="max", **metadata
        ),
        "softmax": agg.softmax_weighted_vote(
            answers, scores, temperature=0.7, **metadata
        ),
        "softmax_infinite": agg.softmax_weighted_vote(
            answers, scores, temperature=math.inf, **metadata
        ),
        "rank_p0": agg.rank_weighted_vote(answers, scores, p=0.0, **metadata),
        "rank_p1": agg.rank_weighted_vote(answers, scores, p=1.0, **metadata),
        "rank_fractional": agg.rank_weighted_vote(answers, scores, p=1.7, **metadata),
        "logit": agg.logit_weighted_vote(answers, scores, **metadata),
        "linear": agg.logit_weighted_vote(
            answers, scores, threshold=0.2, transform="linear", **metadata
        ),
        "filtered_fraction": agg.filtered_vote(answers, scores, keep=0.5, **metadata),
        "filtered_count": agg.filtered_vote(
            answers, scores, keep=3, weighted=False, **metadata
        ),
        "filtered_all": agg.filtered_vote(
            answers, scores, keep=1.0, weighted=False, **metadata
        ),
    }

    batch_answers = [["A", "B", "A"], ["X", "Y", "Y"], [None, "", np.nan]]
    batch_scores = [[0.2, 0.9, 0.4], [0.8, 0.2, 0.7], [1.0, 2.0, 3.0]]
    batch = {
        "answers": batch_answers,
        "scores": batch_scores,
        "majority_vote": agg.majority_vote(batch_answers, return_index=True),
        "best_of_n": agg.best_of_n(batch_answers, batch_scores, **metadata),
        "weighted": agg.weighted_majority_vote(batch_answers, batch_scores, **metadata),
        "mob": agg.majority_of_the_bests(batch_answers, batch_scores, **metadata),
        "filtered": agg.filtered_vote(
            batch_answers, batch_scores, keep=0.67, **metadata
        ),
    }

    mob_answers = ["D", "B", "D", "B", "A", "B", "A"]
    mob_scores = [0.6667, 0.3333, 0.8333, 0.1667, 0.5, 0.0, 1.0]
    rank_answers = ["A", "B", "C", "A", "B", "C", "A", "C"]
    rank_scores = [2.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0]
    exact = {
        "mob_tie_answers": mob_answers,
        "mob_tie_scores": mob_scores,
        "mob_tie": agg.majority_of_the_bests(mob_answers, mob_scores),
        "rank_answers": rank_answers,
        "rank_scores": rank_scores,
        "rank_integer": agg.rank_weighted_vote(rank_answers, rank_scores, p=3),
        "filtered_sorted_tie": agg.filtered_vote(
            ["A", "B", "B", "A"],
            [0.1, 0.3, 0.8, 0.9],
            keep=1.0,
            weighted=False,
            return_index=True,
        ),
    }

    warmup = [0.7, 2.1, 1.4, 4.8, 3.0, 2.9, 5.2]
    online_topk = [[0.0, -2.0]] * 3 + [[-4.0, -6.0]] * 4
    dominant = ["A"] * 8 + ["B"] * 2
    large_counts = (100_000, 99_999)
    online = {
        "warmup": warmup,
        "threshold_keep_02": agg.deepconf_stop_threshold(warmup, keep=0.2),
        "threshold_keep_all": agg.deepconf_stop_threshold(warmup, keep=1.0),
        "topk": online_topk,
        "token_stop": agg.deepconf_online_stop(online_topk, 2.0, window=3),
        "token_no_stop": agg.deepconf_online_stop(online_topk, 0.5, window=3),
        "token_equal_no_stop": agg.deepconf_online_stop(online_topk, 1.0, window=3),
        "token_later_stop": agg.deepconf_online_stop(
            [[-4.0, -6.0]] * 3 + [[0.0, -2.0]] * 3,
            2.0,
            window=3,
        ),
        "adaptive_dominant": agg.adaptive_consistency_stop(dominant, return_prob=True),
        "adaptive_tie": agg.adaptive_consistency_stop(
            ["A", "B", "A", "B"], return_prob=True
        ),
        "adaptive_invalid": agg.adaptive_consistency_stop([None, ""], return_prob=True),
        "adaptive_large_near_counts": large_counts,
        "adaptive_large_near": agg.adaptive_consistency_stop(
            ["A"] * large_counts[0] + ["B"] * large_counts[1],
            return_prob=True,
        ),
        "dirichlet_three": agg.adaptive_consistency_dirichlet_stop(
            ["A"] * 5 + ["B"] * 2 + ["C"], return_prob=True
        ),
        "dirichlet_large": agg.adaptive_consistency_dirichlet_stop(
            ["A"] * 1000 + ["B"] * 900 + ["C"], return_prob=True
        ),
        "dirichlet_very_large": agg.adaptive_consistency_dirichlet_stop(
            ["A"] * 100_000 + ["B"] * 99_900 + ["C"], return_prob=True
        ),
        "dirichlet_symmetric": agg.adaptive_consistency_dirichlet_stop(
            ["A"] * 1000 + ["B"] * 1000 + ["C"] * 1000, return_prob=True
        ),
        "crp_dominant": agg.adaptive_consistency_crp_stop(
            ["A"] * 7 + ["B"],
            horizon=12,
            n_alpha=20,
            n_simulations=200,
            seed=7,
            return_prob=True,
        ),
        "crp_tie": agg.adaptive_consistency_crp_stop(
            ["A", "B", "A", "B"],
            horizon=12,
            n_alpha=20,
            n_simulations=200,
            seed=7,
            return_prob=True,
        ),
        "esc_true": agg.esc_stop(["A", "A", "A"]),
        "esc_false": agg.esc_stop(["A", "B", "A"]),
        "esc_invalid": agg.esc_stop(["A", None, "A"]),
    }

    fitted = agg.fit_kde_vote_calibration(
        [0.8, 0.9, 0.1, 0.2], [1, 1, 0, 0], n_bins=2, bandwidth=0.5
    )
    calibration = {
        "correct_logits": fitted.correct_logits,
        "incorrect_logits": fitted.incorrect_logits,
        "correct_bandwidth": fitted.correct_bandwidth,
        "incorrect_bandwidth": fitted.incorrect_bandwidth,
        "bin_edges": fitted.bin_edges,
        "bin_probability": fitted.bin_probability,
        "calibrated_probability": fitted.calibrated_probability([0.2, 0.7, 0.8, 0.95]),
        "log_density_ratio": fitted.log_density_ratio([0.4, 0.7]),
        "vote": agg.kde_weighted_vote(["A", "A", "B"], [0.2, 0.2, 0.8], fitted),
    }

    cges = {
        "vote": agg.cges_vote(
            ["A", "A", "B"],
            [0.7, 0.9, 0.6],
            return_index=True,
            return_score=True,
        ),
        "other": agg.cges_vote(["A"], [0.1], allow_other=True),
        "other_metadata": agg.cges_vote(
            ["A"],
            [0.1],
            allow_other=True,
            return_index=True,
            return_score=True,
        ),
        "stop": agg.cges_stop(["A"], [0.9], threshold=0.8, return_prob=True),
        "other_stop": agg.cges_stop(
            ["A"], [0.1], threshold=0.8, include_other=True, return_prob=True
        ),
    }

    return serializable(
        {
            "confidence": confidence,
            "prm": prm,
            "selection": selection,
            "batch": batch,
            "exact": exact,
            "online": online,
            "calibration": calibration,
            "cges": cges,
        }
    )


def main() -> None:
    fixture = build_fixture()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
