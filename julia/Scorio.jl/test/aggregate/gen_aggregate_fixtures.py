"""Generate Julia aggregate parity fixtures from the Python reference.

Run from the repository root:

    python julia/Scorio.jl/test/aggregate/gen_aggregate_fixtures.py

The fixture deliberately covers the deterministic public functions exported by
``scorio.aggregate``. Julia preserves Python's 0-based candidate/token indices
and ``-1`` missing-index sentinel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scorio import aggregate as A  # noqa: E402


def jsonable(value):
    """Recursively convert NumPy arrays/scalars and tuples for JSON output."""
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def main() -> None:
    logprobs = [-0.05, -0.7, -0.2, -1.1]
    topk = [
        [-0.05, -1.4, -3.2],
        [-0.7, -0.8, -2.4],
        [-0.2, -1.1, -2.8],
        [-1.0, -1.2, -1.7],
    ]
    ragged_topk = [[-0.1, -1.0], [-0.3, -0.8, -2.7], [-0.05]]
    answers = ["A", "A", "A", "B", "B", "C"]
    scores = [0.50, 0.55, 0.60, 0.90, 0.40, 0.30]

    confidence = {
        "mean_logprob": A.mean_logprob(logprobs),
        "sequence_logprob": A.sequence_logprob(logprobs),
        "perplexity": A.perplexity(logprobs),
        "picsar": {
            "whole": A.picsar(logprobs),
            "split": A.picsar(logprobs, answer_start=3),
            "normalized": A.picsar(logprobs, answer_start=3, normalize_reasoning=True),
            "split_zero": A.picsar(logprobs, answer_start=0),
            "split_end": A.picsar(logprobs, answer_start=len(logprobs)),
        },
        "self_certainty": {
            how: A.self_certainty(topk, aggregate=how) for how in ("mean", "min", "max")
        },
        "token_entropy": {
            how: A.token_entropy(topk, aggregate=how) for how in ("mean", "min", "max")
        },
        "varentropy": {
            how: A.varentropy(topk, aggregate=how) for how in ("mean", "min", "max")
        },
        "max_softmax_probability": {
            how: A.max_softmax_probability(topk, aggregate=how)
            for how in ("mean", "min", "max")
        },
        "logprob_margin": {
            "log": A.logprob_margin(topk),
            "prob": A.logprob_margin(topk, use_prob=True),
            "min": A.logprob_margin(topk, aggregate="min"),
            "ragged": A.logprob_margin(ragged_topk),
        },
        "token_confidence": A.token_confidence(topk).tolist(),
        "deepconf_confidence": {
            "mean": A.deepconf_confidence(topk, mode="mean"),
            "tail": A.deepconf_confidence(topk, mode="tail", tail_tokens=2),
            "lowest_group": A.deepconf_confidence(topk, mode="lowest_group", window=2),
            "bottom_group": A.deepconf_confidence(
                topk, mode="bottom_group", window=2, bottom_quantile=0.5
            ),
        },
    }

    prm = {
        method: A.prm_aggregate([0.9, 0.4, 0.95], method=method)
        for method in ("last", "min", "mean", "prod", "max")
    }

    selection = {
        "majority_vote": A.majority_vote(answers, return_index=True),
        "best_of_n": A.best_of_n(answers, scores, return_index=True, return_score=True),
        "majority_of_the_bests": A.majority_of_the_bests(
            answers, scores, return_index=True, return_score=True
        ),
        "mob": A.mob(answers, scores, return_index=True, return_score=True),
        "majority_of_the_bests_m1": A.majority_of_the_bests(
            answers, scores, m=1, return_index=True, return_score=True
        ),
        "best_of_majority": A.best_of_majority(
            answers,
            scores,
            alpha=0.4,
            aggregate="mean",
            return_index=True,
            return_score=True,
        ),
        "weighted_majority_vote_sum": A.weighted_majority_vote(
            answers,
            scores,
            aggregate="sum",
            return_index=True,
            return_score=True,
        ),
        "weighted_majority_vote_mean": A.weighted_majority_vote(
            answers,
            scores,
            aggregate="mean",
            return_index=True,
            return_score=True,
        ),
        "softmax_weighted_vote": A.softmax_weighted_vote(
            answers,
            scores,
            temperature=0.7,
            return_index=True,
            return_score=True,
        ),
        "rank_weighted_vote": A.rank_weighted_vote(
            answers,
            scores,
            p=1.3,
            return_index=True,
            return_score=True,
        ),
        "logit_weighted_vote": A.logit_weighted_vote(
            answers,
            scores,
            threshold=0.5,
            transform="logit",
            return_index=True,
            return_score=True,
        ),
        "logit_weighted_vote_linear": A.logit_weighted_vote(
            answers,
            scores,
            threshold=0.2,
            transform="linear",
            return_index=True,
            return_score=True,
        ),
        "filtered_vote_weighted": A.filtered_vote(
            answers,
            scores,
            keep=0.5,
            weighted=True,
            return_index=True,
            return_score=True,
        ),
        "filtered_vote_unweighted": A.filtered_vote(
            answers,
            scores,
            keep=3,
            weighted=False,
            return_index=True,
            return_score=True,
        ),
    }

    batch_answers = [["A", "B", "A"], ["X", "X", "Y"]]
    batch_scores = [[0.1, 0.9, 0.2], [0.4, 0.3, 0.8]]
    batch = {
        "majority_vote": A.majority_vote(batch_answers, return_index=True),
        "best_of_n": A.best_of_n(
            batch_answers, batch_scores, return_index=True, return_score=True
        ),
        "weighted_majority_vote": A.weighted_majority_vote(
            batch_answers, batch_scores, return_index=True, return_score=True
        ),
    }

    online = {
        "adaptive_consistency_stop": A.adaptive_consistency_stop(
            ["A"] * 8 + ["B"] * 2, return_prob=True
        ),
        "adaptive_tie": A.adaptive_consistency_stop(
            ["A", "A", "B", "B"], return_prob=True
        ),
        "esc_stop_true": A.esc_stop(["A", "A", "A"]),
        "esc_stop_false": A.esc_stop(["A", "B", "A"]),
        "deepconf_stop_threshold": A.deepconf_stop_threshold(
            [1.0, 2.0, 3.0, 4.0, 5.0], keep=0.2
        ),
        "deepconf_online_stop": A.deepconf_online_stop(
            [[0.0, -2.0]] * 3 + [[-4.0, -6.0]] * 3,
            threshold=2.0,
            window=3,
        ),
    }

    fixture = jsonable(
        {
            "provenance": "Generated by Python scorio.aggregate (source of truth)",
            "inputs": {
                "logprobs": logprobs,
                "topk": topk,
                "ragged_topk": ragged_topk,
                "answers": answers,
                "scores": scores,
                "batch_answers": batch_answers,
                "batch_scores": batch_scores,
            },
            "confidence": confidence,
            "prm_aggregate": prm,
            "selection": selection,
            "batch": batch,
            "online": online,
        }
    )
    target = Path(__file__).parents[1] / "fixtures" / "aggregate.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
