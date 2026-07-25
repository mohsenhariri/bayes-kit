"""Generate Python-reference fixtures for the TypeScript utils/Prior ports.

Run from the repository root:

    python js/scorio/scripts/gen_utils_fixtures.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from scorio import rank
from scorio import utils


OUT = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "utils.json"


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def rank_score_fixtures():
    cases = [
        {
            "name": "basic_ties",
            "scores": [95.0, 87.5, 87.5, 80.0, 75.0],
            "kwargs": {},
        },
        {"name": "all_tied", "scores": [5.0, 5.0, 5.0], "kwargs": {}},
        {
            "name": "near_equal",
            "scores": [10.0, 10.0 + 1e-14, 5.0],
            "kwargs": {"tol": 1e-12},
        },
        {
            "name": "zscore_ties",
            "scores": [10.0, 9.5, 5.0],
            "kwargs": {
                "sigmas_in_id_order": [1.0, 1.0, 0.1],
                "confidence": 0.95,
            },
        },
        {
            "name": "overlap_ties",
            "scores": [10.0, 9.5, 5.0],
            "kwargs": {
                "sigmas_in_id_order": [1.0, 1.0, 0.1],
                "confidence": 0.95,
                "ci_tie_method": "ci_overlap_adjacent",
            },
        },
        {
            "name": "zero_sigma",
            "scores": [3.0, 2.0, 1.0],
            "kwargs": {"sigmas_in_id_order": [0.0, 0.0, 0.0]},
        },
        {"name": "empty", "scores": [], "kwargs": {}},
    ]
    for case in cases:
        case["expected"] = utils.rank_scores(case["scores"], **case["kwargs"])
    return cases


def comparison_fixtures():
    cases = [
        ("identical", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ("reversed", [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
        ("adjacent_swap", [1, 2, 3, 4, 5], [1, 3, 2, 4, 5]),
        ("scipy_docs", [12, 2, 1, 12, 2], [1, 4, 7, 1, 0]),
        ("ties", [1, 1, 2], [1, 2, 2]),
        ("mixed", [1, 3, 2, 5, 4], [2, 1, 3, 4, 5]),
        (
            "asymptotic_untied",
            list(range(1, 35)),
            list(range(4, 35)) + [2, 1, 3],
        ),
        (
            "tie_heavy",
            [1, 1, 1, 4, 4, 6, 7, 7],
            [2, 2, 1, 4, 5, 5, 7, 7],
        ),
    ]
    return [
        {
            "name": name,
            "a": a,
            "b": b,
            "expected": utils.compare_rankings(a, b),
        }
        for name, a, b in cases
    ]


def combinatorial_fixtures():
    lehmer = []
    for n in range(7):
        for permutation in itertools.permutations(range(n)):
            lehmer.append(
                {
                    "n": n,
                    "permutation": permutation,
                    "hash": str(utils.lehmer_hash(permutation)),
                }
            )

    weak_rankings = []
    for n in range(6):
        total = utils.ordered_bell(n)[n]
        for hash_value in range(total):
            weak_rankings.append(
                {
                    "n": n,
                    "hash": str(hash_value),
                    "ranking": utils.unhash_ranking(hash_value, n),
                }
            )

    combinations = []
    for n in range(9):
        for k in range(n + 1):
            for indices in itertools.combinations(range(n), k):
                combinations.append(
                    {
                        "n": n,
                        "k": k,
                        "indices": indices,
                        "rank": str(utils.comb_rank_lex(indices, n, k)),
                    }
                )

    return {
        "ordered_bell_17": [str(value) for value in utils.ordered_bell(17)],
        "lehmer": lehmer,
        "weak_rankings": weak_rankings,
        "combinations": combinations,
        "large": {
            "lehmer_reverse_n19": str(utils.lehmer_hash(list(reversed(range(19))))),
            "ranking_all_tied_n17": str(utils.ranking_hash([1] * 17)),
        },
        "blocks": [
            {
                "ranks": ranks,
                "tol": tolerance,
                "expected": utils.blocks_from_rank_list(ranks, tol=tolerance),
            }
            for ranks, tolerance in [
                ([], 1e-12),
                ([1, 2, 3], 1e-12),
                ([1, 1, 1], 1e-12),
                ([1, 2, 2, 4], 1e-12),
                ([3, 2, 1], 1e-12),
                ([1.0, 1.0 + 1e-14, 2.0], 1e-12),
            ]
        ],
    }


def prior_fixtures():
    R0 = np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=int,
    )
    theta = np.array([-0.5, 0.0, 0.5], dtype=float)
    empirical = rank.EmpiricalPrior(R0, var=1.5)
    priors = {
        "gaussian": rank.GaussianPrior(mean=0.2, var=1.5),
        "laplace": rank.LaplacePrior(loc=0.2, scale=1.5),
        "cauchy": rank.CauchyPrior(loc=0.2, scale=1.5),
        "uniform": rank.UniformPrior(),
        "custom": rank.CustomPrior(lambda values: float(np.sum(np.abs(values)))),
        "empirical": empirical,
    }
    return {
        "R0": R0,
        "theta": theta,
        "empirical_prior_mean": empirical.prior_mean,
        "penalties": {name: prior.penalty(theta) for name, prior in priors.items()},
    }


def main():
    fixtures = {
        "public_api": {
            "rank": rank.__all__,
            "utils": utils.__all__,
        },
        "rank_scores": rank_score_fixtures(),
        "comparisons": comparison_fixtures(),
        "combinatorial": combinatorial_fixtures(),
        "priors": prior_fixtures(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(json_safe(fixtures), indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
