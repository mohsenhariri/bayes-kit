"""Generate ground-truth fixtures for the TypeScript ``sinf`` port.

Runs the Python reference (``scorio.sinf``) on FIXED deterministic inputs
(streams, panels, a small tensor, count vectors, and (mu, sigma) summaries built
with ``numpy.random.default_rng(0)``), dumps both the inputs and the reference
outputs to ``test/fixtures/sinf.json``, and the vitest suite asserts the TS port
reproduces them (to ~1e-6 for CS bounds, exactly for discrete decisions).

Non-finite floats (e.g. the degenerate ``z = inf`` of ``pairwise_confidence``)
are serialized as the tokens ``"inf"`` / ``"-inf"`` / ``"nan"`` so the JSON stays
importable by the JS bundler; the test un-tokenizes them.

Run from repo root:  python js/scorio/scripts/gen_sinf_fixtures.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

# Import the LOCAL scorio (this repo), not any other installed clone.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scorio import sinf  # noqa: E402
from scorio.sinf import _panel  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "sinf.json"


def clean(o):
    """Recursively JSON-sanitize numpy scalars/arrays and non-finite floats."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return clean(float(o))
    if isinstance(o, float):
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        if math.isnan(o):
            return "nan"
        return o
    return o


def bounds(lohi):
    lo, hi = lohi
    return {"lo": clean(lo), "hi": clean(hi)}


def build():
    rng = np.random.default_rng(0)

    # ---------------- streams ----------------
    stream_inputs = {
        "uniform50": rng.random(50),
        "bernoulli80": (rng.random(80) < 0.65).astype(float),
        "short5": rng.random(5),
        "high60": np.clip(0.6 + 0.4 * rng.random(60), 0.0, 1.0),
    }
    streams = []
    for name, x in stream_inputs.items():
        entry = {"name": name, "x": clean(x), "confseq": {}}
        for method in ("betting", "hoeffding", "asymp"):
            lo, hi = sinf.confseq_mean_path(x, method=method)
            fin_lo, fin_hi = sinf.confseq_mean(x, method=method)
            entry["confseq"][method] = {
                "lo": clean(lo),
                "hi": clean(hi),
                "final": {"lo": clean(fin_lo), "hi": clean(fin_hi)},
            }
        entry["fixed_ci"] = bounds(sinf.fixed_ci_path(x))
        streams.append(entry)

    # ---------------- single-model panel ----------------
    RA = rng.random((6, 40))
    RB = rng.random((6, 45))
    panel_single = {
        "R": clean(RA),
        "trial_scores": clean(_panel.trial_scores(RA)),
        "question_scores": clean(_panel.question_scores(RA)),
        "stream_trials": clean(_panel.stream_from_tensor(RA, axis="trials")),
        "stream_questions": clean(_panel.stream_from_tensor(RA, axis="questions")),
        "score_confseq_path": bounds(sinf.score_confseq_path(RA)),
        "score_confseq": {
            "lo": clean(sinf.score_confseq(RA)[0]),
            "hi": clean(sinf.score_confseq(RA)[1]),
        },
        "precision_stop_met": clean(sinf.precision_stop(RA, 0.25)),
        "precision_stop_unmet": clean(sinf.precision_stop(RA, 0.001)),
    }

    # ---------------- paired panels ----------------
    panel_pair_indep = {
        "RA": clean(RA),
        "RB": clean(RB),
        "paired_trial_diffs": clean(_panel.paired_trial_diffs(RA, RB)),
        "compare_paired_path": bounds(sinf.compare_paired_path(RA, RB)),
        "compare_paired": {
            "lo": clean(sinf.compare_paired(RA, RB)[0]),
            "hi": clean(sinf.compare_paired(RA, RB)[1]),
        },
        "decide_better": clean(sinf.decide_better(RA, RB)),
    }
    # decisive pair: A clearly above B on shared questions -> decide "A".
    AH = 0.55 + 0.4 * rng.random((6, 50))
    BL = 0.4 * rng.random((6, 50))
    panel_pair_decisive = {
        "RA": clean(AH),
        "RB": clean(BL),
        "compare_paired": {
            "lo": clean(sinf.compare_paired(AH, BL)[0]),
            "hi": clean(sinf.compare_paired(AH, BL)[1]),
        },
        "decide_better": clean(sinf.decide_better(AH, BL)),
    }

    # ---------------- multi-model tensor ----------------
    R = rng.random((4, 6, 30))
    tensor = {
        "R": clean(R),
        "empirical_scores": clean(sinf.empirical_scores(R)),
        "top1_pairs": clean(sinf.should_stop_top1_av(R)),
        "top1_leader": clean(sinf.should_stop_top1_av(R, correction="leader")),
        "top1_none": clean(sinf.should_stop_top1_av(R, correction="none")),
        "full_ranking": clean(sinf.should_stop_full_ranking(R)),
        "allocation": clean(sinf.suggest_next_allocation_stratified(R)),
    }

    # Clearly-separated (constant) tensor -> resolved leader / full order.
    consts = np.array([0.2, 0.45, 0.65, 0.9])
    R_sep = np.repeat(consts[:, None, None], 5, axis=1).repeat(20, axis=2)
    tensor_sep = {
        "R": clean(R_sep),
        "empirical_scores": clean(sinf.empirical_scores(R_sep)),
        "top1_pairs": clean(sinf.should_stop_top1_av(R_sep)),
        "full_ranking": clean(sinf.should_stop_full_ranking(R_sep)),
    }

    # select_best_fixed_budget: RNG parity is NOT expected; on a clearly
    # separated tensor `best` (and the shape-determined `spent`) still match.
    sb = sinf.select_best_fixed_budget(R_sep, budget=64, seed=0)
    select_best = {
        "R": clean(R_sep),
        "budget": 64,
        "seed": 0,
        "best": clean(sb["best"]),
        "spent": clean(sb["spent"]),
        "rounds": clean(sb["rounds"]),
    }

    # ---------------- inference-time voting ----------------
    count_vectors = [[12, 5, 3], [8, 7], [20], [30, 2], [1, 1, 1]]
    votes = []
    for cv in count_vectors:
        arr = np.array(cv, dtype=float)
        votes.append(
            {
                "counts": cv,
                "should_stop_sampling": clean(sinf.should_stop_sampling(arr)),
                "should_stop_sampling_a01": clean(
                    sinf.should_stop_sampling(arr, alpha=0.01)
                ),
                "adaptive": clean(sinf.adaptive_consistency_stop(arr)),
                "adaptive_t80": clean(
                    sinf.adaptive_consistency_stop(arr, thresh=0.80)
                ),
            }
        )

    answers = ["a", "b", "a", "", "c", "a", "b", None, "c", "a"]
    labels, counts = sinf.counts_from_answers(answers)
    counts_from_answers = {
        "answers": answers,
        "labels": clean(labels),
        "counts": clean(counts),
    }

    # ---------------- fixed-look legacy (mu, sigma) ----------------
    mus = [0.80, 0.75, 0.60, 0.50]
    sigmas = [0.03, 0.04, 0.05, 0.02]
    rc = sinf.ranking_confidence(mus[0], sigmas[0], mus[1], sigmas[1])
    pc = sinf.pairwise_confidence(mus[0], sigmas[0], mus[1], sigmas[1], cov=0.0005)
    pc0 = sinf.pairwise_confidence(mus[0], sigmas[0], mus[1], sigmas[1])
    legacy = {
        "mus": mus,
        "sigmas": sigmas,
        "ranking_confidence": {"rho": clean(rc[0]), "z": clean(rc[1])},
        "ci_90": {
            "lo": clean(sinf.ci_from_mu_sigma(0.7, 0.05, confidence=0.9)[0]),
            "hi": clean(sinf.ci_from_mu_sigma(0.7, 0.05, confidence=0.9)[1]),
        },
        "ci_90_clip": {
            "lo": clean(
                sinf.ci_from_mu_sigma(0.97, 0.05, confidence=0.9, clip=(0.0, 1.0))[0]
            ),
            "hi": clean(
                sinf.ci_from_mu_sigma(0.97, 0.05, confidence=0.9, clip=(0.0, 1.0))[1]
            ),
        },
        "should_stop_half": clean(
            sinf.should_stop(0.01, confidence=0.95, max_half_width=0.02)
        ),
        "should_stop_half_false": clean(
            sinf.should_stop(0.02, confidence=0.95, max_half_width=0.02)
        ),
        "should_stop_ci": clean(
            sinf.should_stop(0.02, confidence=0.95, max_ci_width=0.1)
        ),
        "top1_ci_overlap": clean(
            sinf.should_stop_top1(mus, sigmas, method="ci_overlap")
        ),
        "top1_zscore": clean(sinf.should_stop_top1(mus, sigmas, method="zscore")),
        "alloc_ci_overlap": list(
            sinf.suggest_next_allocation(mus, sigmas, method="ci_overlap")
        ),
        "alloc_zscore": list(
            sinf.suggest_next_allocation(mus, sigmas, method="zscore")
        ),
        "pairwise_cov": {"rho": clean(pc[0]), "z": clean(pc[1])},
        "pairwise_indep": {"rho": clean(pc0[0]), "z": clean(pc0[1])},
        # degenerate branches (var == 0)
        "pairwise_tie": {
            "rho": clean(sinf.pairwise_confidence(0.8, 0.0, 0.8, 0.0)[0]),
            "z": clean(sinf.pairwise_confidence(0.8, 0.0, 0.8, 0.0)[1]),
        },
        "ranking_conf_certain": {
            "rho": clean(sinf.ranking_confidence(0.8, 0.0, 0.7, 0.0)[0]),
            "z": clean(sinf.ranking_confidence(0.8, 0.0, 0.7, 0.0)[1]),
        },
    }

    return {
        "streams": streams,
        "panel_single": panel_single,
        "panel_pair_indep": panel_pair_indep,
        "panel_pair_decisive": panel_pair_decisive,
        "tensor": tensor,
        "tensor_sep": tensor_sep,
        "select_best": select_best,
        "votes": votes,
        "counts_from_answers": counts_from_answers,
        "legacy": legacy,
    }


def main():
    fx = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fx, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
