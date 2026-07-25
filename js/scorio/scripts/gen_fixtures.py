"""Generate ground-truth fixtures for the TypeScript rank port.

Reproduces the conftest fixtures from tests/rank/conftest.py and dumps, for every
ranking method, the exact input tensor plus the reference (ranking, scores) so the
vitest suite can assert numerical parity with the Python reference implementation.

Run from repo root:  python js/scorio/scripts/gen_fixtures.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scorio import rank  # noqa: E402

DATA = ROOT / "tests" / "data" / "R_top_p.npz"
OUT = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "rank.json"


def load_aime25() -> np.ndarray:
    with np.load(DATA, allow_pickle=True) as d:
        return d["aime25"].astype(int, copy=False)


def build_fixtures():
    top = load_aime25()

    # --- conftest.ordered_binary_R ---
    raw0 = top[0, :24, :10]
    raw1 = top[1, :24, :10]
    raw2 = top[2, :24, :10]
    best = np.maximum(raw0, raw1)
    mid_high = raw0.copy()
    mid_low = np.minimum(mid_high, raw1)
    worst = np.minimum(mid_low, raw2)
    ordered_binary_R = np.stack([best, mid_high, mid_low, worst], axis=0).astype(int)

    ordered_binary_small_R = ordered_binary_R[:, :10, :5]
    ordered_binary_matrix = ordered_binary_small_R[:, :, 0]

    # --- conftest.tie_heavy_R: identical, rolled, inverted models (discriminating) ---
    base = top[4, :6, :4]
    tie_heavy_R = np.stack(
        [base, base.copy(), np.roll(base, shift=1, axis=1), 1 - base], axis=0
    ).astype(int)
    tie_heavy_matrix = tie_heavy_R[:, :, 0]

    # --- D3: four distinct, moderately separated models (stable non-trivial order) ---
    distinct_R = top[[2, 0, 5, 1], :12, :8].astype(int)
    distinct_matrix = distinct_R[:, :, 0]

    # --- conftest.multiclass_rank_data ---
    R_multi = (top[0:3, :10, :7] + top[3:6, :10, :7]).astype(int)
    w = np.array([0.0, 0.5, 1.0], dtype=float)
    R0_shared = (top[6, :10, :3] + top[7, :10, :3]).astype(int)

    # kind:
    #   "exact"      -> ranking exact + scores allclose (rtol 1e-6)
    #   "loose"      -> ranking exact + scores allclose only when |score|<50 (optimizer)
    #   "structural" -> valid ranking + ordering sanity only (stochastic / non-unique)
    #   "error"      -> finalized Python rejects the input (for example a
    #                   separated maximum-likelihood profile)
    cases = []

    def add(name, ranking, scores, kind, inp):
        cases.append(
            {
                "name": name,
                "input": np.asarray(inp).tolist(),
                "ranking": np.asarray(ranking, dtype=float).tolist(),
                "scores": np.asarray(scores, dtype=float).tolist(),
                "kind": kind,
            }
        )

    def add_call(name, kind, inp, builder):
        try:
            ranking, scores = builder(inp)
        except (TypeError, ValueError, RuntimeError) as exc:
            cases.append(
                {
                    "name": name,
                    "input": np.asarray(inp).tolist(),
                    "kind": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return
        add(name, ranking, scores, kind, inp)

    # Each binary method: builder(R) -> (ranking, scores). Run over both 3D datasets.
    binary_methods = [
        ("avg", "exact", lambda R: rank.avg(R, return_scores=True)),
        ("pass_at_k", "exact", lambda R: rank.pass_at_k(R, k=2, return_scores=True)),
        ("pass_hat_k", "exact", lambda R: rank.pass_hat_k(R, k=2, return_scores=True)),
        (
            "g_pass_at_k_tau",
            "exact",
            lambda R: rank.g_pass_at_k_tau(R, k=2, tau=0.7, return_scores=True),
        ),
        (
            "mg_pass_at_k",
            "exact",
            lambda R: rank.mg_pass_at_k(R, k=2, return_scores=True),
        ),
        (
            "inverse_difficulty",
            "exact",
            lambda R: rank.inverse_difficulty(R, return_scores=True),
        ),
        ("elo", "exact", lambda R: rank.elo(R, return_scores=True)),
        ("glicko", "exact", lambda R: rank.glicko(R, return_scores=True)),
        ("trueskill", "exact", lambda R: rank.trueskill(R, return_scores=True)),
        (
            "bradley_terry",
            "loose",
            lambda R: rank.bradley_terry(R, max_iter=80, return_scores=True),
        ),
        (
            "bradley_terry_map",
            "loose",
            lambda R: rank.bradley_terry_map(
                R, prior=1.0, max_iter=80, return_scores=True
            ),
        ),
        (
            "bradley_terry_davidson",
            "loose",
            lambda R: rank.bradley_terry_davidson(R, max_iter=80, return_scores=True),
        ),
        (
            "bradley_terry_davidson_map",
            "loose",
            lambda R: rank.bradley_terry_davidson_map(
                R, prior=1.0, max_iter=80, return_scores=True
            ),
        ),
        (
            "rao_kupper",
            "loose",
            lambda R: rank.rao_kupper(
                R, tie_strength=1.1, max_iter=80, return_scores=True
            ),
        ),
        (
            "rao_kupper_map",
            "loose",
            lambda R: rank.rao_kupper_map(
                R, tie_strength=1.1, prior=1.0, max_iter=80, return_scores=True
            ),
        ),
        (
            "thompson",
            "structural",
            lambda R: rank.thompson(R, n_samples=700, seed=7, return_scores=True),
        ),
        (
            "bayesian_mcmc",
            "structural",
            lambda R: rank.bayesian_mcmc(
                R, n_samples=400, burnin=100, seed=7, return_scores=True
            ),
        ),
        ("borda", "exact", lambda R: rank.borda(R, return_scores=True)),
        ("copeland", "exact", lambda R: rank.copeland(R, return_scores=True)),
        ("win_rate", "exact", lambda R: rank.win_rate(R, return_scores=True)),
        ("minimax", "exact", lambda R: rank.minimax(R, return_scores=True)),
        ("schulze", "exact", lambda R: rank.schulze(R, return_scores=True)),
        ("ranked_pairs", "exact", lambda R: rank.ranked_pairs(R, return_scores=True)),
        (
            "kemeny_young",
            "exact",
            lambda R: rank.kemeny_young(R, time_limit=1.0, return_scores=True),
        ),
        ("nanson", "exact", lambda R: rank.nanson(R, return_scores=True)),
        ("baldwin", "exact", lambda R: rank.baldwin(R, return_scores=True)),
        (
            "majority_judgment",
            "exact",
            lambda R: rank.majority_judgment(R, return_scores=True),
        ),
        ("rasch", "loose", lambda R: rank.rasch(R, max_iter=60, return_scores=True)),
        (
            "rasch_map",
            "loose",
            lambda R: rank.rasch_map(R, prior=1.0, max_iter=60, return_scores=True),
        ),
        (
            "rasch_2pl",
            "loose",
            lambda R: rank.rasch_2pl(R, max_iter=60, return_scores=True),
        ),
        (
            "rasch_2pl_map",
            "loose",
            lambda R: rank.rasch_2pl_map(R, prior=1.0, max_iter=60, return_scores=True),
        ),
        (
            "rasch_3pl",
            "loose",
            lambda R: rank.rasch_3pl(
                R, max_iter=50, fix_guessing=0.2, return_scores=True
            ),
        ),
        (
            "rasch_3pl_map",
            "loose",
            lambda R: rank.rasch_3pl_map(
                R, prior=1.0, max_iter=50, fix_guessing=0.2, return_scores=True
            ),
        ),
        (
            "rasch_mml",
            "loose",
            lambda R: rank.rasch_mml(
                R, max_iter=10, em_iter=6, n_quadrature=9, return_scores=True
            ),
        ),
        (
            "rasch_mml_credible",
            "loose",
            lambda R: rank.rasch_mml_credible(
                R,
                quantile=0.1,
                max_iter=10,
                em_iter=6,
                n_quadrature=9,
                return_scores=True,
            ),
        ),
        ("pagerank", "exact", lambda R: rank.pagerank(R, return_scores=True)),
        ("spectral", "exact", lambda R: rank.spectral(R, return_scores=True)),
        (
            "alpharank",
            "exact",
            lambda R: rank.alpharank(
                R, population_size=20, max_iter=10_000, return_scores=True
            ),
        ),
        ("nash", "structural", lambda R: rank.nash(R, return_scores=True)),
        (
            "rank_centrality",
            "exact",
            lambda R: rank.rank_centrality(R, return_scores=True),
        ),
        ("serial_rank", "exact", lambda R: rank.serial_rank(R, return_scores=True)),
        ("hodge_rank", "exact", lambda R: rank.hodge_rank(R, return_scores=True)),
        (
            "plackett_luce",
            "loose",
            lambda R: rank.plackett_luce(R, max_iter=80, return_scores=True),
        ),
        (
            "plackett_luce_map",
            "loose",
            lambda R: rank.plackett_luce_map(
                R, prior=1.0, max_iter=80, return_scores=True
            ),
        ),
        (
            "davidson_luce",
            "loose",
            lambda R: rank.davidson_luce(R, max_iter=80, return_scores=True),
        ),
        (
            "davidson_luce_map",
            "loose",
            lambda R: rank.davidson_luce_map(
                R, prior=1.0, max_iter=80, return_scores=True
            ),
        ),
        (
            "bradley_terry_luce",
            "loose",
            lambda R: rank.bradley_terry_luce(R, max_iter=80, return_scores=True),
        ),
        (
            "bradley_terry_luce_map",
            "loose",
            lambda R: rank.bradley_terry_luce_map(
                R, prior=1.0, max_iter=80, return_scores=True
            ),
        ),
    ]

    datasets = [
        ("D1", ordered_binary_small_R),
        ("D2", tie_heavy_R),
        ("D3", distinct_R),
    ]
    for name, kind, builder in binary_methods:
        for dslabel, R in datasets:
            # D2 has two identical models; optimizer symmetry-breaking is
            # implementation-specific, so only require structure there for
            # optimizer-based methods.
            eff = "structural" if (kind == "loose" and dslabel == "D2") else kind
            add_call(f"{name}@{dslabel}", eff, R, builder)

    # dynamic_irt takes a 2D matrix.
    for dslabel, M in [
        ("D1", ordered_binary_matrix),
        ("D2", tie_heavy_matrix),
        ("D3", distinct_matrix),
    ]:
        eff = "structural" if dslabel == "D2" else "loose"
        add_call(
            f"dynamic_irt@{dslabel}",
            eff,
            M,
            lambda X: rank.dynamic_irt(
                X, variant="linear", max_iter=60, return_scores=True
            ),
        )

    # bayes: multiclass weighted with shared R0 prior (its own input).
    ranking, scores = rank.bayes(R_multi, w=w, R0=R0_shared, return_scores=True)
    add("bayes@multi", ranking, scores, "exact", R_multi)

    # --- Non-default option paths (locked on the distinct dataset D3) ---------
    option_cases = []

    def add_opt(name, fn):
        ranking, scores = fn()
        option_cases.append(
            {
                "name": name,
                "ranking": np.asarray(ranking, dtype=float).tolist(),
                "scores": np.asarray(scores, dtype=float).tolist(),
            }
        )

    D = distinct_R
    add_opt("borda_dense", lambda: rank.borda(D, method="dense", return_scores=True))
    add_opt(
        "minimax_wv",
        lambda: rank.minimax(D, variant="winning_votes", return_scores=True),
    )
    add_opt(
        "schulze_ignore",
        lambda: rank.schulze(D, tie_policy="ignore", return_scores=True),
    )
    add_opt(
        "nash_eq", lambda: rank.nash(D, score_type="equilibrium", return_scores=True)
    )
    add_opt(
        "nash_adv",
        lambda: rank.nash(D, score_type="advantage_vs_equilibrium", return_scores=True),
    )
    add_opt("glicko_c30", lambda: rank.glicko(D, c=30.0, return_scores=True))
    add_opt(
        "elo_draw_k16",
        lambda: rank.elo(D, tie_handling="draw", K=16.0, return_scores=True),
    )
    add_opt(
        "trueskill_draw",
        lambda: rank.trueskill(
            D, tie_handling="draw", draw_margin=0.1, return_scores=True
        ),
    )
    add_opt(
        "hodge_decisive",
        lambda: rank.hodge_rank(D, weight_method="decisive", return_scores=True),
    )
    add_opt(
        "hodge_logodds",
        lambda: rank.hodge_rank(D, pairwise_stat="log_odds", return_scores=True),
    )
    add_opt(
        "rc_teleport",
        lambda: rank.rank_centrality(
            D, tie_handling="half", teleport=0.1, return_scores=True
        ),
    )
    add_opt(
        "rc_ignore_smooth",
        lambda: rank.rank_centrality(
            D, tie_handling="ignore", smoothing=1.0, return_scores=True
        ),
    )
    add_opt(
        "serial_sign",
        lambda: rank.serial_rank(D, comparison="sign", return_scores=True),
    )
    add_opt(
        "bayes_q05_dense",
        lambda: rank.bayes(D, quantile=0.05, method="dense", return_scores=True),
    )
    add_opt(
        "kemeny_not_tieaware",
        lambda: rank.kemeny_young(D, tie_aware=False, return_scores=True),
    )
    add_opt(
        "rp_wv",
        lambda: rank.ranked_pairs(D, strength="winning_votes", return_scores=True),
    )
    add_opt("nanson_min", lambda: rank.nanson(D, rank_ties="min", return_scores=True))
    add_opt("pagerank_damp", lambda: rank.pagerank(D, damping=0.6, return_scores=True))
    add_opt(
        "bt_dense", lambda: rank.bradley_terry(D, method="dense", return_scores=True)
    )
    add_opt("pl_avg", lambda: rank.plackett_luce(D, method="avg", return_scores=True))

    return {
        "shared_inputs": {
            "ordered_binary_small_R": ordered_binary_small_R.tolist(),
            "ordered_binary_matrix": ordered_binary_matrix.tolist(),
            "tie_heavy_R": tie_heavy_R.tolist(),
            "distinct_R": distinct_R.tolist(),
            "R_multi": R_multi.tolist(),
            "w": w.tolist(),
            "R0_shared": R0_shared.tolist(),
        },
        "cases": cases,
        "option_cases": option_cases,
    }


def main():
    fx = build_fixtures()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fx, indent=2))
    print(f"wrote {OUT} with {len(fx['cases'])} cases")


if __name__ == "__main__":
    main()
