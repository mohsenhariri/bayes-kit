import numpy as np
import pytest

from scorio import rank


def test_public_rank_api_exports_have_valid_smoke_calls(
    ordered_binary_small_R: np.ndarray,
    ordered_binary_matrix: np.ndarray,
    multiclass_rank_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rank_assertions,
) -> None:
    R_multi, w, R0_shared, _ = multiclass_rank_data
    finite_mle_R = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )

    function_calls = {
        # Eval-based
        "avg": lambda: rank.avg(ordered_binary_small_R, return_scores=True),
        "bayes": lambda: rank.bayes(R_multi, w=w, R0=R0_shared, return_scores=True),
        "pass_at_k": lambda: rank.pass_at_k(
            ordered_binary_small_R, k=2, return_scores=True
        ),
        "pass_hat_k": lambda: rank.pass_hat_k(
            ordered_binary_small_R, k=2, return_scores=True
        ),
        "g_pass_at_k_tau": lambda: rank.g_pass_at_k_tau(
            ordered_binary_small_R, k=2, tau=0.7, return_scores=True
        ),
        "mg_pass_at_k": lambda: rank.mg_pass_at_k(
            ordered_binary_small_R, k=2, return_scores=True
        ),
        # Pointwise and pairwise ratings
        "inverse_difficulty": lambda: rank.inverse_difficulty(
            ordered_binary_small_R, return_scores=True
        ),
        "elo": lambda: rank.elo(ordered_binary_small_R, return_scores=True),
        "glicko": lambda: rank.glicko(ordered_binary_small_R, return_scores=True),
        "trueskill": lambda: rank.trueskill(ordered_binary_small_R, return_scores=True),
        # Bradley-Terry family
        "bradley_terry": lambda: rank.bradley_terry(
            finite_mle_R, max_iter=80, return_scores=True
        ),
        "bradley_terry_map": lambda: rank.bradley_terry_map(
            ordered_binary_small_R, prior=1.0, max_iter=80, return_scores=True
        ),
        "bradley_terry_davidson": lambda: rank.bradley_terry_davidson(
            ordered_binary_small_R, max_iter=80, return_scores=True
        ),
        "bradley_terry_davidson_map": lambda: rank.bradley_terry_davidson_map(
            ordered_binary_small_R, prior=1.0, max_iter=80, return_scores=True
        ),
        "rao_kupper": lambda: rank.rao_kupper(
            ordered_binary_small_R, tie_strength=1.1, max_iter=80, return_scores=True
        ),
        "rao_kupper_map": lambda: rank.rao_kupper_map(
            ordered_binary_small_R,
            tie_strength=1.1,
            prior=1.0,
            max_iter=80,
            return_scores=True,
        ),
        # Bayesian
        "thompson": lambda: rank.thompson(
            ordered_binary_small_R, n_samples=700, seed=7, return_scores=True
        ),
        "bayesian_mcmc": lambda: rank.bayesian_mcmc(
            ordered_binary_small_R,
            n_samples=400,
            burnin=100,
            seed=7,
            return_scores=True,
        ),
        # Voting
        "borda": lambda: rank.borda(ordered_binary_small_R, return_scores=True),
        "copeland": lambda: rank.copeland(ordered_binary_small_R, return_scores=True),
        "win_rate": lambda: rank.win_rate(ordered_binary_small_R, return_scores=True),
        "minimax": lambda: rank.minimax(ordered_binary_small_R, return_scores=True),
        "schulze": lambda: rank.schulze(ordered_binary_small_R, return_scores=True),
        "ranked_pairs": lambda: rank.ranked_pairs(
            ordered_binary_small_R, return_scores=True
        ),
        "kemeny_young": lambda: rank.kemeny_young(
            ordered_binary_small_R, time_limit=1.0, return_scores=True
        ),
        "nanson": lambda: rank.nanson(ordered_binary_small_R, return_scores=True),
        "baldwin": lambda: rank.baldwin(ordered_binary_small_R, return_scores=True),
        "majority_judgment": lambda: rank.majority_judgment(
            ordered_binary_small_R, return_scores=True
        ),
        # IRT
        "rasch": lambda: rank.rasch(
            ordered_binary_small_R, max_iter=60, return_scores=True
        ),
        "rasch_map": lambda: rank.rasch_map(
            ordered_binary_small_R, prior=1.0, max_iter=60, return_scores=True
        ),
        "rasch_2pl": lambda: rank.rasch_2pl(
            ordered_binary_small_R, max_iter=300, return_scores=True
        ),
        "rasch_2pl_map": lambda: rank.rasch_2pl_map(
            ordered_binary_small_R, prior=1.0, max_iter=300, return_scores=True
        ),
        "rasch_3pl": lambda: rank.rasch_3pl(
            ordered_binary_small_R,
            max_iter=500,
            fix_guessing=0.2,
            return_scores=True,
        ),
        "rasch_3pl_map": lambda: rank.rasch_3pl_map(
            ordered_binary_small_R,
            prior=1.0,
            max_iter=500,
            fix_guessing=0.2,
            return_scores=True,
        ),
        "rasch_mml": lambda: rank.rasch_mml(
            ordered_binary_small_R,
            max_iter=10,
            em_iter=6,
            n_quadrature=9,
            return_scores=True,
        ),
        "rasch_mml_credible": lambda: rank.rasch_mml_credible(
            ordered_binary_small_R,
            quantile=0.1,
            max_iter=10,
            em_iter=6,
            n_quadrature=9,
            return_scores=True,
        ),
        "dynamic_irt": lambda: rank.dynamic_irt(
            ordered_binary_small_R,
            variant="linear",
            max_iter=60,
            return_scores=True,
        ),
        "mirt": lambda: rank.mirt(
            ordered_binary_small_R,
            n_factors=2,
            n_quadrature=7,
            em_iter=10,
            max_iter=30,
            return_scores=True,
        ),
        # Graph / seriation / hodge
        "pagerank": lambda: rank.pagerank(ordered_binary_small_R, return_scores=True),
        "spectral": lambda: rank.spectral(ordered_binary_small_R, return_scores=True),
        "alpharank": lambda: rank.alpharank(
            ordered_binary_small_R,
            population_size=20,
            max_iter=10_000,
            return_scores=True,
        ),
        "nash": lambda: rank.nash(ordered_binary_small_R, return_scores=True),
        "rank_centrality": lambda: rank.rank_centrality(
            ordered_binary_small_R, return_scores=True
        ),
        "serial_rank": lambda: rank.serial_rank(
            ordered_binary_small_R, return_scores=True
        ),
        "hodge_rank": lambda: rank.hodge_rank(
            ordered_binary_small_R, return_scores=True
        ),
        # Listwise
        "plackett_luce": lambda: rank.plackett_luce(
            finite_mle_R, max_iter=80, return_scores=True
        ),
        "plackett_luce_map": lambda: rank.plackett_luce_map(
            ordered_binary_small_R, prior=1.0, max_iter=80, return_scores=True
        ),
        "davidson_luce": lambda: rank.davidson_luce(
            finite_mle_R, max_iter=80, return_scores=True
        ),
        "davidson_luce_map": lambda: rank.davidson_luce_map(
            ordered_binary_small_R, prior=1.0, max_iter=80, return_scores=True
        ),
        "bradley_terry_luce": lambda: rank.bradley_terry_luce(
            finite_mle_R, max_iter=80, return_scores=True
        ),
        "bradley_terry_luce_map": lambda: rank.bradley_terry_luce_map(
            ordered_binary_small_R, prior=1.0, max_iter=80, return_scores=True
        ),
    }

    class_calls = {
        "Prior": lambda: rank.Prior(),
        "GaussianPrior": lambda: rank.GaussianPrior(),
        "LaplacePrior": lambda: rank.LaplacePrior(),
        "CauchyPrior": lambda: rank.CauchyPrior(),
        "UniformPrior": lambda: rank.UniformPrior(),
        "CustomPrior": lambda: rank.CustomPrior(lambda x: float(np.sum(x**2))),
        "EmpiricalPrior": lambda: rank.EmpiricalPrior(ordered_binary_small_R),
    }

    assert (set(function_calls) | set(class_calls)) == set(rank.__all__)

    for fn in function_calls.values():
        ranking, scores = rank_assertions.assert_ranking_and_scores(fn())
        rank_assertions.assert_ordering_sanity(
            ranking,
            best_idx=0,
            worst_idx=ranking.shape[0] - 1,
        )
        rank_assertions.assert_scores(scores, expected_len=ranking.shape[0])

    for name, build in class_calls.items():
        if name == "Prior":
            with pytest.raises(TypeError):
                build()
            continue

        prior = build()
        theta = np.linspace(-0.5, 0.5, num=ordered_binary_small_R.shape[0])
        value = float(prior.penalty(theta))
        assert np.isfinite(value)
