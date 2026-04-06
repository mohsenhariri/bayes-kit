import numpy as np
import pytest

from scorio import rank


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.pagerank, {}),
        (rank.spectral, {}),
        (rank.alpharank, {"population_size": 20, "max_iter": 20_000}),
        (rank.nash, {}),
        (rank.rank_centrality, {}),
        (rank.serial_rank, {}),
        (rank.hodge_rank, {}),
    ],
)
def test_graph_seriation_hodge_smoke_and_ordering(
    ordered_binary_R: np.ndarray,
    rank_assertions,
    fn,
    kwargs: dict,
) -> None:
    ranking, _ = rank_assertions.assert_ranking_and_scores(
        fn(ordered_binary_R, return_scores=True, **kwargs)
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)


def test_nash_return_equilibrium_branch(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, scores, equilibrium = rank.nash(
        ordered_binary_small_R,
        return_scores=True,
        return_equilibrium=True,
    )
    rank_assertions.assert_ranking(
        ranking, expected_len=ordered_binary_small_R.shape[0]
    )
    rank_assertions.assert_scores(scores, expected_len=ordered_binary_small_R.shape[0])
    assert equilibrium.shape == (ordered_binary_small_R.shape[0],)
    assert np.all(np.isfinite(equilibrium))
    assert np.all(equilibrium >= 0.0)
    assert float(np.sum(equilibrium)) == pytest.approx(1.0)


def test_hodge_rank_return_diagnostics_branch(
    ordered_binary_small_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, scores, diagnostics = rank.hodge_rank(
        ordered_binary_small_R,
        pairwise_stat="log_odds",
        weight_method="decisive",
        return_scores=True,
        return_diagnostics=True,
    )
    rank_assertions.assert_ranking(
        ranking, expected_len=ordered_binary_small_R.shape[0]
    )
    rank_assertions.assert_scores(scores, expected_len=ordered_binary_small_R.shape[0])
    assert set(diagnostics) == {"residual_l2", "relative_residual_l2"}
    assert np.isfinite(float(diagnostics["residual_l2"]))
    assert np.isfinite(float(diagnostics["relative_residual_l2"]))


def test_graph_seriation_hodge_validation_errors(
    ordered_binary_small_R: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"damping must be in \(0, 1\)"):
        rank.pagerank(ordered_binary_small_R, damping=1.0)

    with pytest.raises(ValueError, match="teleport must have shape"):
        rank.pagerank(ordered_binary_small_R, teleport=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="alpha must be >= 0"):
        rank.alpharank(ordered_binary_small_R, alpha=-0.1)

    with pytest.raises(ValueError, match='solver must be "lp"'):
        rank.nash(ordered_binary_small_R, solver="bad")

    with pytest.raises(ValueError, match='tie_handling must be "ignore" or "half"'):
        rank.rank_centrality(ordered_binary_small_R, tie_handling="bad")

    with pytest.raises(ValueError, match='comparison must be "prob_diff" or "sign"'):
        rank.serial_rank(ordered_binary_small_R, comparison="bad")

    with pytest.raises(
        ValueError, match='pairwise_stat must be one of: "binary", "log_odds"'
    ):
        rank.hodge_rank(ordered_binary_small_R, pairwise_stat="bad")
