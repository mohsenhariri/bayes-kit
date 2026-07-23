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


def test_nash_equilibrium_is_label_invariant_across_accuracy_maximizers() -> None:
    R = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 0, 1],
        ]
    )
    ranking, scores, equilibrium = rank.nash(
        R,
        score_type="equilibrium",
        return_scores=True,
        return_equilibrium=True,
    )
    np.testing.assert_array_equal(ranking, [1, 1, 3])
    np.testing.assert_array_equal(scores, [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(equilibrium, scores)


def test_nash_does_not_fabricate_an_equilibrium_after_solver_failure(
    monkeypatch,
) -> None:
    from scorio.rank import graph

    class FailedResult:
        status = 2
        x = None
        message = "infeasible test result"

    monkeypatch.setattr(graph, "linprog", lambda *args, **kwargs: FailedResult())
    R = np.array([[1, 1, 0], [0, 0, 1]])
    with pytest.raises(RuntimeError, match="infeasible test result"):
        graph.nash(R)


def test_pagerank_preserves_pairwise_strength_and_strict_dominance() -> None:
    two_model_R = np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
        ]
    )
    ranking, scores = rank.pagerank(two_model_R, return_scores=True)

    assert ranking.tolist() == [1, 2]
    assert scores[0] > scores[1]
    assert not np.allclose(scores, 0.5)

    dominance_chain_R = np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    chain_ranking, chain_scores = rank.pagerank(dominance_chain_R, return_scores=True)

    assert chain_ranking.tolist() == [1, 2, 3]
    assert np.all(np.diff(chain_scores) < 0.0)


def test_spectral_uses_laplace_smoothed_keener_perron_scores() -> None:
    R = np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    ranking, scores = rank.spectral(R, return_scores=True)

    assert ranking.tolist() == [1, 2, 3]
    assert scores == pytest.approx(np.array([19.0, 13.0, 10.0]) / 42.0)

    _, rank_centrality_scores = rank.rank_centrality(R, return_scores=True)
    assert not np.allclose(scores, rank_centrality_scores)


def test_spectral_is_permutation_equivariant_and_uniform_for_equal_models() -> None:
    R = np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    ranking, scores = rank.spectral(R, return_scores=True)
    permutation = np.array([2, 0, 1])
    permuted_ranking, permuted_scores = rank.spectral(
        R[permutation], return_scores=True
    )

    assert permuted_ranking.tolist() == ranking[permutation].tolist()
    assert permuted_scores == pytest.approx(scores[permutation])

    equal_R = np.repeat(np.array([[1, 0, 1, 0]]), repeats=3, axis=0)
    equal_ranking, equal_scores = rank.spectral(equal_R, return_scores=True)
    assert equal_ranking.tolist() == [1, 1, 1]
    assert equal_scores == pytest.approx(np.full(3, 1.0 / 3.0))


def test_rank_centrality_ignore_requires_directed_connectivity() -> None:
    one_way_chain_R = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
        ]
    )

    with pytest.raises(ValueError, match="strongly connected"):
        rank.rank_centrality(one_way_chain_R, tie_handling="ignore")
    with pytest.raises(ValueError, match="strongly connected"):
        rank.rank_centrality(np.zeros((3, 2), dtype=int), tie_handling="ignore")

    smoothed_ranking = rank.rank_centrality(
        one_way_chain_R, tie_handling="ignore", smoothing=1e-3
    )
    teleported_ranking = rank.rank_centrality(
        one_way_chain_R, tie_handling="ignore", teleport=0.05
    )
    assert smoothed_ranking.tolist() == [3, 2, 1]
    assert teleported_ranking.tolist() == [3, 2, 1]

    strongly_connected_R = np.eye(3, dtype=int)
    connected_ranking, connected_scores = rank.rank_centrality(
        strongly_connected_R,
        tie_handling="ignore",
        return_scores=True,
    )
    assert connected_ranking.tolist() == [1, 1, 1]
    assert connected_scores == pytest.approx(np.full(3, 1.0 / 3.0))


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
