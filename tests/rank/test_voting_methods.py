import importlib

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from scorio import rank

voting_module = importlib.import_module("scorio.rank.voting")


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (rank.borda, {}),
        (rank.copeland, {}),
        (rank.win_rate, {}),
        (rank.minimax, {"variant": "margin", "tie_policy": "half"}),
        (rank.schulze, {"tie_policy": "half"}),
        (rank.ranked_pairs, {"strength": "margin", "tie_policy": "half"}),
        (rank.kemeny_young, {"tie_policy": "half", "time_limit": 1.0}),
        (rank.nanson, {"rank_ties": "average"}),
        (rank.baldwin, {"rank_ties": "average"}),
        (rank.majority_judgment, {}),
    ],
)
def test_voting_methods_smoke_and_ordering(
    ordered_binary_R: np.ndarray,
    rank_assertions,
    fn,
    kwargs: dict,
) -> None:
    ranking, _ = rank_assertions.assert_ranking_and_scores(
        fn(ordered_binary_R, return_scores=True, **kwargs)
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)


def test_voting_option_branches(
    ordered_binary_small_R: np.ndarray, rank_assertions
) -> None:
    out_minimax = rank.minimax(
        ordered_binary_small_R,
        variant="winning_votes",
        tie_policy="ignore",
        return_scores=True,
    )
    out_ranked_pairs = rank.ranked_pairs(
        ordered_binary_small_R,
        strength="winning_votes",
        tie_policy="ignore",
        return_scores=True,
    )
    out_kemeny = rank.kemeny_young(
        ordered_binary_small_R,
        tie_policy="ignore",
        tie_aware=False,
        time_limit=1.0,
        return_scores=True,
    )

    rank_assertions.assert_ranking_and_scores(out_minimax)
    rank_assertions.assert_ranking_and_scores(out_ranked_pairs)
    rank_assertions.assert_ranking_and_scores(out_kemeny)


def test_nanson_eliminates_candidates_at_the_round_mean() -> None:
    # Five strict ballots represented as Borda grades (3 is best, 0 is worst).
    grades = np.array(
        [
            [3, 2, 0, 2, 1],
            [0, 1, 3, 0, 0],
            [2, 0, 2, 1, 3],
            [1, 3, 1, 3, 2],
        ],
        dtype=int,
    )
    R = (np.arange(3)[None, None, :] < grades[:, :, None]).astype(int)

    ranking, scores = rank.nanson(R, return_scores=True)

    # Round 1 eliminates model 1. In round 2, scores are [4, 5, 6] for
    # models [0, 2, 3], so original Nanson eliminates models 0 and 2 because
    # both are at or below the mean of 5.
    np.testing.assert_array_equal(scores, np.array([1.0, 0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(ranking, np.array([2, 4, 2, 1]))


def test_kemeny_rejects_unproven_initial_incumbent(
    ordered_binary_small_R: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def limited_milp(*args, **kwargs) -> OptimizeResult:
        return OptimizeResult(
            success=False,
            status=1,
            message="Time limit reached",
            x=np.zeros_like(args[0]),
            fun=0.0,
        )

    monkeypatch.setattr(voting_module.optimize, "milp", limited_milp)

    with pytest.raises(RuntimeError, match="did not prove an optimal solution"):
        voting_module.kemeny_young(ordered_binary_small_R)


def test_kemeny_rejects_unproven_tie_aware_subproblem(
    ordered_binary_small_R: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_milp = voting_module.optimize.milp
    call_count = 0

    def limited_after_initial(*args, **kwargs) -> OptimizeResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_milp(*args, **kwargs)
        return OptimizeResult(
            success=False,
            status=1,
            message="Time limit reached",
            x=np.zeros_like(args[0]),
            fun=0.0,
        )

    monkeypatch.setattr(voting_module.optimize, "milp", limited_after_initial)

    with pytest.raises(RuntimeError, match="did not prove an optimal subproblem"):
        voting_module.kemeny_young(ordered_binary_small_R, tie_aware=True)


def test_kemeny_accepts_proven_infeasible_reverse_subproblem(
    ordered_binary_small_R: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    rank_assertions,
) -> None:
    original_milp = voting_module.optimize.milp
    call_count = 0

    def one_infeasible_reverse(*args, **kwargs) -> OptimizeResult:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return OptimizeResult(
                success=False,
                status=2,
                message="The problem is infeasible",
                x=None,
                fun=None,
            )
        return original_milp(*args, **kwargs)

    monkeypatch.setattr(voting_module.optimize, "milp", one_infeasible_reverse)

    out = voting_module.kemeny_young(
        ordered_binary_small_R,
        tie_aware=True,
        return_scores=True,
    )
    rank_assertions.assert_ranking_and_scores(out)
    assert call_count > 2


def test_voting_validation_errors(ordered_binary_small_R: np.ndarray) -> None:
    with pytest.raises(ValueError, match="variant must be one of"):
        rank.minimax(ordered_binary_small_R, variant="bad")

    with pytest.raises(ValueError, match="strength must be one of"):
        rank.ranked_pairs(ordered_binary_small_R, strength="bad")

    with pytest.raises(ValueError, match="tie_policy must be one of"):
        rank.schulze(ordered_binary_small_R, tie_policy="bad")

    with pytest.raises(ValueError, match="time_limit must be a positive finite scalar"):
        rank.kemeny_young(ordered_binary_small_R, time_limit=0.0)

    with pytest.raises(ValueError, match='unknown method "bad"'):
        rank.nanson(ordered_binary_small_R, rank_ties="bad")
