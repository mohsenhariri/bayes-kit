import numpy as np
import pytest

from scorio import rank


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
