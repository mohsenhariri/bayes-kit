import numpy as np
import pytest

from scorio import rank


def test_inverse_difficulty_smoke_and_ordering(
    ordered_binary_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, scores = rank_assertions.assert_ranking_and_scores(
        rank.inverse_difficulty(ordered_binary_R, return_scores=True)
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_inverse_difficulty_clip_validation(ordered_binary_R: np.ndarray) -> None:
    with pytest.raises(ValueError, match="clip_range must satisfy 0 < low < high <= 1"):
        rank.inverse_difficulty(ordered_binary_R, clip_range=(0.9, 0.5))


def test_elo_smoke_and_tie_handling_paths(
    ordered_binary_R: np.ndarray,
    tie_heavy_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, scores = rank_assertions.assert_ranking_and_scores(
        rank.elo(ordered_binary_R, return_scores=True)
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)

    _, scores_skip = rank.elo(tie_heavy_R, tie_handling="skip", return_scores=True)
    _, scores_draw = rank.elo(tie_heavy_R, tie_handling="draw", return_scores=True)
    assert not np.allclose(scores_skip, scores_draw)


def test_elo_validation_errors(ordered_binary_R: np.ndarray) -> None:
    with pytest.raises(ValueError, match="K must be a positive finite scalar"):
        rank.elo(ordered_binary_R, K=0.0)

    with pytest.raises(ValueError, match="tie_handling must be one of"):
        rank.elo(ordered_binary_R, tie_handling="invalid")


def test_glicko_smoke_and_return_deviation_branch(
    ordered_binary_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, ratings, rd = rank.glicko(
        ordered_binary_R,
        return_scores=True,
        return_deviation=True,
    )
    rank_assertions.assert_ranking(ranking, expected_len=ordered_binary_R.shape[0])
    rank_assertions.assert_scores(ratings, expected_len=ordered_binary_R.shape[0])
    assert rd.shape == (ordered_binary_R.shape[0],)
    assert np.all(np.isfinite(rd))
    assert np.all(rd > 0.0)
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)


def test_glicko_validation_errors(ordered_binary_R: np.ndarray) -> None:
    with pytest.raises(ValueError, match="initial_rd must be > 0 and finite"):
        rank.glicko(ordered_binary_R, initial_rd=0.0)

    with pytest.raises(ValueError, match="tie_handling must be one of"):
        rank.glicko(ordered_binary_R, tie_handling="invalid")


def test_trueskill_smoke_and_tie_handling_paths(
    ordered_binary_R: np.ndarray,
    tie_heavy_R: np.ndarray,
    rank_assertions,
) -> None:
    ranking, scores = rank_assertions.assert_ranking_and_scores(
        rank.trueskill(ordered_binary_R, return_scores=True)
    )
    rank_assertions.assert_ordering_sanity(ranking, best_idx=0, worst_idx=3)

    _, scores_skip = rank.trueskill(
        tie_heavy_R,
        tie_handling="skip",
        draw_margin=0.0,
        return_scores=True,
    )
    _, scores_draw = rank.trueskill(
        tie_heavy_R,
        tie_handling="draw",
        draw_margin=0.1,
        return_scores=True,
    )
    assert not np.allclose(scores_skip, scores_draw)


def test_trueskill_validation_errors(ordered_binary_R: np.ndarray) -> None:
    with pytest.raises(
        ValueError, match="draw_margin must be a nonnegative finite scalar"
    ):
        rank.trueskill(ordered_binary_R, draw_margin=-0.1)

    with pytest.raises(ValueError, match="tie_handling must be one of"):
        rank.trueskill(ordered_binary_R, tie_handling="invalid")
