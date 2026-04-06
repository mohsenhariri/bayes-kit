from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class RankAssertionHelper:
    def assert_ranking(
        self, ranking: np.ndarray, expected_len: int | None = None
    ) -> None:
        arr = np.asarray(ranking, dtype=float)
        assert arr.ndim == 1
        if expected_len is not None:
            assert arr.shape == (expected_len,)

        L = int(arr.shape[0])
        assert L >= 2
        assert np.all(np.isfinite(arr))
        assert float(np.min(arr)) == pytest.approx(1.0)
        assert np.all(arr >= 1.0)
        assert np.all(arr <= float(L))

    def assert_scores(self, scores: np.ndarray, expected_len: int) -> None:
        arr = np.asarray(scores, dtype=float)
        assert arr.shape == (expected_len,)
        assert np.all(np.isfinite(arr))

    def assert_ranking_and_scores(
        self,
        out: tuple[np.ndarray, np.ndarray],
        expected_len: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(out, tuple)
        assert len(out) >= 2

        ranking = np.asarray(out[0])
        scores = np.asarray(out[1])

        L = int(ranking.shape[0])
        self.assert_ranking(ranking, expected_len=expected_len)
        self.assert_scores(scores, expected_len=L)

        eps = 1e-12
        for i in range(L):
            for j in range(L):
                if scores[i] > scores[j] + eps:
                    assert ranking[i] <= ranking[j] + eps
                elif scores[i] < scores[j] - eps:
                    assert ranking[i] >= ranking[j] - eps

        return ranking, scores

    def assert_ordering_sanity(
        self,
        ranking: np.ndarray,
        best_idx: int = 0,
        worst_idx: int = -1,
    ) -> None:
        arr = np.asarray(ranking, dtype=float)
        assert arr[best_idx] == np.min(arr)
        assert arr[worst_idx] == np.max(arr)


@pytest.fixture(scope="session")
def rank_assertions() -> RankAssertionHelper:
    return RankAssertionHelper()


@pytest.fixture(scope="session")
def ordered_binary_R(top_p_task_aime25: np.ndarray) -> np.ndarray:
    raw0 = top_p_task_aime25[0, :24, :10]
    raw1 = top_p_task_aime25[1, :24, :10]
    raw2 = top_p_task_aime25[2, :24, :10]

    best = np.maximum(raw0, raw1)
    mid_high = raw0.copy()
    mid_low = np.minimum(mid_high, raw1)
    worst = np.minimum(mid_low, raw2)

    R = np.stack([best, mid_high, mid_low, worst], axis=0).astype(int, copy=False)

    means = R.mean(axis=(1, 2))
    assert np.all(means[:-1] >= means[1:])
    return R


@pytest.fixture(scope="session")
def ordered_binary_small_R(ordered_binary_R: np.ndarray) -> np.ndarray:
    return ordered_binary_R[:, :10, :5]


@pytest.fixture(scope="session")
def ordered_binary_matrix(ordered_binary_small_R: np.ndarray) -> np.ndarray:
    return ordered_binary_small_R[:, :, 0]


@pytest.fixture(scope="session")
def tie_heavy_R(top_p_task_aime25: np.ndarray) -> np.ndarray:
    base = top_p_task_aime25[4, :6, :4]
    return np.stack(
        [
            base,
            base.copy(),
            np.roll(base, shift=1, axis=1),
            1 - base,
        ],
        axis=0,
    ).astype(int, copy=False)


@pytest.fixture(scope="session")
def equal_information_R(top_p_task_aime25: np.ndarray) -> np.ndarray:
    base = top_p_task_aime25[5, :7, :5]
    return np.repeat(base[np.newaxis, :, :], repeats=4, axis=0)


@pytest.fixture(scope="session")
def multiclass_rank_data(
    top_p_task_aime25: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = (top_p_task_aime25[0:3, :10, :7] + top_p_task_aime25[3:6, :10, :7]).astype(
        int, copy=False
    )
    w = np.array([0.0, 0.5, 1.0], dtype=float)
    R0_shared = (top_p_task_aime25[6, :10, :3] + top_p_task_aime25[7, :10, :3]).astype(
        int, copy=False
    )
    R0_per_model = (
        top_p_task_aime25[8:11, :10, :3] + top_p_task_aime25[11:14, :10, :3]
    ).astype(int, copy=False)

    assert np.all((0 <= R) & (R <= 2))
    assert np.all((0 <= R0_shared) & (R0_shared <= 2))
    assert np.all((0 <= R0_per_model) & (R0_per_model <= 2))

    return R, w, R0_shared, R0_per_model
