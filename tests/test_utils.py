"""Comprehensive tests for scorio.utils module."""

import math

import numpy as np
import pytest
from scipy.stats import kendalltau, norm, spearmanr, weightedtau

from scorio import eval as scorio_eval
from scorio.utils import (
    blocks_from_rank_list,
    comb_rank_lex,
    comb_unrank_lex,
    compare_rankings,
    lehmer_hash,
    lehmer_unhash,
    ordered_bell,
    rank_scores,
    ranking_hash,
    unhash_ranking,
)


class TestRankScores:
    def test_basic_competition_ranking(self) -> None:
        scores = [95.0, 87.5, 87.5, 80.0, 75.0]
        ranks = rank_scores(scores)
        np.testing.assert_array_equal(ranks["competition"], [1, 2, 2, 4, 5])
        np.testing.assert_array_equal(ranks["competition_max"], [1, 3, 3, 4, 5])
        np.testing.assert_array_equal(ranks["dense"], [1, 2, 2, 3, 4])
        np.testing.assert_allclose(ranks["avg"], [1.0, 2.5, 2.5, 4.0, 5.0])

    def test_no_ties(self) -> None:
        scores = [10.0, 8.0, 6.0, 4.0, 2.0]
        ranks = rank_scores(scores)
        np.testing.assert_array_equal(ranks["competition"], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(ranks["dense"], [1, 2, 3, 4, 5])

    def test_all_tied(self) -> None:
        scores = [5.0, 5.0, 5.0]
        ranks = rank_scores(scores)
        np.testing.assert_array_equal(ranks["competition"], [1, 1, 1])
        np.testing.assert_array_equal(ranks["competition_max"], [3, 3, 3])
        np.testing.assert_array_equal(ranks["dense"], [1, 1, 1])
        np.testing.assert_allclose(ranks["avg"], [2.0, 2.0, 2.0])

    def test_higher_scores_get_lower_ranks(self) -> None:
        scores = [1.0, 3.0, 2.0]
        ranks = rank_scores(scores)
        assert (
            ranks["competition"][1] < ranks["competition"][2] < ranks["competition"][0]
        )

    def test_tolerance_groups_near_equal_scores(self) -> None:
        scores = [10.0, 10.0 + 1e-14, 5.0]
        ranks = rank_scores(scores, tol=1e-12)
        assert ranks["competition"][0] == ranks["competition"][1]

    def test_1d_validation(self) -> None:
        with pytest.raises(ValueError, match="1D sequence"):
            rank_scores(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_uncertainty_aware_ties_zscore(self) -> None:
        scores = [10.0, 9.5, 5.0]
        sigmas = [1.0, 1.0, 0.1]  # High uncertainty between top two
        ranks = rank_scores(scores, sigmas_in_id_order=sigmas, confidence=0.95)

        assert "competition_ci" in ranks
        assert "dense_ci" in ranks
        # Top two should be tied due to overlapping uncertainty
        assert ranks["dense_ci"][0] == ranks["dense_ci"][1]
        # Third should be distinct
        assert ranks["dense_ci"][2] > ranks["dense_ci"][0]

    def test_uncertainty_aware_ties_ci_overlap(self) -> None:
        scores = [10.0, 9.5, 5.0]
        sigmas = [1.0, 1.0, 0.1]
        ranks = rank_scores(
            scores,
            sigmas_in_id_order=sigmas,
            confidence=0.95,
            ci_tie_method="ci_overlap_adjacent",
        )
        assert "competition_ci" in ranks
        assert ranks["dense_ci"][0] == ranks["dense_ci"][1]

    def test_no_ci_keys_without_sigmas(self) -> None:
        scores = [10.0, 5.0, 1.0]
        ranks = rank_scores(scores)
        assert "competition_ci" not in ranks

    def test_sigmas_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            rank_scores([1.0, 2.0, 3.0], sigmas_in_id_order=[0.1, 0.2])

    def test_unknown_ci_tie_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ci_tie_method"):
            rank_scores([1.0, 2.0], sigmas_in_id_order=[0.1, 0.1], ci_tie_method="bad")

    def test_with_real_bayes_scores(self, top_p_data: dict[str, np.ndarray]) -> None:
        """Rank real model scores from simulation data."""
        R = top_p_data["aime25"]
        L = R.shape[0]
        scores_arr = np.empty(L)
        sigmas_arr = np.empty(L)
        for i in range(L):
            scores_arr[i], sigmas_arr[i] = scorio_eval.bayes(R[i])

        ranks = rank_scores(scores_arr, sigmas_in_id_order=sigmas_arr)

        # Basic properties
        for key in ["competition", "competition_max", "dense", "avg"]:
            assert ranks[key].shape == (L,)
            assert float(np.min(ranks[key])) == pytest.approx(1.0)
            assert np.all(ranks[key] >= 1)
            assert np.all(ranks[key] <= L)

        # CI-aware ranks should also be present
        for key in ["competition_ci", "competition_max_ci", "dense_ci", "avg_ci"]:
            assert key in ranks
            assert ranks[key].shape == (L,)

        # Best scorer should have rank 1
        best = int(np.argmax(scores_arr))
        assert ranks["competition"][best] == 1


class TestCompareRankings:
    def test_identical_rankings(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 5]
        result = compare_rankings(a, b, method="all")
        tau, _ = result["kendalltau"]
        rho, _ = result["spearmanr"]
        assert tau == pytest.approx(1.0)
        assert rho == pytest.approx(1.0)
        assert result["fraction_mismatched"] == pytest.approx(0.0)
        assert result["max_disp"] == pytest.approx(0.0)

    def test_reversed_rankings(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [5, 4, 3, 2, 1]
        result = compare_rankings(a, b, method="all")
        tau, _ = result["kendalltau"]
        rho, _ = result["spearmanr"]
        assert tau == pytest.approx(-1.0)
        assert rho == pytest.approx(-1.0)
        assert result["fraction_mismatched"] == pytest.approx(0.8)

    def test_single_method_kendall(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [1, 3, 2, 4, 5]
        tau, pval = compare_rankings(a, b, method="kendall")
        assert tau == pytest.approx(0.8)
        assert 0.0 <= pval <= 1.0

    def test_single_method_spearman(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [1, 3, 2, 4, 5]
        rho, pval = compare_rankings(a, b, method="spearman")
        assert np.isfinite(rho)
        assert 0.0 <= pval <= 1.0

    def test_single_method_weighted_kendall(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [1, 3, 2, 4, 5]
        wtau, pval = compare_rankings(a, b, method="weighted_kendall")
        assert np.isfinite(wtau)

    def test_matches_scipy_directly(self) -> None:
        a = np.array([1, 3, 2, 5, 4])
        b = np.array([2, 1, 3, 4, 5])
        result = compare_rankings(a, b, method="all")

        tau_expected = kendalltau(a, b)
        rho_expected = spearmanr(a, b)

        assert result["kendalltau"][0] == pytest.approx(float(tau_expected.statistic))
        assert result["spearmanr"][0] == pytest.approx(float(rho_expected.statistic))

    def test_fraction_mismatched_manual(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 4, 3, 5]
        result = compare_rankings(a, b, method="all")
        # positions 2 and 3 differ
        assert result["fraction_mismatched"] == pytest.approx(2.0 / 5.0)

    def test_max_displacement(self) -> None:
        a = [1, 2, 3, 4, 5]
        b = [5, 2, 3, 4, 1]
        result = compare_rankings(a, b, method="all")
        # max displacement: |5-1|=4 or |1-5|=4, normalized by n-1=4
        assert result["max_disp"] == pytest.approx(4.0 / 4.0)

    def test_different_length_raises(self) -> None:
        with pytest.raises(ValueError, match="same non-zero length"):
            compare_rankings([1, 2], [1, 2, 3])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="same non-zero length"):
            compare_rankings([], [])

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            compare_rankings(["a", "b"], ["c", "d"])

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN or inf"):
            compare_rankings([1.0, float("nan")], [1.0, 2.0])

    def test_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN or inf"):
            compare_rankings([1.0, float("inf")], [1.0, 2.0])

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            compare_rankings([1, 2], [1, 2], method="bad")

    def test_with_real_rank_comparison(self, top_p_data: dict[str, np.ndarray]) -> None:
        """Compare rankings from two different methods on real data."""
        from scorio import rank

        R = top_p_data["aime25"][:6, :10, :12]
        rank_avg = rank.avg(R)
        rank_bayes = rank.bayes(R)

        result = compare_rankings(rank_avg, rank_bayes, method="all")
        tau, _ = result["kendalltau"]
        rho, _ = result["spearmanr"]
        # Avg and Bayes should be strongly correlated on same data
        assert tau > 0.5
        assert rho > 0.5


class TestLehmerHash:
    def test_identity_permutation(self) -> None:
        assert lehmer_hash([0, 1, 2]) == 0

    def test_reverse_permutation(self) -> None:
        assert lehmer_hash([2, 1, 0]) == 5

    def test_known_values(self) -> None:
        assert lehmer_hash([0, 2, 1]) == 1
        assert lehmer_hash([1, 0, 2]) == 2
        assert lehmer_hash([1, 2, 0]) == 3
        assert lehmer_hash([2, 0, 1]) == 4

    def test_all_permutations_of_3_are_unique(self) -> None:
        from itertools import permutations

        hashes = set()
        for p in permutations(range(3)):
            h = lehmer_hash(list(p))
            assert h not in hashes
            hashes.add(h)
        assert hashes == set(range(6))

    def test_all_permutations_of_4_are_unique(self) -> None:
        from itertools import permutations

        hashes = set()
        for p in permutations(range(4)):
            h = lehmer_hash(list(p))
            assert h not in hashes
            hashes.add(h)
        assert hashes == set(range(24))

    def test_non_integer_raises(self) -> None:
        with pytest.raises(TypeError, match="permutation of integers"):
            lehmer_hash([0.5, 1.5, 2.5])

    def test_not_permutation_raises(self) -> None:
        with pytest.raises(ValueError, match="permutation of 0..n-1"):
            lehmer_hash([0, 0, 1])


class TestLehmerUnhash:
    def test_identity(self) -> None:
        assert lehmer_unhash(0, 3) == [0, 1, 2]

    def test_reverse(self) -> None:
        assert lehmer_unhash(5, 3) == [2, 1, 0]

    def test_known_values(self) -> None:
        assert lehmer_unhash(1, 3) == [0, 2, 1]
        assert lehmer_unhash(2, 3) == [1, 0, 2]
        assert lehmer_unhash(3, 3) == [1, 2, 0]
        assert lehmer_unhash(4, 3) == [2, 0, 1]

    def test_roundtrip_n3(self) -> None:
        for h in range(6):
            perm = lehmer_unhash(h, 3)
            assert lehmer_hash(perm) == h

    def test_roundtrip_n4(self) -> None:
        for h in range(24):
            perm = lehmer_unhash(h, 4)
            assert lehmer_hash(perm) == h

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in range"):
            lehmer_unhash(6, 3)  # 3! = 6, max valid is 5
        with pytest.raises(ValueError, match="must be in range"):
            lehmer_unhash(-1, 3)


class TestOrderedBell:
    def test_known_fubini_numbers(self) -> None:
        # OEIS A000670: 1, 1, 3, 13, 75, 541, 4683
        F = ordered_bell(6)
        assert F == [1, 1, 3, 13, 75, 541, 4683]

    def test_F0_is_1(self) -> None:
        F = ordered_bell(0)
        assert F == [1]


class TestRankingHash:
    def test_no_ties(self) -> None:
        assert ranking_hash([0, 1, 2]) == 0

    def test_known_values_n3(self) -> None:
        # From docstring examples
        assert ranking_hash([0, 1, 1]) == 2
        assert ranking_hash([0, 0, 1]) == 9
        assert ranking_hash([0, 0, 0]) == 12
        assert ranking_hash([1, 0, 1]) == 5

    def test_all_rankings_of_3_are_unique(self) -> None:
        """Every hash from 0..F(3)-1=12 roundtrips, confirming all 13 are reachable."""
        F = ordered_bell(3)
        assert F[3] == 13
        hashes = set()
        for h in range(F[3]):
            ranking = unhash_ranking(h, 3)
            r0 = [x - 1 for x in ranking]
            computed_h = ranking_hash(r0)
            assert computed_h == h
            hashes.add(computed_h)
        assert len(hashes) == 13

    def test_roundtrip_n3(self) -> None:
        """Every hash from 0..F(3)-1=12 should roundtrip."""
        F = ordered_bell(3)
        for h in range(F[3]):
            ranking = unhash_ranking(h, 3)
            assert ranking_hash([r - 1 for r in ranking]) == h

    def test_roundtrip_n4(self) -> None:
        """Every hash from 0..F(4)-1=74 should roundtrip."""
        F = ordered_bell(4)
        for h in range(F[4]):
            ranking = unhash_ranking(h, 4)
            assert ranking_hash([r - 1 for r in ranking]) == h

    def test_different_rankings_different_hashes(self) -> None:
        h1 = ranking_hash([0, 1, 2])
        h2 = ranking_hash([0, 0, 1])
        h3 = ranking_hash([1, 0, 1])
        assert h1 != h2
        assert h2 != h3
        assert h1 != h3


class TestUnhashRanking:
    def test_first_ranking(self) -> None:
        assert unhash_ranking(0, 3) == [1, 2, 3]

    def test_known_values(self) -> None:
        assert unhash_ranking(2, 3) == [1, 2, 2]
        assert unhash_ranking(9, 3) == [1, 1, 3]
        assert unhash_ranking(12, 3) == [1, 1, 1]
        assert unhash_ranking(5, 3) == [2, 1, 2]

    def test_out_of_range_raises(self) -> None:
        F = ordered_bell(3)
        with pytest.raises(ValueError, match="h out of range"):
            unhash_ranking(F[3], 3)
        with pytest.raises(ValueError, match="h out of range"):
            unhash_ranking(-1, 3)

    def test_returns_competition_format(self) -> None:
        """Unhashed rankings use competition (min-rank with gaps) format."""
        r = unhash_ranking(2, 3)
        # [1, 2, 2] - items 1,2 tied at rank 2, item 0 at rank 1
        assert min(r) == 1
        # Gaps: after tie of 2, next rank is 2+1=3 (not present since all covered)


class TestCombRankUnrank:
    def test_first_combination(self) -> None:
        assert comb_rank_lex([0, 1], 4, 2) == 0

    def test_last_combination(self) -> None:
        assert comb_rank_lex([2, 3], 4, 2) == 5

    def test_roundtrip(self) -> None:
        n, k = 5, 3
        total = math.comb(n, k)
        for r in range(total):
            combo = comb_unrank_lex(r, n, k)
            assert comb_rank_lex(combo, n, k) == r

    def test_all_combinations_covered(self) -> None:
        from itertools import combinations

        n, k = 5, 2
        combos = list(combinations(range(n), k))
        ranks = [comb_rank_lex(list(c), n, k) for c in combos]
        assert sorted(ranks) == list(range(len(combos)))

    def test_empty_combination(self) -> None:
        assert comb_unrank_lex(0, 5, 0) == []

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            comb_unrank_lex(10, 4, 2)  # C(4,2)=6, max valid is 5


class TestBlocksFromRankList:
    def test_no_ties(self) -> None:
        blocks = blocks_from_rank_list([1, 2, 3])
        assert blocks == [[0], [1], [2]]

    def test_all_tied(self) -> None:
        blocks = blocks_from_rank_list([1, 1, 1])
        assert blocks == [[0, 1, 2]]

    def test_some_ties(self) -> None:
        blocks = blocks_from_rank_list([1, 2, 2, 4])
        assert blocks == [[0], [1, 2], [3]]

    def test_empty(self) -> None:
        assert blocks_from_rank_list([]) == []

    def test_reversed_order(self) -> None:
        blocks = blocks_from_rank_list([3, 2, 1])
        assert blocks == [[2], [1], [0]]

    def test_float_tolerance(self) -> None:
        blocks = blocks_from_rank_list([1.0, 1.0 + 1e-14, 2.0], tol=1e-12)
        assert len(blocks) == 2
        assert blocks[0] == [0, 1]

    def test_with_real_rank_output(self, top_p_data: dict[str, np.ndarray]) -> None:
        """Convert real ranking output to blocks."""
        from scorio import rank

        R = top_p_data["aime25"][:6, :10, :12]
        ranking = rank.avg(R)
        blocks = blocks_from_rank_list(ranking)
        # All items should be covered
        all_items = sorted(item for block in blocks for item in block)
        assert all_items == list(range(6))


def test_utils_public_api_exports() -> None:
    from scorio import utils

    expected = {
        "rank_scores",
        "compare_rankings",
        "lehmer_hash",
        "lehmer_unhash",
        "ranking_hash",
        "unhash_ranking",
    }
    assert set(utils.__all__) == expected
