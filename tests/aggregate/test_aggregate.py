"""Tests for scorio.aggregate answer-aggregation / selection rules."""

from __future__ import annotations

import itertools
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scorio
from scorio import agg
from scorio.aggregate._base import _is_valid, _keep_count
from scorio.aggregate.cges import _row_cges_posterior


def test_short_agg_alias_is_aggregate_module() -> None:
    assert agg is scorio.aggregate
    assert scorio.agg is scorio.aggregate


class TestMajorityVote:
    def test_basic_mode(self) -> None:
        assert agg.majority_vote(["A", "B", "A", "C", "A"]) == "A"

    def test_return_index_is_first_occurrence(self) -> None:
        assert agg.majority_vote(["A", "B", "A"], return_index=True) == ("A", 0)

    def test_tie_broken_by_first_appearance(self) -> None:
        # A and B both appear twice; B appears first.
        assert agg.majority_vote(["B", "A", "A", "B"]) == "B"

    def test_invalid_entries_ignored(self) -> None:
        assert agg.majority_vote(["A", None, "", "A", "B"]) == "A"

    def test_all_invalid_returns_sentinel(self) -> None:
        sel, idx = agg.majority_vote([None, ""], return_index=True)
        assert sel is None and idx == -1

    def test_batch_returns_object_array(self) -> None:
        out = agg.majority_vote([["A", "B", "A"], ["X", "X", "Y"]])
        assert isinstance(out, np.ndarray)
        assert out.tolist() == ["A", "X"]

    def test_integer_answer_labels(self) -> None:
        assert agg.majority_vote([3, 3, 7]) == 3


class TestWeightedMajorityVote:
    def test_sum_vs_mean_differ(self) -> None:
        answers = ["A", "A", "A", "B"]
        scores = [0.3, 0.3, 0.3, 0.85]
        assert agg.weighted_majority_vote(answers, scores) == "A"  # sum 0.9 > 0.85
        assert (
            agg.weighted_majority_vote(answers, scores, aggregate="mean") == "B"
        )  # 0.3 < 0.85

    def test_unit_scores_reduce_to_majority(self) -> None:
        answers = ["A", "B", "A", "C", "A", "B"]
        scores = [1.0] * len(answers)
        assert agg.weighted_majority_vote(answers, scores) == agg.majority_vote(answers)

    def test_return_index_is_best_member_of_winner(self) -> None:
        sel, idx = agg.weighted_majority_vote(
            ["A", "A", "B"], [0.1, 0.9, 0.5], return_index=True
        )
        assert sel == "A" and idx == 1  # highest-scoring A

    def test_tie_broken_by_first_appearance(self) -> None:
        # sums equal (B=A=1.0); A appears first.
        assert agg.weighted_majority_vote(["A", "B", "B"], [1.0, 0.5, 0.5]) == "A"

    def test_invalid_entries_ignored(self) -> None:
        assert agg.weighted_majority_vote(["A", None, "B"], [0.1, 9.0, 0.2]) == "B"

    def test_bad_aggregate_raises(self) -> None:
        with pytest.raises(ValueError):
            agg.weighted_majority_vote(["A"], [1.0], aggregate="median")


class TestBestOfN:
    def test_selects_argmax_score(self) -> None:
        assert agg.best_of_n(["A", "B", "A"], [0.1, 0.9, 0.2]) == "B"

    def test_return_index(self) -> None:
        assert agg.best_of_n(["A", "B", "A"], [0.1, 0.9, 0.2], return_index=True) == (
            "B",
            1,
        )

    def test_score_tie_picks_lowest_index(self) -> None:
        assert agg.best_of_n(["A", "B"], [0.5, 0.5], return_index=True) == ("A", 0)

    def test_invalid_answer_not_selected(self) -> None:
        # highest score is at the unparsable candidate; it must be skipped.
        assert agg.best_of_n(["A", None, "B"], [0.2, 9.0, 0.3]) == "B"

    def test_all_invalid_returns_sentinel(self) -> None:
        sel, idx = agg.best_of_n([None, ""], [1.0, 2.0], return_index=True)
        assert sel is None and idx == -1

    def test_batch(self) -> None:
        out = agg.best_of_n([["A", "B"], ["C", "D"]], [[0.9, 0.1], [0.2, 0.8]])
        assert out.tolist() == ["A", "D"]

    def test_missing_scores_raises(self) -> None:
        with pytest.raises(TypeError):
            agg.best_of_n(["A", "B"])  # type: ignore[call-arg]


def _exact_boot_mode(A: list[Any], S: list[float], m: int | None = None) -> Any:
    """Gold-standard bootstrap mode by enumerating all n**m equal-probability
    with-replacement resamples (matches the paper's Best-of-N-on-resample rule).

    Uses the same deterministic tie-breaks as the implementation: within a
    resample the winner is the highest score / lowest index; across answer
    groups the mode ties are broken by earliest appearance.
    """
    n = len(A)
    mm = max(1, math.isqrt(n)) if m is None else min(max(int(m), 1), n)
    dist: dict[Any, int] = {}
    for tup in itertools.product(range(n), repeat=mm):
        j = min(tup, key=lambda k: (-float(S[k]), k))
        dist[A[j]] = dist.get(A[j], 0) + 1
    first = {a: A.index(a) for a in dist}
    return min(dist, key=lambda a: (-dist[a], first[a]))


class TestMajorityOfTheBests:
    def test_robust_to_high_reward_outlier(self) -> None:
        # A is the majority; B has the single highest reward. BoN -> B, MoB -> A.
        answers = ["A", "A", "A", "B", "B", "C"]
        scores = [0.5, 0.55, 0.6, 0.9, 0.4, 0.3]
        assert agg.best_of_n(answers, scores) == "B"
        assert agg.majority_of_the_bests(answers, scores) == "A"

    def test_mob_alias(self) -> None:
        assert agg.mob is agg.majority_of_the_bests

    def test_single_candidate(self) -> None:
        assert agg.majority_of_the_bests(["A"], [9.0]) == "A"

    def test_weight_tie_broken_by_first_appearance(self) -> None:
        # A and D both accumulate 20/49 of the bootstrap mass; D appears first.
        A = ["D", "B", "D", "B", "A", "B", "A"]
        S = [0.6667, 0.3333, 0.8333, 0.1667, 0.5, 0.0, 1.0]
        assert agg.majority_of_the_bests(A, S) == "D"
        assert agg.majority_of_the_bests(A, S) == _exact_boot_mode(A, S)

    def test_return_index_is_best_member_of_winner(self) -> None:
        answers = ["A", "A", "A", "B", "B", "C"]
        scores = [0.5, 0.55, 0.6, 0.9, 0.4, 0.3]
        sel, idx = agg.majority_of_the_bests(answers, scores, return_index=True)
        assert sel == "A" and idx == 2  # highest-reward A

    def test_explicit_m(self) -> None:
        answers = ["A", "A", "B", "B", "C"]
        scores = [0.1, 0.2, 0.9, 0.3, 0.4]
        for m in (1, 2, 3, 5):
            assert agg.majority_of_the_bests(answers, scores, m=m) == _exact_boot_mode(
                answers, scores, m=m
            )

    def test_bad_m_raises(self) -> None:
        with pytest.raises(ValueError):
            agg.majority_of_the_bests(["A", "B"], [0.1, 0.2], m=0)

    def test_matches_exact_enumeration_random(self) -> None:
        """Closed-form (B=inf) MoB must equal the exact bootstrap mode."""
        rng = np.random.default_rng(7)
        for t in range(600):
            n = int(rng.integers(2, 8))
            A = [str(x) for x in rng.choice(list("ABCD"), size=n)]
            if t % 3 == 0:  # tied integer scores
                S = [float(x) for x in rng.integers(0, 3, size=n)]
            elif t % 3 == 1:  # distinct scores
                S = [float(x) for x in rng.permutation(np.arange(n))]
            else:  # binary rewards
                S = [float(x) for x in rng.integers(0, 2, size=n)]
            assert agg.majority_of_the_bests(A, S) == _exact_boot_mode(A, S)


def _ref_softmax(A: list[Any], S: list[float], T: float) -> Any:
    """Reference: argmax_z sum_{i:z} exp(S_i / T), ties by first appearance."""
    w: dict[Any, float] = {}
    first: dict[Any, int] = {}
    for j, (a, s) in enumerate(zip(A, S, strict=True)):
        if not _is_valid(a):
            continue
        w[a] = w.get(a, 0.0) + math.exp(s / T)
        first.setdefault(a, j)
    if not w:
        return None
    return min(w, key=lambda a: (-w[a], first[a]))


class TestSoftmaxWeightedVote:
    def test_matches_reference_random(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(300):
            n = int(rng.integers(2, 8))
            A = [str(x) for x in rng.choice(list("ABC"), size=n)]
            S = [float(x) for x in rng.normal(size=n)]
            T = float(rng.uniform(0.1, 3.0))
            assert agg.softmax_weighted_vote(A, S, temperature=T) == _ref_softmax(
                A, S, T
            )

    def test_temperature_inf_is_majority(self) -> None:
        rng = np.random.default_rng(11)
        A = [str(x) for x in rng.choice(list("ABCD"), size=25)]
        S = list(rng.normal(size=25))
        assert agg.softmax_weighted_vote(
            A, S, temperature=float("inf")
        ) == agg.majority_vote(A)

    def test_small_temperature_is_best_of_n(self) -> None:
        A = ["A", "A", "B", "C"]
        S = [0.1, 0.2, 0.9, 0.3]  # distinct scores
        assert agg.softmax_weighted_vote(A, S, temperature=1e-6) == agg.best_of_n(A, S)

    def test_return_index_is_best_member_of_winner(self) -> None:
        sel, idx = agg.softmax_weighted_vote(
            ["A", "A", "B"],
            [0.1, 0.9, 0.5],
            temperature=float("inf"),
            return_index=True,
        )
        assert sel == "A" and idx == 1  # highest-scoring A

    def test_bad_temperature_raises(self) -> None:
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                agg.softmax_weighted_vote(["A"], [1.0], temperature=bad)


def _ref_rank(A: list[Any], S: list[float], p: float) -> Any:
    """Reference: rank by -score,index; weight (n-t)^p; argmax group, first-tie."""
    valid = [j for j, a in enumerate(A) if agg._base._is_valid(a)]
    n = len(valid)
    order = sorted(valid, key=lambda j: (-S[j], j))
    w: dict[Any, float] = {}
    first: dict[Any, int] = {}
    for t, j in enumerate(order):
        w[A[j]] = w.get(A[j], 0.0) + (n - t) ** p
    for j in valid:
        first.setdefault(A[j], j)
    if not w:
        return None
    return min(w, key=lambda a: (-w[a], first[a]))


class TestRankWeightedVote:
    def test_matches_reference_random(self) -> None:
        rng = np.random.default_rng(5)
        for _ in range(300):
            n = int(rng.integers(2, 9))
            A = [str(x) for x in rng.choice(list("ABC"), size=n)]
            S = [float(x) for x in rng.permutation(np.arange(n))]  # distinct
            p = float(rng.uniform(0.0, 3.0))
            assert agg.rank_weighted_vote(A, S, p=p) == _ref_rank(A, S, p)

    def test_p_zero_is_majority(self) -> None:
        rng = np.random.default_rng(2)
        A = [str(x) for x in rng.choice(list("ABCD"), size=30)]
        S = list(rng.normal(size=30))
        assert agg.rank_weighted_vote(A, S, p=0.0) == agg.majority_vote(A)

    def test_large_p_is_best_of_n(self) -> None:
        A = ["A", "A", "A", "B"]
        S = [0.1, 0.2, 0.3, 0.9]
        assert agg.rank_weighted_vote(A, S, p=50.0) == agg.best_of_n(A, S)

    def test_very_large_p_does_not_overflow(self) -> None:
        # (n - t) ** p overflowed int**float for large n and p; both the exact
        # integer path (integer p) and the normalized float path (non-integer p)
        # must stay finite and still recover Best-of-N.
        rng = np.random.default_rng(99)
        n = 200
        A = [str(x) for x in rng.choice(list("AB"), size=n)]
        S = list(rng.permutation(np.arange(n)).astype(float))  # distinct
        assert agg.rank_weighted_vote(A, S, p=300.0) == agg.best_of_n(A, S)  # integer
        assert agg.rank_weighted_vote(A, S, p=300.5) == agg.best_of_n(
            A, S
        )  # non-integer

    def test_integer_p_ties_are_exact(self) -> None:
        # Integer p must use exact arbitrary-precision weights, so genuine
        # group-weight ties resolve by earliest appearance, not float noise.
        def exact(A: list[Any], S: list[float], p: int) -> Any:
            valid = [j for j, a in enumerate(A) if _is_valid(a)]
            n = len(valid)
            order = sorted(valid, key=lambda j: (-S[j], j))
            w: dict[Any, int] = {}
            first: dict[Any, int] = {}
            for t, j in enumerate(order):
                w[A[j]] = w.get(A[j], 0) + (n - t) ** p
            for j in valid:
                first.setdefault(A[j], j)
            return min(w, key=lambda a: (-w[a], first[a]))

        rng = np.random.default_rng(2026)
        for _ in range(400):
            n = int(rng.integers(2, 9))
            A = [str(x) for x in rng.choice(list("ABC"), size=n)]
            S = [float(x) for x in rng.integers(0, 3, size=n)]  # heavy ties
            for p in (0, 1, 2, 3):
                assert agg.rank_weighted_vote(A, S, p=float(p)) == exact(A, S, p)

    def test_invariant_to_monotone_rescale(self) -> None:
        A = ["A", "B", "A", "C", "B"]
        S = [0.1, 0.2, 0.15, 0.05, 0.9]
        S2 = [s**3 + 100.0 for s in S]  # strictly increasing transform
        assert agg.rank_weighted_vote(A, S, p=1.3) == agg.rank_weighted_vote(
            A, S2, p=1.3
        )

    def test_bad_p_raises(self) -> None:
        for bad in (-1.0, float("inf")):
            with pytest.raises(ValueError):
                agg.rank_weighted_vote(["A", "B"], [1.0, 2.0], p=bad)


class TestLogitWeightedVote:
    def test_linear_threshold_zero_equals_weighted_sum(self) -> None:
        rng = np.random.default_rng(9)
        A = [str(x) for x in rng.choice(list("ABC"), size=12)]
        S = list(rng.random(12))
        assert agg.logit_weighted_vote(
            A, S, transform="linear", threshold=0.0
        ) == agg.weighted_majority_vote(A, S, aggregate="sum")

    def test_negative_votes_flip_the_winner(self) -> None:
        # Three A-candidates, all below b=0.5, vote negative; a single strong B wins,
        # even though the raw-score weighted vote (all positive) picks A.
        A = ["A", "A", "A", "B"]
        S = [0.45, 0.45, 0.45, 0.9]
        assert agg.weighted_majority_vote(A, S) == "A"  # raw sum 1.35 > 0.9
        assert agg.logit_weighted_vote(A, S) == "B"  # A's log-odds are negative

    def test_logit_matches_manual(self) -> None:
        A = ["A", "A", "B"]
        S = [0.6, 0.6, 0.95]
        b = 0.5
        tb = math.log(b / (1 - b))
        wa = 2 * (math.log(0.6 / 0.4) - tb)
        wb = math.log(0.95 / 0.05) - tb
        expected = "A" if wa >= wb else "B"
        assert agg.logit_weighted_vote(A, S, threshold=b) == expected

    def test_logit_requires_scores_in_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            agg.logit_weighted_vote(["A", "B"], [0.5, 1.0])  # 1.0 not in (0,1)
        with pytest.raises(ValueError):
            agg.logit_weighted_vote(["A", "B"], [0.5, 2.0])

    def test_linear_allows_unbounded_scores(self) -> None:
        # No error for scores outside (0, 1) under the linear transform.
        assert (
            agg.logit_weighted_vote(
                ["A", "B"], [5.0, -3.0], transform="linear", threshold=0.0
            )
            == "A"
        )

    def test_bad_threshold_and_transform_raise(self) -> None:
        with pytest.raises(ValueError):
            agg.logit_weighted_vote(["A"], [0.5], threshold=1.0)  # not in (0,1)
        with pytest.raises(ValueError):
            agg.logit_weighted_vote(["A"], [0.5], transform="sqrt")


class TestFilteredVote:
    def test_filters_before_voting(self) -> None:
        # A is the majority (3 vs 3) but the top-3 by score are all B.
        A = ["A", "A", "A", "B", "B", "B"]
        S = [0.1, 0.2, 0.3, 0.8, 0.85, 0.9]
        assert agg.filtered_vote(A, S, keep=0.5) == "B"

    def test_keep_all_weighted_equals_weighted_sum(self) -> None:
        rng = np.random.default_rng(13)
        A = [str(x) for x in rng.choice(list("ABC"), size=10)]
        S = list(rng.random(10))
        assert agg.filtered_vote(A, S, keep=1.0, weighted=True) == (
            agg.weighted_majority_vote(A, S, aggregate="sum")
        )

    def test_keep_all_unweighted_equals_majority(self) -> None:
        rng = np.random.default_rng(14)
        A = [str(x) for x in rng.choice(list("ABC"), size=11)]
        S = list(rng.random(11))
        assert agg.filtered_vote(A, S, keep=1.0, weighted=False) == agg.majority_vote(A)

    def test_keep_one_int_is_best_of_n(self) -> None:
        A = ["A", "B", "A", "C"]
        S = [0.1, 0.9, 0.2, 0.3]
        assert agg.filtered_vote(A, S, keep=1) == agg.best_of_n(A, S)  # int count = 1

    def test_fraction_rounds_up_and_keeps_one_minimum(self) -> None:
        A = ["A", "B", "C", "D"]
        S = [0.4, 0.3, 0.2, 0.1]
        # keep=0.1 * 4 = 0.4 -> ceil -> 1 candidate (the top, A).
        assert agg.filtered_vote(A, S, keep=0.1) == "A"

    def test_invalid_entries_excluded_from_filter(self) -> None:
        A = ["A", None, "B", "B"]
        S = [0.9, 0.99, 0.5, 0.6]  # top raw score is the invalid one
        # valid = A(0.9), B(0.5), B(0.6); keep top 2 -> A(0.9), B(0.6) -> tie 1-1, A first
        assert agg.filtered_vote(A, S, keep=2, weighted=False) == "A"

    def test_bad_keep_raises(self) -> None:
        for bad in (0.0, 1.5, -1, 0):
            with pytest.raises(ValueError):
                agg.filtered_vote(["A", "B"], [0.1, 0.2], keep=bad)


class TestKeepCount:
    def test_fraction_boundary_no_off_by_one(self) -> None:
        # 0.07 * 100 == 7.000000000000001 in float; must resolve to 7, not 8.
        assert _keep_count(0.07, 100) == 7
        assert _keep_count(0.29, 100) == 29
        assert _keep_count(0.5, 6) == 3
        assert _keep_count(1.0, 6) == 6  # keep all
        assert _keep_count(0.1, 4) == 1  # ceil(0.4) rounds up, min one kept

    def test_count_semantics(self) -> None:
        assert _keep_count(3, 10) == 3
        assert _keep_count(50, 10) == 10  # capped at n
        assert _keep_count(1, 10) == 1

    def test_matches_exact_rational_over_grid(self) -> None:
        from fractions import Fraction

        for n in range(1, 60):
            for pct in range(1, 101):
                keep = pct / 100.0
                exact = max(1, -(-(Fraction(pct, 100) * n) // 1))  # ceil of exact ratio
                assert _keep_count(keep, n) == int(exact), (n, pct)


class TestBestOfMajority:
    def test_frequency_gate_beats_lone_outlier(self) -> None:
        A = ["A", "A", "A", "B"]
        S = [0.5, 0.55, 0.6, 0.99]
        assert agg.best_of_n(A, S) == "B"
        assert agg.best_of_majority(A, S, alpha=0.5) == "A"  # B (freq 1/4) gated out

    def test_alpha_zero_mean_equals_weighted_mean(self) -> None:
        rng = np.random.default_rng(21)
        A = [str(x) for x in rng.choice(list("ABC"), size=15)]
        S = list(rng.random(15))
        assert agg.best_of_majority(A, S, alpha=0.0, aggregate="mean") == (
            agg.weighted_majority_vote(A, S, aggregate="mean")
        )

    def test_alpha_zero_sum_equals_weighted_sum(self) -> None:
        rng = np.random.default_rng(22)
        A = [str(x) for x in rng.choice(list("ABC"), size=15)]
        S = list(rng.random(15))
        assert agg.best_of_majority(A, S, alpha=0.0, aggregate="sum") == (
            agg.weighted_majority_vote(A, S, aggregate="sum")
        )

    def test_empty_gate_relaxes(self) -> None:
        # alpha=1.0 with no unanimous answer -> gate empties -> relax to all valid.
        A = ["A", "A", "B"]
        S = [0.2, 0.3, 0.9]
        assert agg.best_of_majority(A, S, alpha=1.0, aggregate="max") == "B"

    def test_aggregate_max_option(self) -> None:
        # A has the single highest reward (max -> A) but B has the higher average
        # (mean -> B); both answers clear the alpha=0.5 frequency gate.
        A = ["A", "A", "B", "B"]
        S = [0.1, 0.95, 0.6, 0.7]
        assert (
            agg.best_of_majority(A, S, alpha=0.5, aggregate="max") == "A"
        )  # 0.95 > 0.7
        assert (
            agg.best_of_majority(A, S, alpha=0.5, aggregate="mean") == "B"
        )  # 0.65 > 0.525

    def test_return_index_is_best_member(self) -> None:
        A = ["A", "A", "A", "B"]
        S = [0.5, 0.55, 0.6, 0.99]
        sel, idx = agg.best_of_majority(A, S, alpha=0.5, return_index=True)
        assert sel == "A" and idx == 2  # highest-reward A

    def test_bad_args_raise(self) -> None:
        with pytest.raises(ValueError):
            agg.best_of_majority(["A"], [0.5], alpha=1.5)
        with pytest.raises(ValueError):
            agg.best_of_majority(["A"], [0.5], aggregate="median")


class TestInputHandling:
    def test_single_question_returns_scalar(self) -> None:
        assert np.ndim(agg.majority_vote(["A", "B", "A"])) == 0

    def test_batch_shape_preserved(self) -> None:
        out = agg.majority_vote([["A", "A"], ["B", "B"], ["C", "C"]])
        assert out.shape == (3,)

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            agg.best_of_n([["A", "B", "C"]], [[0.1, 0.2]])

    def test_missing_scores_raise(self) -> None:
        with pytest.raises(ValueError):
            agg.weighted_majority_vote(["A", "B"], None)  # type: ignore[arg-type]

    def test_empty_pool_raises(self) -> None:
        with pytest.raises(ValueError):
            agg.majority_vote(np.empty((2, 0), dtype=object))

    def test_three_dimensional_raises(self) -> None:
        with pytest.raises(ValueError):
            agg.majority_vote(np.zeros((2, 2, 2), dtype=object))

    def test_batch_and_loop_agree(self) -> None:
        rng = np.random.default_rng(1)
        answers = rng.choice(list("ABC"), size=(20, 6))
        scores = rng.random((20, 6))
        for fn, kw in [
            (agg.majority_vote, {}),
            (agg.best_of_n, {"scores": scores}),
            (agg.weighted_majority_vote, {"scores": scores}),
            (agg.majority_of_the_bests, {"scores": scores}),
            (agg.softmax_weighted_vote, {"scores": scores}),
            (agg.rank_weighted_vote, {"scores": scores}),
            (agg.logit_weighted_vote, {"scores": scores}),
            (agg.filtered_vote, {"scores": scores}),
            (agg.best_of_majority, {"scores": scores}),
        ]:
            batched = fn(answers, **kw)  # type: ignore[operator]
            rowwise = [
                fn(answers[i], **{k: v[i] for k, v in kw.items()})  # type: ignore[operator]
                for i in range(answers.shape[0])
            ]
            assert batched.tolist() == rowwise


class TestReturnScore:
    def test_best_of_n_returns_argmax_score(self) -> None:
        assert agg.best_of_n(["A", "B", "A"], [0.1, 0.9, 0.2], return_score=True) == (
            "B",
            0.9,
        )

    def test_best_of_n_index_then_score_order(self) -> None:
        assert agg.best_of_n(
            ["A", "B", "A"], [0.1, 0.9, 0.2], return_index=True, return_score=True
        ) == ("B", 1, 0.9)

    def test_weighted_returns_representative_score(self) -> None:
        # winner is A (sum 0.7 > B 0.5); representative is A's best member (0.4).
        assert agg.weighted_majority_vote(
            ["A", "A", "B"], [0.3, 0.4, 0.5], return_score=True
        ) == ("A", 0.4)

    def test_mob_returns_representative_score(self) -> None:
        answers = ["A", "A", "A", "B", "B", "C"]
        scores = [0.5, 0.55, 0.6, 0.9, 0.4, 0.3]
        assert agg.majority_of_the_bests(answers, scores, return_score=True) == (
            "A",
            0.6,
        )

    def test_all_invalid_score_is_nan(self) -> None:
        _, idx, score = agg.best_of_n(
            [None, ""], [1.0, 2.0], return_index=True, return_score=True
        )
        assert idx == -1
        assert math.isnan(score)

    def test_batch_scores_are_float_array_with_nan_sentinel(self) -> None:
        answers = [["A", "B", "A"], [None, "", float("nan")]]
        scores = [[0.1, 0.9, 0.2], [0.5, 0.5, 0.5]]
        sel, scr = agg.best_of_n(answers, scores, return_score=True)
        assert sel.tolist() == ["B", None]
        assert scr[0] == pytest.approx(0.9)
        assert math.isnan(scr[1])

    def test_default_return_unchanged(self) -> None:
        # opt-in flag must not perturb the bare-selection default.
        assert agg.best_of_n(["A", "B"], [0.1, 0.9]) == "B"


class TestCgesInternals:
    def test_public_api_and_other_sentinel(self) -> None:
        assert repr(agg.CGES_OTHER) == "CGES_OTHER"
        assert pickle.loads(pickle.dumps(agg.CGES_OTHER)) is agg.CGES_OTHER
        assert not hasattr(agg, "cges_posterior")

    def test_paper_formula(self) -> None:
        posterior = _row_cges_posterior(["A", "B"], [0.8, 0.6])
        assert posterior["A"] == pytest.approx(2 / 3)
        assert posterior["B"] == pytest.approx(1 / 4)
        assert posterior[agg.CGES_OTHER] == pytest.approx(1 / 12)
        assert sum(posterior.values()) == pytest.approx(1.0)

    def test_support_is_recomputed_when_new_answer_appears(self) -> None:
        before = _row_cges_posterior(["A", "A"], [0.8, 0.6])
        after = _row_cges_posterior(["A", "A", "B"], [0.8, 0.6, 0.7])
        assert list(before) == ["A", agg.CGES_OTHER]
        assert list(after) == ["A", "B", agg.CGES_OTHER]
        assert after["A"] != pytest.approx(before["A"])

    def test_invalid_answers_are_ignored(self) -> None:
        posterior = _row_cges_posterior([None, "", np.nan, "A"], [0.0, 1.0, 2.0, 0.7])
        assert posterior["A"] == pytest.approx(0.7)

    def test_all_invalid_answers(self) -> None:
        assert _row_cges_posterior([None, ""], [0.0, 1.0]) == {agg.CGES_OTHER: 1.0}

    @pytest.mark.parametrize("score", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
    def test_scores_must_be_probabilities(self, score: float) -> None:
        with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
            _row_cges_posterior(["A"], [score])

    def test_other_is_reserved(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            _row_cges_posterior([agg.CGES_OTHER], [0.8])

    def test_log_space_stays_stable(self) -> None:
        posterior = _row_cges_posterior(["A"] * 1000 + ["B"] * 1000, [0.99] * 2000)
        assert all(math.isfinite(value) for value in posterior.values())
        assert sum(posterior.values()) == pytest.approx(1.0)


class TestCgesVote:
    def test_default_returns_observed_answer(self) -> None:
        assert agg.cges_vote(["A"], [0.1]) == "A"

    def test_other_is_opt_in(self) -> None:
        assert agg.cges_vote(["A"], [0.1], allow_other=True) is agg.CGES_OTHER

    def test_return_index_and_score(self) -> None:
        assert agg.cges_vote(
            ["A", "A", "B"],
            [0.7, 0.9, 0.6],
            return_index=True,
            return_score=True,
        ) == ("A", 1, 0.9)

    def test_other_has_no_representative(self) -> None:
        answer, index, score = agg.cges_vote(
            ["A"],
            [0.1],
            allow_other=True,
            return_index=True,
            return_score=True,
        )
        assert answer is agg.CGES_OTHER and index == -1 and math.isnan(score)

    def test_empty_row_contract(self) -> None:
        answer, index, score = agg.cges_vote(
            [None, ""], [0.2, 0.8], return_index=True, return_score=True
        )
        assert answer is None and index == -1 and math.isnan(score)

    def test_batch(self) -> None:
        answers = agg.cges_vote([["A", "B"], ["X", "X"]], [[0.8, 0.6], [0.7, 0.8]])
        assert isinstance(answers, np.ndarray)
        assert answers.tolist() == ["A", "X"]

    def test_concrete_argmax_matches_logit_vote(self) -> None:
        answers = ["A", "A", "B", "C", "B"]
        scores = [0.7, 0.4, 0.8, 0.65, 0.3]
        threshold = 1 / (len(set(answers)) + 1)
        assert agg.cges_vote(answers, scores) == agg.logit_weighted_vote(
            answers, scores, threshold=threshold
        )


class TestCgesStop:
    def test_observed_answer_crosses_threshold(self) -> None:
        stop, probability = agg.cges_stop(["A"], [0.9], threshold=0.8, return_prob=True)
        assert stop is True and probability == pytest.approx(0.9)

    def test_other_does_not_stop_by_default(self) -> None:
        stop, probability = agg.cges_stop(["A"], [0.1], threshold=0.8, return_prob=True)
        assert stop is False and probability == pytest.approx(0.1)

    def test_other_can_trigger_paper_rule(self) -> None:
        stop, probability = agg.cges_stop(
            ["A"], [0.1], threshold=0.8, include_other=True, return_prob=True
        )
        assert stop is True and probability == pytest.approx(0.9)

    def test_minimum_samples_count_valid_answers(self) -> None:
        assert not agg.cges_stop([None, "A"], [0.0, 0.9], threshold=0.8, min_samples=2)

    def test_no_valid_answers(self) -> None:
        assert agg.cges_stop([None, ""], [0.0, 1.0], return_prob=True) == (
            False,
            0.0,
        )

    def test_batch_rejected(self) -> None:
        with pytest.raises(ValueError, match="1D sampling stream"):
            agg.cges_stop([["A"], ["B"]], [[0.8], [0.7]])

    @pytest.mark.parametrize("threshold", [0.0, 1.0, np.nan])
    def test_bad_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            agg.cges_stop(["A"], [0.8], threshold=threshold)

    @pytest.mark.parametrize("minimum", [0, -1, 1.5, True])
    def test_bad_min_samples(self, minimum: object) -> None:
        with pytest.raises(ValueError, match="min_samples"):
            agg.cges_stop(["A"], [0.8], min_samples=minimum)  # type: ignore[arg-type]


def _kde_logit(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.log(array) - np.log1p(-array)


def _constant_kde_calibration(probability: float) -> agg.KDEVoteCalibration:
    # Identical correct/incorrect KDEs isolate the reliability-offset behavior.
    samples = _kde_logit([0.3, 0.7])
    return agg.KDEVoteCalibration(
        correct_logits=samples,
        incorrect_logits=samples,
        correct_bandwidth=0.5,
        incorrect_bandwidth=0.5,
        bin_edges=np.array([-np.inf, np.inf]),
        bin_probability=np.array([probability]),
    )


class TestKDEVoteCalibration:
    def test_public_api_does_not_add_duplicate_parametric_fitter(self) -> None:
        assert agg.KDEVoteCalibration.__module__ == "scorio.aggregate.calibration"
        assert callable(agg.fit_kde_vote_calibration)
        assert callable(agg.kde_weighted_vote)
        assert not hasattr(agg, "fit_weighted_vote_threshold")

    def test_fit_separates_classes_and_defensively_copies(self) -> None:
        scores = np.array([0.8, 0.9, 0.1, 0.2])
        calibration = agg.fit_kde_vote_calibration(
            scores,
            [1, 1, 0, 0],
            n_bins=2,
            bandwidth=0.5,
        )
        np.testing.assert_allclose(calibration.correct_logits, _kde_logit([0.8, 0.9]))
        np.testing.assert_allclose(calibration.incorrect_logits, _kde_logit([0.1, 0.2]))
        expected = calibration.correct_logits.copy()
        scores[:2] = 0.5
        np.testing.assert_array_equal(calibration.correct_logits, expected)
        assert calibration.n_bins == 2
        assert not calibration.correct_logits.flags.writeable
        assert not calibration.bin_probability.flags.writeable

    def test_quantile_bins_calibrate_final_answer_correctness(self) -> None:
        calibration = agg.fit_kde_vote_calibration(
            [0.1, 0.2, 0.8, 0.9],
            [0, 0, 1, 1],
            n_bins=2,
            bandwidth=0.5,
        )
        np.testing.assert_array_equal(
            calibration.calibrated_probability([0.2, 0.7, 0.8, 0.95]),
            [0.0, 0.0, 1.0, 1.0],
        )

    def test_gaussian_log_density_ratio_matches_direct_formula(self) -> None:
        bandwidth = 0.4
        calibration = agg.fit_kde_vote_calibration(
            [0.7, 0.8, 0.2, 0.3],
            [1, 1, 0, 0],
            n_bins=1,
            bandwidth=bandwidth,
        )
        query_logit = float(_kde_logit([0.65])[0])

        def log_density(samples: np.ndarray) -> float:
            kernels = np.exp(-0.5 * ((query_logit - samples) / bandwidth) ** 2)
            density = np.sum(kernels) / (
                samples.size * bandwidth * math.sqrt(2.0 * math.pi)
            )
            return math.log(float(density))

        expected = log_density(calibration.correct_logits) - log_density(
            calibration.incorrect_logits
        )
        assert calibration.log_density_ratio([0.65])[0] == pytest.approx(expected)

    def test_weight_formula_includes_density_and_reliability_terms(self) -> None:
        calibration = agg.KDEVoteCalibration(
            correct_logits=_kde_logit([0.7, 0.8]),
            incorrect_logits=_kde_logit([0.2, 0.3]),
            correct_bandwidth=0.5,
            incorrect_bandwidth=0.5,
            bin_edges=np.array([-np.inf, np.inf]),
            bin_probability=np.array([0.6]),
        )
        scores = np.array([0.4, 0.7])
        expected = calibration.log_density_ratio(scores) + math.log(0.6 * 2 / 0.4)
        np.testing.assert_allclose(calibration.weights(scores, n_answers=3), expected)

    def test_weights_requires_one_response_pool(self) -> None:
        calibration = _constant_kde_calibration(0.5)
        with pytest.raises(ValueError, match="nonempty 1D"):
            calibration.weights([[0.4, 0.6]], n_answers=2)
        with pytest.raises(ValueError, match="nonempty 1D"):
            calibration.weights([], n_answers=2)

    def test_scott_bandwidth_and_scalar_numeric_bandwidth(self) -> None:
        scott = agg.fit_kde_vote_calibration(
            [0.65, 0.75, 0.85, 0.15, 0.25, 0.4],
            [1, 1, 1, 0, 0, 0],
        )
        assert scott.correct_bandwidth > 0.0
        assert scott.incorrect_bandwidth > 0.0

        explicit = agg.fit_kde_vote_calibration(
            [0.8, 0.9, 0.1, 0.2],
            [1, 1, 0, 0],
            bandwidth=np.array(0.3),
        )
        assert explicit.correct_bandwidth == pytest.approx(0.3)
        assert explicit.incorrect_bandwidth == pytest.approx(0.3)

    @pytest.mark.parametrize("n_bins", [0, -1, 1.5, True])
    def test_bad_bin_count(self, n_bins: object) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            agg.fit_kde_vote_calibration(
                [0.8, 0.9, 0.1, 0.2],
                [1, 1, 0, 0],
                n_bins=n_bins,  # type: ignore[arg-type]
                bandwidth=0.5,
            )

    @pytest.mark.parametrize("scores", [[0.0, 0.5], [1.0, 0.5], [np.nan, 0.5]])
    def test_scores_must_be_probabilities(self, scores: list[float]) -> None:
        with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
            agg.fit_kde_vote_calibration(scores, [0, 1], bandwidth=0.5)

    def test_correctness_shape_values_and_classes(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            agg.fit_kde_vote_calibration([0.8, 0.2], [1], bandwidth=0.5)
        with pytest.raises(ValueError, match="boolean or 0/1"):
            agg.fit_kde_vote_calibration([0.8, 0.2], [1, 2], bandwidth=0.5)
        with pytest.raises(ValueError, match="correct and incorrect"):
            agg.fit_kde_vote_calibration([0.8, 0.9], [1, 1], bandwidth=0.5)

    def test_raw_prm_step_matrix_is_not_a_response_score_bank(self) -> None:
        # One correctness label per response cannot align with a matrix of
        # Response labels provide one target per response, not one per step.
        with pytest.raises(ValueError, match="same shape"):
            agg.fit_kde_vote_calibration(
                [[0.9, 0.8], [0.7, 0.6]],
                [1, 0],
                bandwidth=0.5,
            )

    def test_scott_rejects_singular_class(self) -> None:
        scores = [0.8, 0.8, 0.2, 0.3]
        correct = [1, 1, 0, 0]
        with pytest.raises(ValueError, match="constant correct-class"):
            agg.fit_kde_vote_calibration(scores, correct)
        calibration = agg.fit_kde_vote_calibration(scores, correct, bandwidth=0.5)
        assert calibration.correct_bandwidth == pytest.approx(0.5)


class TestKDEWeightedVote:
    def test_paper_kde_vote_example(self) -> None:
        calibration = agg.fit_kde_vote_calibration(
            [0.8, 0.9, 0.1, 0.2],
            [1, 1, 0, 0],
            n_bins=2,
            bandwidth=0.5,
        )
        assert (
            agg.kde_weighted_vote(["A", "A", "B"], [0.2, 0.2, 0.8], calibration) == "B"
        )

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [(0.6, "A"), (0.4, "B"), (1.0, "A"), (0.0, "B")],
    )
    def test_reliability_offset_and_exact_limits(
        self, probability: float, expected: str
    ) -> None:
        calibration = _constant_kde_calibration(probability)
        assert (
            agg.kde_weighted_vote(["A", "A", "B"], [0.4, 0.5, 0.6], calibration)
            == expected
        )

    def test_tie_and_single_group(self) -> None:
        calibration = _constant_kde_calibration(0.5)
        assert agg.kde_weighted_vote(["B", "A"], [0.4, 0.6], calibration) == "B"
        assert agg.kde_weighted_vote(["A", "A"], [0.2, 0.8], calibration) == "A"

    def test_invalid_answers_follow_package_convention(self) -> None:
        calibration = _constant_kde_calibration(0.5)
        # Invalid candidates neither vote nor enter the reliability estimate.
        assert (
            agg.kde_weighted_vote(["A", None, "B"], [0.4, 9.0, 0.6], calibration) == "A"
        )
        answer, index, score = agg.kde_weighted_vote(
            [None, ""],
            [0.0, 1.0],
            calibration,
            return_index=True,
            return_score=True,
        )
        assert answer is None and index == -1 and math.isnan(score)

    def test_batch_and_return_metadata(self) -> None:
        calibration = _constant_kde_calibration(0.6)
        selected, indices, scores = agg.kde_weighted_vote(
            [["A", "A", "B"], ["X", "Y", "Y"]],
            [[0.4, 0.7, 0.6], [0.8, 0.5, 0.6]],
            calibration,
            return_index=True,
            return_score=True,
        )
        assert selected.tolist() == ["A", "Y"]
        assert indices.tolist() == [1, 2]
        np.testing.assert_allclose(scores, [0.7, 0.6])

    def test_valid_scores_and_calibration_type_are_checked(self) -> None:
        calibration = _constant_kde_calibration(0.5)
        with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
            agg.kde_weighted_vote(["A", "B"], [0.5, 1.0], calibration)
        with pytest.raises(TypeError, match="KDEVoteCalibration"):
            agg.kde_weighted_vote(
                ["A", "B"],
                [0.4, 0.6],
                object(),  # type: ignore[arg-type]
            )


@pytest.mark.parametrize(
    "rule",
    [
        agg.weighted_majority_vote,
        agg.majority_of_the_bests,
        agg.best_of_majority,
        agg.softmax_weighted_vote,
        agg.rank_weighted_vote,
        agg.logit_weighted_vote,
        agg.filtered_vote,
    ],
    ids=["weighted", "mob", "bom", "softmax", "rank", "logit", "filtered"],
)
def test_score_aware_rules_share_the_all_invalid_contract(rule: Any) -> None:
    selected, index, score = rule(
        [None, "", np.nan],
        [0.1, 0.2, 0.3],
        return_index=True,
        return_score=True,
    )
    assert selected is None
    assert index == -1
    assert math.isnan(score)


@pytest.mark.parametrize(
    ("rule", "kwargs"),
    [
        (agg.softmax_weighted_vote, {"temperature": math.inf}),
        (agg.rank_weighted_vote, {"p": 0.0}),
        (
            agg.logit_weighted_vote,
            {"transform": "linear", "threshold": 0.0},
        ),
        (agg.filtered_vote, {"keep": 1.0, "weighted": True}),
        (agg.best_of_majority, {"alpha": 0.0, "aggregate": "sum"}),
    ],
    ids=["softmax", "rank", "logit", "filtered", "bom"],
)
def test_score_aware_rules_return_the_winners_best_member(
    rule: Any, kwargs: dict[str, Any]
) -> None:
    assert rule(
        ["A", "A", "B"],
        [0.3, 0.4, 0.5],
        return_index=True,
        return_score=True,
        **kwargs,
    ) == ("A", 1, 0.4)


@pytest.mark.parametrize("m", [True, 1.5, "2"])
def test_mob_rejects_non_integer_resample_sizes(m: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        agg.majority_of_the_bests(["A", "B"], [0.8, 0.2], m=m)


def test_mob_accepts_numpy_integer_resample_size() -> None:
    assert agg.majority_of_the_bests(["A", "B"], [0.8, 0.2], m=np.int64(1)) == "A"


def test_keep_rejects_bool_instead_of_treating_it_as_one() -> None:
    with pytest.raises(ValueError, match="bool"):
        _keep_count(True, 10)


def _valid_calibration_kwargs() -> dict[str, Any]:
    return {
        "correct_logits": np.array([0.5, 1.0]),
        "incorrect_logits": np.array([-1.0, -0.5]),
        "correct_bandwidth": 0.4,
        "incorrect_bandwidth": 0.6,
        "bin_edges": np.array([-np.inf, np.inf]),
        "bin_probability": np.array([0.5]),
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"correct_logits": [[0.0, 1.0]]}, "one-dimensional"),
        ({"correct_logits": []}, "correct and incorrect samples"),
        ({"incorrect_logits": [-1.0, np.inf]}, "logit samples"),
        ({"correct_bandwidth": 0.0}, "bandwidths"),
        (
            {"bin_edges": [-np.inf, 0.0, np.inf]},
            "exactly one more value",
        ),
        ({"bin_edges": [0.0, np.inf]}, "start at -inf"),
        ({"bin_edges": [-np.inf, 0.0]}, r"end at \+inf"),
        (
            {
                "bin_edges": [-np.inf, 0.0, 0.0, np.inf],
                "bin_probability": [0.2, 0.3, 0.4],
            },
            "strictly increasing",
        ),
        ({"bin_probability": [-0.1]}, r"in \[0, 1\]"),
        ({"kernel": "epanechnikov"}, "gaussian"),
        ({"binning": "uniform"}, "quantile"),
    ],
    ids=[
        "vector-shape",
        "empty-class",
        "nonfinite-logit",
        "bandwidth",
        "edge-count",
        "edge-start",
        "edge-end",
        "edge-order",
        "bin-probability",
        "kernel",
        "binning",
    ],
)
def test_direct_calibration_state_validates_all_invariants(
    overrides: dict[str, Any], message: str
) -> None:
    kwargs = _valid_calibration_kwargs()
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        agg.KDEVoteCalibration(**kwargs)


@pytest.mark.parametrize("n_answers", [True, 1, 2.5])
def test_calibration_weights_validate_answer_count(n_answers: Any) -> None:
    with pytest.raises(ValueError, match="n_answers"):
        _constant_kde_calibration(0.5).weights([0.4, 0.6], n_answers=n_answers)


@pytest.mark.parametrize(
    ("probability", "sign_check"),
    [(0.0, np.isneginf), (1.0, np.isposinf)],
)
def test_calibration_weights_preserve_exact_reliability_limits(
    probability: float, sign_check: Any
) -> None:
    weights = _constant_kde_calibration(probability).weights([0.4, 0.6], n_answers=2)
    assert np.all(sign_check(weights))


def test_calibration_accepts_distinct_class_bandwidths() -> None:
    calibration = agg.fit_kde_vote_calibration(
        [0.8, 0.9, 0.1, 0.2],
        [1, 1, 0, 0],
        bandwidth=(0.2, 0.8),
    )
    assert calibration.correct_bandwidth == pytest.approx(0.2)
    assert calibration.incorrect_bandwidth == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("bandwidth", "message"),
    [
        ("silverman", "positive number"),
        ([0.5], "sequence"),
        (object(), "positive number"),
        (0.0, "finite and > 0"),
        (np.inf, "finite and > 0"),
    ],
)
def test_calibration_rejects_invalid_bandwidth_specifications(
    bandwidth: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        agg.fit_kde_vote_calibration(
            [0.8, 0.9, 0.1, 0.2],
            [1, 1, 0, 0],
            bandwidth=bandwidth,
        )


def test_scott_bandwidth_needs_two_samples_in_each_class() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        agg.fit_kde_vote_calibration([0.8, 0.2, 0.3], [1, 0, 0])


def test_calibration_rejects_empty_and_nonnumeric_targets() -> None:
    with pytest.raises(ValueError, match="at least one calibration response"):
        agg.fit_kde_vote_calibration([], [], bandwidth=0.5)
    with pytest.raises(ValueError, match="boolean or 0/1"):
        agg.fit_kde_vote_calibration(
            [0.8, 0.2], ["correct", "incorrect"], bandwidth=0.5
        )


def test_repeated_calibration_scores_collapse_empty_quantile_bins() -> None:
    calibration = agg.fit_kde_vote_calibration(
        [0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
        [0, 0, 0, 1, 1, 1],
        n_bins=20,
        bandwidth=0.5,
    )
    assert calibration.n_bins == 1
    np.testing.assert_allclose(calibration.bin_probability, [0.5])


def test_cges_boolean_options_are_not_truthiness_flags() -> None:
    with pytest.raises(ValueError, match="allow_other must be a bool"):
        agg.cges_vote(["A"], [0.8], allow_other=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="include_other must be a bool"):
        agg.cges_stop(["A"], [0.8], include_other="yes")  # type: ignore[arg-type]
    assert agg.cges_vote(["A"], [0.8], allow_other=np.bool_(False)) == "A"


REAL_GENERATIONS = (
    Path(__file__).resolve().parent / "fixtures" / "aime25_q06_compass.json"
)


def _load_real_generations() -> dict[str, Any]:
    return json.loads(REAL_GENERATIONS.read_text(encoding="utf-8"))


def test_real_verifier_scores_change_selection_and_fit_kde_voting() -> None:
    fixture = _load_real_generations()
    answers = [str(answer) for answer in fixture["answers"]]
    scores = [float(score) for score in fixture["scores"]]
    correct = [int(label) for label in fixture["correct"]]
    ground_truth = str(fixture["ground_truth"])
    assert fixture["source"] == "aime25/gpt-oss-20b_low/q06.jsonl.gz"
    assert len(answers) == len(scores) == len(correct) == 80
    assert ground_truth == "821"
    assert all(0.0 < score < 1.0 and math.isfinite(score) for score in scores)

    counts = Counter(answers)
    expected_majority = max(counts, key=counts.__getitem__)
    assert agg.majority_vote(answers) == expected_majority == "271"

    totals: defaultdict[str, float] = defaultdict(float)
    for answer, score in zip(answers, scores, strict=True):
        totals[answer] += score
    expected_weighted = max(totals, key=totals.__getitem__)
    assert agg.weighted_majority_vote(answers, scores) == expected_weighted
    assert expected_weighted == ground_truth

    calibration = agg.fit_kde_vote_calibration(
        scores[:40],
        correct[:40],
        n_bins=5,
        bandwidth=0.5,
    )

    assert agg.kde_weighted_vote(answers[40:], scores[40:], calibration) == ground_truth
