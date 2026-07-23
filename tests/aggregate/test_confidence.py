"""Tests for scorio.aggregate confidence signals (confidence.py)."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.special import logsumexp

from scorio.aggregate import confidence as C


class TestSequenceLikelihood:
    def test_mean_and_sequence_logprob(self) -> None:
        lp = [-0.1, -0.2, -0.3]
        assert C.mean_logprob(lp) == pytest.approx(-0.2)
        assert C.sequence_logprob(lp) == pytest.approx(-0.6)

    def test_perplexity_is_exp_neg_mean(self) -> None:
        lp = [-0.1, -0.7, -0.4]
        assert C.perplexity(lp) == pytest.approx(math.exp(-C.mean_logprob(lp)))

    def test_perplexity_of_certain_trace_is_one(self) -> None:
        assert C.perplexity([0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_higher_logprob_is_more_confident(self) -> None:
        assert C.mean_logprob([-0.05] * 4) > C.mean_logprob([-0.9] * 4)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            C.mean_logprob([])

    def test_nonfinite_raises(self) -> None:
        with pytest.raises(ValueError):
            C.mean_logprob([0.0, float("-inf")])


class TestPicsar:
    def test_no_split_equals_sequence_logprob(self) -> None:
        lp = [-0.1, -0.2, -0.3, -0.4]
        assert C.picsar(lp) == pytest.approx(C.sequence_logprob(lp))

    def test_split_sums_reasoning_and_answer(self) -> None:
        lp = [-0.1, -0.2, -0.3, -0.4]
        assert C.picsar(lp, answer_start=3) == pytest.approx(-0.6 + -0.4)

    def test_normalize_reasoning(self) -> None:
        lp = [-0.2, -0.4, -0.5]  # reasoning mean = -0.3, answer = -0.5
        assert C.picsar(lp, answer_start=2, normalize_reasoning=True) == pytest.approx(
            -0.3 + -0.5
        )

    def test_bad_answer_start_raises(self) -> None:
        with pytest.raises(ValueError):
            C.picsar([-0.1, -0.2], answer_start=5)


UNIFORM3 = [[-math.log(3.0)] * 3]
PEAKED = [[0.0, -20.0, -20.0]]


class TestSelfCertainty:
    def test_uniform_is_zero(self) -> None:
        assert C.self_certainty(UNIFORM3) == pytest.approx(0.0, abs=1e-9)

    def test_peaked_is_large(self) -> None:
        assert C.self_certainty(PEAKED) > 5.0

    def test_invariant_to_common_logprob_shift(self) -> None:
        # renormalization means an additive shift of a row's logprobs is a no-op
        base = [[-0.2, -1.0, -3.0]]
        shifted = [[v - 2.5 for v in base[0]]]
        assert C.self_certainty(base) == pytest.approx(C.self_certainty(shifted))

    def test_aggregate_min_le_mean(self) -> None:
        tk = [[0.0, -20.0], [-0.7, -0.7]]
        assert C.self_certainty(tk, aggregate="min") <= C.self_certainty(tk)

    def test_bad_aggregate_raises(self) -> None:
        with pytest.raises(ValueError):
            C.self_certainty(PEAKED, aggregate="median")


class TestEntropy:
    def test_uniform_entropy_is_log_k(self) -> None:
        assert C.token_entropy(UNIFORM3) == pytest.approx(math.log(3.0))

    def test_one_hot_entropy_near_zero(self) -> None:
        assert C.token_entropy(PEAKED) < 1e-6

    def test_max_aggregate_picks_worst_token(self) -> None:
        tk = [[0.0, -20.0], [-math.log(2.0), -math.log(2.0)]]
        assert C.token_entropy(tk, aggregate="max") == pytest.approx(math.log(2.0))

    def test_varentropy_uniform_is_zero(self) -> None:
        assert C.varentropy(UNIFORM3) == pytest.approx(0.0, abs=1e-9)

    def test_varentropy_nonnegative(self) -> None:
        assert C.varentropy([[0.0, -1.0, -3.0], [-0.2, -0.4, -5.0]]) >= 0.0


class TestMarginAndMaxProb:
    def test_max_softmax_probability_is_exact_top1(self) -> None:
        # rows: p1 = 1/(1+e^-20) ~ 1, and 0.5
        tk = [[0.0, -20.0], [-math.log(2.0), -math.log(2.0)]]
        assert C.max_softmax_probability(tk) == pytest.approx((1.0 + 0.5) / 2, abs=1e-6)

    def test_max_prob_min_aggregate(self) -> None:
        tk = [[0.0, -20.0], [-math.log(2.0), -math.log(2.0)]]
        assert C.max_softmax_probability(tk, aggregate="min") == pytest.approx(0.5)

    def test_logprob_margin_value(self) -> None:
        tk = [[0.0, -1.0, -2.0], [-0.5, -0.7, -3.0]]  # margins 1.0 and 0.2
        assert C.logprob_margin(tk) == pytest.approx(0.6)

    def test_prob_margin(self) -> None:
        tk = [[0.0, -math.inf if False else -100.0]]  # top1~1, top2~0
        assert C.logprob_margin(tk, use_prob=True) == pytest.approx(1.0, abs=1e-6)

    def test_single_candidate_row_margin_zero(self) -> None:
        assert C.logprob_margin([[0.0]]) == pytest.approx(0.0)


class TestDeepConf:
    def test_token_confidence_values(self) -> None:
        assert C.token_confidence([[0.0, -2.0], [-1.0, -3.0]]).tolist() == [1.0, 2.0]

    def test_mean_mode(self) -> None:
        tk = [[0.0, -2.0]] * 3 + [[-1.0, -3.0]] * 3  # conf 1,1,1,2,2,2
        assert C.deepconf_confidence(tk, mode="mean") == pytest.approx(1.5)

    def test_tail_mode(self) -> None:
        tk = [[0.0, -2.0]] * 3 + [[-1.0, -3.0]] * 3
        assert C.deepconf_confidence(tk, mode="tail", tail_tokens=3) == pytest.approx(
            2.0
        )

    def test_lowest_group(self) -> None:
        tk = [[0.0, -2.0]] * 3 + [[-1.0, -3.0]] * 3
        assert C.deepconf_confidence(
            tk, mode="lowest_group", window=3
        ) == pytest.approx(1.0)

    def test_group_reductions_ordered(self) -> None:
        rng = np.random.default_rng(0)
        tk = rng.uniform(-5.0, 0.0, size=(200, 8))
        low = C.deepconf_confidence(tk, mode="lowest_group", window=32)
        bot = C.deepconf_confidence(tk, mode="bottom_group", window=32)
        mean = C.deepconf_confidence(tk, mode="mean")
        assert low <= bot <= mean

    def test_window_longer_than_trace_is_single_group(self) -> None:
        tk = [[0.0, -2.0], [-1.0, -3.0]]  # conf 1, 2 ; mean 1.5
        assert C.deepconf_confidence(
            tk, mode="lowest_group", window=999
        ) == pytest.approx(1.5)

    def test_bad_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            C.deepconf_confidence([[0.0, -1.0]], mode="p50")

    def test_bad_bottom_quantile_raises(self) -> None:
        with pytest.raises(ValueError):
            C.deepconf_confidence(
                [[0.0, -1.0]], mode="bottom_group", bottom_quantile=0.0
            )


class TestInputHandling:
    def test_rectangular_matches_ragged(self) -> None:
        mat = [[0.0, -1.0, -3.0], [-0.2, -0.9, -4.0], [-0.1, -2.0, -2.5]]
        for fn in (
            C.self_certainty,
            C.token_entropy,
            C.varentropy,
            C.max_softmax_probability,
            C.logprob_margin,
        ):
            assert fn(np.array(mat)) == pytest.approx(fn([list(r) for r in mat]))

    def test_ragged_topk_supported(self) -> None:
        # a genuinely ragged trace (different top-k per position)
        tk = [[0.0, -1.0], [-0.2, -0.9, -4.0]]
        val = C.token_entropy(tk)
        assert np.isfinite(val)

    def test_empty_topk_raises(self) -> None:
        with pytest.raises(ValueError):
            C.self_certainty([])

    def test_nonfinite_topk_raises(self) -> None:
        with pytest.raises(ValueError):
            C.token_entropy([[0.0, float("nan")]])

    def test_single_token_trace(self) -> None:
        assert C.deepconf_confidence([[0.0, -2.0]]) == pytest.approx(1.0)

    def test_extreme_finite_logits_remain_finite_without_warnings(self) -> None:
        topk = [[0.0, -1000.0]]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            entropy = C.token_entropy(topk)
            varentropy = C.varentropy(topk)
            certainty = C.self_certainty(topk)

        assert entropy == pytest.approx(0.0, abs=1e-300)
        assert varentropy == pytest.approx(0.0, abs=1e-300)
        assert certainty == pytest.approx(500.0 - math.log(2.0))

    def test_ragged_signals_match_independent_formulas(self) -> None:
        rows = [[0.0, -1.0], [-0.2, -0.9, -4.0], [-0.4]]
        reference: dict[Any, list[float]] = {
            C.self_certainty: [],
            C.token_entropy: [],
            C.varentropy: [],
            C.max_softmax_probability: [],
            C.logprob_margin: [],
        }
        expected_probability_margins: list[float] = []
        expected_confidences: list[float] = []
        for row in rows:
            values = np.asarray(row, dtype=float)
            log_p = values - logsumexp(values)
            probabilities = np.exp(log_p)
            entropy = float(np.sum(probabilities * -log_p))
            reference[C.self_certainty].append(
                -math.log(values.size) - float(np.mean(log_p))
            )
            reference[C.token_entropy].append(entropy)
            reference[C.varentropy].append(
                float(np.sum(probabilities * (-log_p - entropy) ** 2))
            )
            reference[C.max_softmax_probability].append(math.exp(float(np.max(values))))
            ordered = np.sort(values)
            if values.size == 1:
                reference[C.logprob_margin].append(0.0)
                expected_probability_margins.append(0.0)
            else:
                reference[C.logprob_margin].append(float(ordered[-1] - ordered[-2]))
                expected_probability_margins.append(
                    math.exp(float(ordered[-1])) - math.exp(float(ordered[-2]))
                )
            expected_confidences.append(-float(np.mean(values)))

        for rule, per_token_values in reference.items():
            assert rule(rows) == pytest.approx(np.mean(per_token_values))
        assert C.logprob_margin(rows, use_prob=True) == pytest.approx(
            np.mean(expected_probability_margins)
        )
        np.testing.assert_allclose(C.token_confidence(rows), expected_confidences)

    @pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
    def test_rectangular_input_needs_tokens_and_candidates(
        self, shape: tuple[int, int]
    ) -> None:
        with pytest.raises(ValueError, match="one token and one top-k candidate"):
            C.self_certainty(np.empty(shape))

    def test_ragged_input_validates_each_row(self) -> None:
        with pytest.raises(ValueError, match="every position"):
            C.token_entropy([[0.0], []])
        with pytest.raises(ValueError, match="finite"):
            C.token_entropy([[0.0], [np.nan, -1.0]])

    def test_empty_generator_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            C.token_entropy(row for row in [])

    def test_group_confidence_rejects_nonpositive_window(self) -> None:
        with pytest.raises(ValueError, match="window must be positive"):
            C.deepconf_confidence(
                [[0.0, -1.0], [-0.2, -0.8]],
                mode="lowest_group",
                window=0,
            )

    def test_bare_one_position_row_is_supported(self) -> None:
        np.testing.assert_allclose(C.token_confidence([0.0, -2.0]), [1.0])
