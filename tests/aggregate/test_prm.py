"""Tests for scorio.aggregate.prm (process/outcome reward aggregation)."""

from __future__ import annotations

import numpy as np
import pytest

from scorio.aggregate import prm_aggregate


class TestPrmAggregate:
    STEPS = [0.9, 0.4, 0.95]

    def test_last(self) -> None:
        assert prm_aggregate(self.STEPS, method="last") == pytest.approx(0.95)

    def test_min(self) -> None:
        assert prm_aggregate(self.STEPS, method="min") == pytest.approx(0.4)

    def test_max(self) -> None:
        assert prm_aggregate(self.STEPS, method="max") == pytest.approx(0.95)

    def test_mean(self) -> None:
        assert prm_aggregate(self.STEPS, method="mean") == pytest.approx(
            (0.9 + 0.4 + 0.95) / 3
        )

    def test_prod(self) -> None:
        assert prm_aggregate(self.STEPS, method="prod") == pytest.approx(
            0.9 * 0.4 * 0.95
        )

    def test_default_is_last(self) -> None:
        assert prm_aggregate(self.STEPS) == prm_aggregate(self.STEPS, method="last")

    def test_single_step(self) -> None:
        assert prm_aggregate([0.7], method="min") == pytest.approx(0.7)

    def test_bad_method_raises(self) -> None:
        with pytest.raises(ValueError):
            prm_aggregate(self.STEPS, method="median")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            prm_aggregate([], method="mean")

    def test_nonfinite_raises(self) -> None:
        with pytest.raises(ValueError):
            prm_aggregate([0.5, float("inf")], method="mean")

    def test_builds_score_matrix_for_selection(self) -> None:
        # (M=2 questions, N=2 traces), each trace a list of step scores
        traces = [[[0.9, 0.8], [0.7, 0.6]], [[0.5, 0.9], [0.95, 0.99]]]
        scores = np.array([[prm_aggregate(t, method="min") for t in q] for q in traces])
        assert scores.tolist() == [[0.8, 0.6], [0.5, 0.95]]
