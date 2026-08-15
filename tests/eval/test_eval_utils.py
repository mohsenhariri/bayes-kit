from __future__ import annotations

import pytest

from scorio.eval.utils import normal_credible_interval


@pytest.mark.parametrize(
    ("mu", "expected"),
    [
        (2.0, (1.0, 1.0)),
        (-1.0, (0.0, 0.0)),
    ],
)
def test_normal_interval_clipping_cannot_invert_bounds(
    mu: float, expected: tuple[float, float]
) -> None:
    interval = normal_credible_interval(mu, 0.01, bounds=(0.0, 1.0))

    assert interval == pytest.approx(expected)
    assert 0.0 <= interval[0] <= interval[1] <= 1.0
