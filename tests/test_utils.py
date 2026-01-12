import math

import pytest

from dev_estim.utils import brent_root, normal_quantile


def test_brent_root_finds_positive_root_of_quadratic() -> None:
    root = brent_root(lambda x: x * x - 2.0, a=0.0, b=2.0)
    assert math.isclose(root, math.sqrt(2.0), rel_tol=0, abs_tol=1e-10)


def test_brent_root_returns_endpoint_if_zero_at_a() -> None:
    root = brent_root(lambda x: x, a=0.0, b=1.0)
    assert root == 0.0


def test_brent_root_raises_when_interval_not_bracketing() -> None:
    with pytest.raises(ValueError):
        brent_root(lambda x: x * x + 1.0, a=-1.0, b=1.0)


def test_brent_root_raises_when_maxiter_zero() -> None:
    with pytest.raises(RuntimeError):
        brent_root(lambda x: x - 1.0, a=0.0, b=2.0, maxiter=0)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def test_normal_quantile_returns_zero_at_half() -> None:
    assert normal_quantile(0.5) == 0.0


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.975, 1.959963984540054),  # ~1.96
        (0.025, -1.959963984540054),
        (0.15865525393145707, -1.0),  # CDF(-1) ~ 0.1587
    ],
)
def test_normal_quantile_matches_known_values(p: float, expected: float) -> None:
    assert math.isclose(normal_quantile(p), expected, rel_tol=0, abs_tol=1e-6)


def test_normal_quantile_roundtrip_with_cdf() -> None:
    p = 0.8
    z = normal_quantile(p)
    assert math.isclose(_normal_cdf(z), p, rel_tol=0, abs_tol=1e-9)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.1])
def test_normal_quantile_rejects_invalid_p(p: float) -> None:
    with pytest.raises(ValueError):
        normal_quantile(p)
