import math

import pytest

from dev_estim.utils import brent_root


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
