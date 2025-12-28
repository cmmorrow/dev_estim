from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from math import erf, log, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------
# Working-day math (due exclusive)
# -----------------------------


def _to_np_date(d: date) -> np.datetime64:
    return np.datetime64(d.isoformat(), "D")


def working_days_between(start: date, end: date) -> int:
    """
    Business days in [start, end), i.e. start inclusive, end exclusive.
    This matches "due date exclusive".
    """
    return int(np.busday_count(_to_np_date(start), _to_np_date(end)))


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def lognormal_cdf(x: float, median: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    z = (log(x) - log(median)) / sigma
    return normal_cdf(z)


# -----------------------------
# Story points -> working days
# -----------------------------

def _load_points_to_days() -> Dict[int, float]:
    json_path = Path(__file__).with_name("points_to_days.json")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): float(v) for k, v in data.items()}


POINTS_TO_DAYS: Dict[int, float] = _load_points_to_days()


def estimate_days(points: int) -> float:
    if points not in POINTS_TO_DAYS:
        raise ValueError(f"Missing mapping for points={points}")
    return float(POINTS_TO_DAYS[points])


# -----------------------------
# Bayesian developer model
# -----------------------------


@dataclass
class DeveloperDurationModel:
    """
    Duration model:
      T = m * exp(eps),  eps ~ Normal(0, sigma^2)

    Prior:
      sigma^2 ~ InvGamma(alpha0, beta0)

    Completed-task conjugate update:
      eps_i = ln(T_i / m_i)
      sum_sq = Σ eps_i^2
      alpha_n = alpha0 + n/2
      beta_n  = beta0  + sum_sq/2
    """

    alpha0: float = 2.0
    beta0: float = 0.2
    n_completed: int = 0
    sum_sq: float = 0.0
    _eps: List[float] = field(default_factory=list)

    def posterior_params(self) -> Tuple[float, float]:
        return (self.alpha0 + 0.5 * self.n_completed, self.beta0 + 0.5 * self.sum_sq)

    def add_completed(self, points: int, actual_working_days: float) -> None:
        m = estimate_days(points)
        eps = log(actual_working_days / m)
        self.n_completed += 1
        self.sum_sq += eps * eps
        self._eps.append(eps)

    def sample_sigma(
        self, n: int = 50000, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        alpha, beta = self.posterior_params()
        # If sigma^2 ~ InvGamma(alpha,beta), then precision tau=1/sigma^2 ~ Gamma(alpha, rate=beta)
        tau = rng.gamma(shape=alpha, scale=1.0 / beta, size=n)
        return np.sqrt(1.0 / tau)

    def p_within_multiplier(self, multiplier: float, n_samples: int = 50000) -> float:
        """
        Posterior-predictive P(T <= multiplier * m) for a fresh task (t=0),
        which depends only on sigma via: Phi(ln(multiplier)/sigma).
        """
        sig = self.sample_sigma(n_samples)
        return float(np.mean([normal_cdf(log(multiplier) / s) for s in sig]))

    def probability_finish_by_due(
        self,
        start: date,
        due: date,
        today: date,
        points: int,
        n_samples: int = 50000,
    ) -> float:
        """
        P(T <= D | T > t, developer data), with D and t in working days.
        Due is exclusive because working_days_between uses [start, due).
        """
        if today < start:
            raise ValueError("today must be >= start")
        if due <= start:
            raise ValueError("due must be > start")

        m = estimate_days(points)
        t = float(working_days_between(start, today))
        D = float(working_days_between(start, due))

        if D <= t:
            return 0.0

        sig = self.sample_sigma(n_samples)
        probs = []
        for s in sig:
            Ft = lognormal_cdf(t, median=m, sigma=s)
            FD = lognormal_cdf(D, median=m, sigma=s)
            denom = 1.0 - Ft
            probs.append(0.0 if denom <= 1e-12 else max(0.0, (FD - Ft) / denom))
        return float(np.mean(probs))
