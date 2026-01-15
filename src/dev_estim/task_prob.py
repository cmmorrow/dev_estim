from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import erf, exp, log, sqrt
from typing import Optional, Tuple

import numpy as np

from dev_estim.utils import brent_root, normal_quantile

# Sentinel value intended for determining if a user provided parameter is missing
MISSING: float = float("nan")

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


def lognormal_cdf(x: float, median: float, sigma: float, bias: float = 0.0) -> float:
    if x <= 0:
        return 0.0
    z = (log(x) - (log(median) + bias)) / sigma
    return normal_cdf(z)


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

    # prior hyperparameters
    alpha0: float = 2.0
    beta0: float = 0.2
    mu0: float = 0.0
    kappa0: float = 4.0

    n_completed: int = 0
    sum_r: float = 0.0
    sum_r2: float = 0.0
    # _eps: List[float] = field(default_factory=list)

    def add_completed(self, m: float, actual_working_days: float) -> None:
        """Adds a completed task to the developer model.
        m is the estimated number of days a task should take to complete.
        actual_working_days is the number of days it took the developer to complete the task."""
        eps = log(actual_working_days / m)
        self.n_completed += 1
        self.sum_r += eps
        self.sum_r2 += eps * eps
        # self._eps.append(eps)


def posterior_params(
    model: DeveloperDurationModel,
) -> Tuple[float, float, float, float]:
    n = model.n_completed
    if n == 0:
        return (model.mu0, model.kappa0, model.alpha0, model.beta0)
    rbar = model.sum_r / n
    sse = model.sum_r2 - n * (rbar * rbar)

    kappa = model.kappa0 + n
    mu = (model.kappa0 * model.mu0 + n * rbar) / kappa
    alpha = model.alpha0 + 0.5 * n
    beta = (
        model.beta0
        + 0.5 * sse
        + (model.kappa0 * n * (rbar - model.mu0) ** 2) / (2.0 * kappa)
    )
    return mu, kappa, alpha, beta


def sample_bias_and_sigma(
    model: DeveloperDurationModel,
    n: int = 50000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw samples from the posterior:
        sigma_total^2 ~ InvGamma(alpha, beta)
        b | sigma_total^2 ~ Normal(mu, sigma_total^2/kappa)
    """
    rng = rng or np.random.default_rng()
    mu, kappa, alpha, beta = posterior_params(model)
    # If sigma^2 ~ InvGamma(alpha,beta), then precision tau=1/sigma^2 ~ Gamma(alpha, rate=beta)
    tau = rng.gamma(shape=alpha, scale=1.0 / beta, size=n)
    sigma2 = 1.0 / tau
    sigma = np.sqrt(sigma2)
    bias = rng.normal(loc=mu, scale=np.sqrt(sigma2 / kappa), size=n)
    return bias, sigma


def p_within_multiplier(
    model: DeveloperDurationModel, multiplier: float, n_samples: int = 50000
) -> float:
    """
    Posterior-predictive P(T <= multiplier * m) for a fresh task (t=0),
    which depends only on sigma via: Phi(ln(multiplier)/sigma).
    """
    _, sig = sample_bias_and_sigma(model, n_samples)
    return float(np.mean([normal_cdf(log(multiplier) / s) for s in sig]))


def fit_inv_gamma_prior_for_multiplier(
    multiplier: float = 1.5,
    target_prob: float = 0.8,
    prior_equiv_tasks: int = 1,
    n_samples: int = 80000,
) -> Tuple[float, float]:
    """
    Returns (alpha0, beta0) such that
    E[ Phi( ln(multiplier) / sigma ) ] ≈ target_prob
    """
    alpha0 = 1.0 + prior_equiv_tasks / 2.0

    def objective(beta0) -> float:
        rng = np.random.default_rng()
        tau = rng.gamma(shape=alpha0, scale=1.0 / beta0, size=n_samples)
        sigma = np.sqrt(1.0 / tau)
        p = float(np.mean([normal_cdf(log(multiplier) / s) for s in sigma]))
        return p - target_prob

    # Reasonable search interval for beta0
    beta0 = brent_root(objective, 0.01, 10.0)
    # self.alpha0 = alpha0
    # self.beta0 = beta0
    return alpha0, beta0


def probability_finish_by_due(
    model: DeveloperDurationModel,
    start: date,
    due: date,
    today: date,
    m: float,
    n_samples: int = 50000,
) -> float:
    """
    P(T <= D | T > t, developer data), with D and t in working days.
    Due is exclusive because working_days_between uses [start, due).

    T modeled as LogNormal(mu=ln(m)+b, sigma_total).
    We sample (b, sigma_total) from posterior, then compute conditional probability.
    """
    if today < start:
        raise ValueError("today must be >= start")
    if due <= start:
        raise ValueError("due must be > start")

    t = float(working_days_between(start, today))
    D = float(working_days_between(start, due))

    if D <= t:
        return 0.0

    bias, sigma = sample_bias_and_sigma(model, n_samples)
    probs = []
    for b, s in zip(bias, sigma):
        Ft = lognormal_cdf(t, median=m, sigma=s, bias=b)
        FD = lognormal_cdf(D, median=m, sigma=s, bias=b)
        denom = 1.0 - Ft
        probs.append(0.0 if denom <= 1e-12 else max(0.0, (FD - Ft) / denom))
    return float(np.mean(probs))


def realistic_estimated_days(
    H: float, sigma: float, bias: float = 0.0, p: float = 0.95
) -> float:
    z = normal_quantile(p)
    return H / exp(bias + z * sigma)
