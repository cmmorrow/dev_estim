from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import erf, log, sqrt
from typing import Optional, Protocol, Tuple

import numpy as np

from dev_estim.utils import brent_root, normal_quantile

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


class EstimationModel(Protocol):
    def add_completed(self, m: float, actual_working_days: float) -> None:
        """Adds a completed task to the model.
        m is the estimated number of days a task should take to complete.
        actual_working_days is the number of days it took to complete the task."""
        ...


@dataclass
class DeveloperEfficiencyModel:
    """Tracks time-varying bias and volatility of r = ln(T/m) with exponential time dacay.
    Half-life is specified by the number of recently completed tasks in the estimation."""

    tasks_half_life: int = 15
    n_completed: int = 0
    sum_r: float = 0.0
    sum_r2: float = 0.0

    def add_completed(self, m: float, actual_working_days: float) -> None:
        decay = 0.5 ** (1 / self.tasks_half_life)
        self.n_completed += 1
        r = log(actual_working_days / m)
        self.sum_r = decay * self.sum_r + r
        self.sum_r2 = decay * self.sum_r2 + r * r

    @property
    def bias(self) -> float:
        """Returns the bias of the model."""
        n = self.n_completed
        rbar = self.sum_r / n if n > 0 else 0.0
        return rbar

    @property
    def sigma(self) -> float:
        """Returns the standard deviation of the model."""
        n = self.n_completed
        if n <= 0:
            return 0.0
        rbar = self.sum_r / n
        rbar2 = self.sum_r2 / n
        return sqrt(rbar2 - rbar * rbar)


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
    """
    Posterior parameters for the Normal–Inverse-Gamma on (bias, sigma^2):
      mu, kappa, alpha, beta where
        bias | sigma^2 ~ Normal(mu, sigma^2 / kappa)
        sigma^2        ~ InvGamma(alpha, beta)
    Returns prior hyperparameters when no completed tasks are present.
    """
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
    model: EstimationModel, multiplier: float, n_samples: int = 50000
) -> float:
    """
    Posterior-predictive P(T <= multiplier * m) for a fresh task (t=0),
    which depends only on sigma via: Phi(ln(multiplier)/sigma).
    """
    if isinstance(model, DeveloperDurationModel):
        _, sig = sample_bias_and_sigma(model, n=n_samples)
    elif isinstance(model, DeveloperEfficiencyModel):
        sig = np.array([model.sigma])
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
    return float(np.mean([normal_cdf(log(multiplier) / s) for s in sig]))


def calibrate_prior(
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
    return alpha0, beta0


def probability_finish_by_due(
    model: EstimationModel,
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

    if isinstance(model, DeveloperDurationModel):
        bias, sigma = sample_bias_and_sigma(model, n=n_samples)
    elif isinstance(model, DeveloperEfficiencyModel):
        bias = np.array([model.bias])
        sigma = np.array([model.sigma])
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    probs = []
    for b, s in zip(bias, sigma):
        Ft = lognormal_cdf(t, median=m, sigma=s, bias=b)
        FD = lognormal_cdf(D, median=m, sigma=s, bias=b)
        denom = 1.0 - Ft
        probs.append(0.0 if denom <= 1e-12 else max(0.0, (FD - Ft) / denom))
    return float(np.mean(probs))


def task_duration_estimated_days(
    model: EstimationModel,
    H: float,
    p_complete: float = 0.95,
    n_samples: int = 50000,
    conservative_quantile: float = 0.1,
) -> float:
    """
    Compute a conservative "safe estimate" m (in working days) such that the task
    completes within horizon H with probability p, while ALSO accounting
    for uncertainty in the developer hyperparameters (bias + execution variability).

    Model:
        ln T ~ Normal(ln m + b, sigma_tot^2)
        sigma_tot = sqrt(sigma_est^2 + sigma_exec^2)

    If b and sigma_exec are uncertain (posterior), we:
        1) draw samples (b_s, sigma_exec_s) from their posterior
        2) compute per-draw safe estimate:
            m_s = H / exp(b_s + z_p * sigma_tot_s)
        3) return a conservative summary, e.g. the 10th percentile of {m_s}.

    Parameters
    ----------
    H : float
        Available time horizon in working days.
    p_complete : float
        Desired completion probability within H for a fixed set of hyperparameters.
        Example: 0.95.
    n_samples : int
        Number of posterior draws to use.
    conservative_quantile : float
        Quantile of m_s to return. Smaller => more conservative.
        Example: 0.10 returns the 10th percentile.

    Returns
    -------
    float
        Conservative safe estimate m (working days).

    Notes
    -----
    - This provides robustness to hyperparameter uncertainty. If you set
        conservative_quantile=0.50, you get the median m under hyperparameter uncertainty.
    - Choose conservative_quantile based on risk tolerance (0.05–0.20 are common).
    """
    if H <= 0:
        raise ValueError("H must be > 0")
    if not (0.0 < p_complete < 1.0):
        raise ValueError("p_complete must be in (0,1)")
    z = normal_quantile(p_complete)
    if isinstance(model, DeveloperDurationModel):
        bias, sigma = sample_bias_and_sigma(model, n=n_samples)
    elif isinstance(model, DeveloperEfficiencyModel):
        bias = np.array([model.bias])
        sigma = np.array([model.sigma])
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
    estimates = H / np.exp(bias + z * sigma)
    return float(np.quantile(estimates, conservative_quantile))
