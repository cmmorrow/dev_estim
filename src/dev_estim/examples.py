from datetime import date

from dev_estim.task_prob import (
    DeveloperDurationModel,
    calibrate_prior,
    p_within_multiplier,
    posterior_params,
    probability_finish_by_due,
)

# -----------------------------
# Sanity check with your records
# -----------------------------
dev = DeveloperDurationModel(alpha0=2.0, beta0=0.2)

records = [
    (3, 7.0),
    (2, 2.0),
    (1, 0.5),  # "< 1 day" assumed as 0.5; change to 0.75 if you want
    (5, 5.0),
    (2, 1.0),
]

for pts, actual in records:
    dev.add_completed(pts, actual)

mu_n, kappa_n, alpha_n, beta_n = posterior_params(dev)
print("Posterior InvGamma params for sigma^2:")
print(f"  alpha={alpha_n:.3f}, beta={beta_n:.3f}, mu={mu_n:.3f}, kappa={kappa_n:.3f}")
print(f"  completed tasks: n={dev.n_completed}, sum_sq={dev.sum_r2:.3f}")

# Sanity check: implied P(within 1.5x estimate) under posterior predictive
p_15x = p_within_multiplier(dev, 1.5, n_samples=80000)
print(f"\nPosterior-predictive P(finish within 1.5× estimate): {p_15x:.3f}")

# Example "plug-in" probability:
# 5-point task, started 2025-12-01, today 2025-12-15, due 2025-12-19 (due exclusive)
p_due = probability_finish_by_due(
    model=dev,
    start=date(2025, 12, 1),
    today=date(2025, 12, 15),
    due=date(2025, 12, 19),
    m=5,
    n_samples=80000,
)
print(f"\nExample P(finish by due | still in progress today): {p_due:.3f}")


# ---- example: build your calibrated prior ----
dev = DeveloperDurationModel()
alpha0, beta0 = calibrate_prior(
    multiplier=1.5,
    target_prob=0.8,
    prior_equiv_tasks=4,  # weak-to-moderate prior
)

print("Calibrated prior:")
print(f"  alpha0 = {alpha0:.3f}")
print(f"  beta0  = {beta0:.3f}")

# Sanity check
dev = DeveloperDurationModel(alpha0=alpha0, beta0=beta0)
p_check = p_within_multiplier(model=dev, multiplier=0.5, n_samples=200000)
print(f"Prior-predictive P(within 1.5×) ≈ {p_check:.3f}")
