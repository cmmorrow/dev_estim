from datetime import date
from math import isclose, log

import numpy as np
import pytest

import dev_estim.task_prob as task_prob
from dev_estim.task_prob import (
    DeveloperDurationModel,
    calibrate_prior,
    lognormal_cdf,
    normal_cdf,
    p_within_multiplier,
    posterior_params,
    probability_finish_by_due,
    sample_bias_and_sigma,
    task_duration_estimated_days,
    working_days_between,
)
from dev_estim.utils import normal_quantile


def test_returns_zero_for_same_start_and_end_date() -> None:
    assert working_days_between(date(2024, 12, 2), date(2024, 12, 2)) == 0


def test_counts_only_weekdays_across_full_week() -> None:
    # Monday to the following Monday counts five working days; weekend excluded.
    assert working_days_between(date(2024, 12, 2), date(2024, 12, 9)) == 5


def test_end_date_is_exclusive() -> None:
    # Tuesday is not counted when due/end is exclusive.
    assert working_days_between(date(2024, 12, 2), date(2024, 12, 3)) == 1


def test_weekend_inside_range_is_ignored() -> None:
    # Start Friday, end Monday -> only Friday counts.
    assert working_days_between(date(2024, 12, 6), date(2024, 12, 9)) == 1


def test_weekend_start_date_skips_to_next_business_day() -> None:
    # Starting on a Saturday should start counting from Monday.
    assert working_days_between(date(2024, 12, 7), date(2024, 12, 10)) == 1


def test_add_completed_updates_counts_and_residuals() -> None:
    model = DeveloperDurationModel()

    model.add_completed(m=1, actual_working_days=2.0)

    expected_eps = log(2.0 / 1.0)  # estimate_days(2) = 1.0
    assert model.n_completed == 1
    assert isclose(model.sum_r, expected_eps, rel_tol=1e-9)
    assert isclose(model.sum_r2, expected_eps * expected_eps, rel_tol=1e-9)


def test_posterior_params_returns_priors_when_no_data() -> None:
    model = DeveloperDurationModel(alpha0=1.1, beta0=0.3, mu0=0.2, kappa0=5.0)
    mu, kappa, alpha, beta = posterior_params(model)
    assert mu == 0.2
    assert kappa == 5.0
    assert alpha == 1.1
    assert beta == 0.3


def test_posterior_params_accumulates_updates() -> None:
    model = DeveloperDurationModel(alpha0=2.0, beta0=0.2)
    model.add_completed(m=2, actual_working_days=6.0)  # estimate_days(3) = 2.0
    model.add_completed(m=0.5, actual_working_days=0.5)  # eps = 0

    mu, kappa, alpha, beta = posterior_params(model)
    eps = log(3.0)  # log(6/2)
    rbar = eps / 2
    expected_mu = (model.kappa0 * model.mu0 + 2 * rbar) / (model.kappa0 + 2)
    sse = eps * eps - 2 * (rbar * rbar)
    expected_beta = (
        model.beta0
        + 0.5 * sse
        + (model.kappa0 * 2 * (rbar - model.mu0) ** 2) / (2 * (model.kappa0 + 2))
    )

    assert isclose(alpha, 3.0, rel_tol=1e-9)
    assert isclose(kappa, model.kappa0 + 2, rel_tol=1e-9)
    assert isclose(mu, expected_mu, rel_tol=1e-9)
    assert isclose(beta, expected_beta, rel_tol=1e-9)


def test_posterior_params_with_custom_prior_values() -> None:
    model = DeveloperDurationModel(alpha0=0.1, beta0=0.05, mu0=0.2, kappa0=2.0)
    model.add_completed(m=1, actual_working_days=1.0)  # matches estimate -> eps = 0
    model.add_completed(
        m=5, actual_working_days=10.0
    )  # estimate_days(5)=5 -> eps=ln(2)

    mu, kappa, alpha, beta = posterior_params(model)

    rbar = log(2.0) / 2
    sse = (log(2.0) ** 2) - 2 * (rbar * rbar)
    expected_mu = (2.0 * 0.2 + 2 * rbar) / (2.0 + 2)
    expected_beta = 0.05 + 0.5 * sse + (2.0 * 2 * (rbar - 0.2) ** 2) / (2 * (2.0 + 2))

    assert isclose(alpha, 0.1 + 1.0, rel_tol=1e-9)  # alpha0 + n/2 with n=2
    assert isclose(kappa, 4.0, rel_tol=1e-9)
    assert isclose(mu, expected_mu, rel_tol=1e-9)
    assert isclose(beta, expected_beta, rel_tol=1e-9)


def test_posterior_params_matches_reference_dataset() -> None:
    model = DeveloperDurationModel(alpha0=2.0, beta0=0.2)
    records = [
        (2.0, 7.0),
        (1.0, 2.0),
        (0.5, 0.5),
        (5.0, 5.0),
        (2.0, 1.0),
    ]

    for m, actual in records:
        model.add_completed(m=m, actual_working_days=actual)

    mu, kappa, alpha, beta = posterior_params(model)
    # expected values computed with the same formulas as posterior_params
    eps = [log(actual / m) for m, actual in records]
    rbar = sum(eps) / len(eps)
    sse = sum(e * e for e in eps) - len(eps) * (rbar * rbar)
    expected_mu = (model.kappa0 * model.mu0 + len(eps) * rbar) / (
        model.kappa0 + len(eps)
    )
    expected_beta = (
        model.beta0
        + 0.5 * sse
        + (model.kappa0 * len(eps) * (rbar - model.mu0) ** 2)
        / (2 * (model.kappa0 + len(eps)))
    )

    assert isclose(alpha, 4.5, rel_tol=1e-12)
    assert isclose(kappa, model.kappa0 + len(eps), rel_tol=1e-12)
    assert isclose(mu, expected_mu, rel_tol=1e-12)
    assert isclose(beta, expected_beta, rel_tol=1e-12)


def test_posterior_params_integration_with_custom_prior() -> None:
    model = DeveloperDurationModel(alpha0=1.5, beta0=0.1, mu0=0.1, kappa0=3.0)
    records = [
        (2.0, 1.0),
        (10.0, 15.0),  # 1.5x estimate -> small positive eps
        (2.0, 1.0),  # faster than estimate -> negative eps
    ]

    for m, actual in records:
        model.add_completed(m=m, actual_working_days=actual)

    mu, kappa, alpha, beta = posterior_params(model)
    eps = [log(actual / m) for m, actual in records]
    rbar = sum(eps) / len(eps)
    sse = sum(e * e for e in eps) - len(eps) * (rbar * rbar)
    expected_mu = (3.0 * 0.1 + len(eps) * rbar) / (3.0 + len(eps))
    expected_beta = (
        0.1 + 0.5 * sse + (3.0 * len(eps) * (rbar - 0.1) ** 2) / (2 * (3.0 + len(eps)))
    )

    assert isclose(alpha, 1.5 + len(records) / 2, rel_tol=1e-12)
    assert isclose(kappa, 3.0 + len(records), rel_tol=1e-12)
    assert isclose(mu, expected_mu, rel_tol=1e-12)
    assert isclose(beta, expected_beta, rel_tol=1e-12)


def test_sample_bias_and_sigma_uses_given_rng_and_shapes() -> None:
    rng = np.random.default_rng(123)
    model = DeveloperDurationModel()
    model.add_completed(m=1, actual_working_days=2.0)

    bias, sigmas = sample_bias_and_sigma(model, n=3, rng=rng)

    assert bias.shape == (3,)
    assert sigmas.shape == (3,)
    assert np.all(sigmas > 0)


def test_sample_bias_and_sigma_zero_samples() -> None:
    rng = np.random.default_rng(42)
    model = DeveloperDurationModel()
    model.add_completed(m=1, actual_working_days=2.0)

    bias, sigmas = sample_bias_and_sigma(model, n=0, rng=rng)

    assert bias.shape == (0,)
    assert sigmas.shape == (0,)


def test_sample_bias_and_sigma_works_with_no_completed_tasks() -> None:
    rng = np.random.default_rng(11)
    model = DeveloperDurationModel(alpha0=2.5, beta0=0.4, mu0=0.1, kappa0=3.0)

    bias, sigmas = sample_bias_and_sigma(model, n=2, rng=rng)

    assert bias.shape == (2,)
    assert sigmas.shape == (2,)
    # Expected draw from the priors (mu0, alpha0, beta0, kappa0)
    expected_rng = np.random.default_rng(11)
    tau = expected_rng.gamma(shape=2.5, scale=1.0 / 0.4, size=2)
    sigma2 = 1.0 / tau
    expected_sigmas = np.sqrt(sigma2)
    expected_bias = expected_rng.normal(loc=0.1, scale=np.sqrt(sigma2 / 3.0), size=2)
    np.testing.assert_allclose(sigmas, expected_sigmas, rtol=1e-8)
    np.testing.assert_allclose(bias, expected_bias, rtol=1e-8)


def test_sample_bias_and_sigma_respects_posterior_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeveloperDurationModel()
    rng = np.random.default_rng(7)
    expected_rng = np.random.default_rng(7)

    monkeypatch.setattr(task_prob, "posterior_params", lambda m: (1.0, 2.0, 3.0, 0.4))

    bias, sigmas = sample_bias_and_sigma(model, n=4, rng=rng)

    tau = expected_rng.gamma(shape=3.0, scale=1.0 / 0.4, size=4)
    sigma2 = 1.0 / tau
    expected_sigmas = np.sqrt(sigma2)
    expected_bias = expected_rng.normal(loc=1.0, scale=np.sqrt(sigma2 / 2.0), size=4)
    np.testing.assert_allclose(sigmas, expected_sigmas, rtol=1e-8)
    np.testing.assert_allclose(bias, expected_bias, rtol=1e-8)


def test_p_within_multiplier_is_mean_of_cdf_for_fixed_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeveloperDurationModel()
    fixed_bias = np.array([0.0, 0.0])
    fixed_sigmas = np.array([0.5, 1.0])

    def _fake_sample_bias_and_sigma(model_arg, n_samples: int, rng=None):  # type: ignore[override]
        return fixed_bias, fixed_sigmas

    monkeypatch.setattr(task_prob, "sample_bias_and_sigma", _fake_sample_bias_and_sigma)

    prob = p_within_multiplier(model, multiplier=1.0, n_samples=2)
    assert isclose(prob, 0.5, rel_tol=1e-12)


def test_p_within_multiplier_handles_multiplier_below_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeveloperDurationModel()
    fixed_bias = np.array([0.0])
    fixed_sigmas = np.array([0.5])

    monkeypatch.setattr(
        task_prob,
        "sample_bias_and_sigma",
        lambda m, n_samples, rng=None: (fixed_bias, fixed_sigmas),
    )

    prob = p_within_multiplier(model, multiplier=0.5, n_samples=1)
    expected = normal_cdf(log(0.5) / fixed_sigmas[0])
    assert isclose(prob, expected, rel_tol=1e-12)


def test_p_within_multiplier_raises_for_non_positive_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeveloperDurationModel()
    monkeypatch.setattr(
        task_prob,
        "sample_bias_and_sigma",
        lambda m, n_samples, rng=None: (np.array([0.0]), np.array([1.0])),
    )

    with pytest.raises(ValueError):
        p_within_multiplier(model, multiplier=0.0, n_samples=1)


def test_probability_finish_by_due_rejects_today_before_start() -> None:
    model = DeveloperDurationModel()

    with pytest.raises(ValueError):
        probability_finish_by_due(
            model,
            start=date(2024, 12, 2),
            due=date(2024, 12, 5),
            today=date(2024, 12, 1),
            m=1.0,
            n_samples=10,
        )


def test_probability_finish_by_due_rejects_due_on_or_before_start() -> None:
    model = DeveloperDurationModel()

    with pytest.raises(ValueError):
        probability_finish_by_due(
            model,
            start=date(2024, 12, 2),
            due=date(2024, 12, 2),
            today=date(2024, 12, 3),
            m=1.0,
            n_samples=10,
        )


def test_probability_finish_by_due_returns_zero_when_due_not_after_progress() -> None:
    model = DeveloperDurationModel()
    start = date(2024, 12, 2)  # Monday
    today = date(2024, 12, 4)  # Wednesday
    due = date(2024, 12, 4)  # end exclusive -> D equals t

    prob = probability_finish_by_due(
        model, start=start, due=due, today=today, m=1.0, n_samples=10
    )

    assert prob == 0.0


def test_probability_finish_by_due_matches_manual_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeveloperDurationModel()
    start = date(2024, 12, 2)  # Monday
    today = date(2024, 12, 3)  # Tuesday -> t = 1 working day
    due = date(2024, 12, 6)  # Friday (exclusive) -> D = 4 working days
    fixed_bias = np.array([0.1, -0.2])
    fixed_sigmas = np.array([0.5, 1.0])

    monkeypatch.setattr(
        task_prob,
        "sample_bias_and_sigma",
        lambda m, n_samples=0, rng=None: (fixed_bias, fixed_sigmas),
    )

    prob = probability_finish_by_due(
        model, start=start, due=due, today=today, m=1.0, n_samples=2
    )

    t = float(working_days_between(start, today))
    D = float(working_days_between(start, due))
    expected = []
    for b, s in zip(fixed_bias, fixed_sigmas):
        Ft = lognormal_cdf(t, median=1.0, sigma=s, bias=b)
        FD = lognormal_cdf(D, median=1.0, sigma=s, bias=b)
        expected.append((FD - Ft) / (1.0 - Ft))
    assert isclose(prob, sum(expected) / len(expected), rel_tol=1e-12)


def test_realistic_estimated_days_percentile_with_zero_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeveloperDurationModel()
    z = normal_quantile(0.95)
    bias = np.zeros(5)
    sigma = np.full(5, 0.5)

    monkeypatch.setattr(
        task_prob, "sample_bias_and_sigma", lambda m, n=50000: (bias, sigma)
    )

    expected_samples = 3.0 / np.exp(bias + z * sigma)
    expected = np.percentile(expected_samples, 10)

    assert isclose(
        task_duration_estimated_days(model, H=3.0, p=0.95, n_samples=5),
        expected,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_realistic_estimated_days_handles_varied_bias_and_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeveloperDurationModel()
    z = normal_quantile(0.9)
    bias = np.array([-0.2, 0.1, 0.3, -0.1])
    sigma = np.array([0.4, 0.6, 0.5, 0.7])

    monkeypatch.setattr(
        task_prob, "sample_bias_and_sigma", lambda m, n=50000: (bias, sigma)
    )

    samples = 5.0 / np.exp(bias + z * sigma)
    expected = np.percentile(samples, 10)

    assert isclose(
        task_duration_estimated_days(model, H=5.0, p=0.9, n_samples=4),
        expected,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_realistic_estimated_days_respects_p_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeveloperDurationModel()
    # p=0.5 -> z=0, so percentile should ignore sigma and depend only on bias
    bias = np.array([0.0, 0.5, -0.5, 1.0])
    sigma = np.ones_like(bias)

    monkeypatch.setattr(
        task_prob, "sample_bias_and_sigma", lambda m, n=50000: (bias, sigma)
    )

    samples = 7.0 / np.exp(bias)  # z=0 => multiplier uses only bias
    expected = np.percentile(samples, 10)

    assert isclose(
        task_duration_estimated_days(model, H=7.0, p=0.5, n_samples=len(bias)),
        expected,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_fit_inv_gamma_prior_for_small_prior_equiv_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_brent_root(fn, left, right):
        assert left == 0.01 and right == 10.0
        # Call objective once to ensure it runs without error
        fn(0.5)
        return 0.5

    monkeypatch.setattr(task_prob, "brent_root", fake_brent_root)

    alpha0, beta0 = calibrate_prior(prior_equiv_tasks=0)

    assert isclose(alpha0, 1.0)  # 1 + prior_equiv_tasks/2 with prior_equiv_tasks=0
    assert isclose(beta0, 0.5)


def test_fit_inv_gamma_prior_for_large_prior_equiv_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_brent_root(fn, left, right):
        fn(2.5)
        return 2.5

    monkeypatch.setattr(task_prob, "brent_root", fake_brent_root)

    alpha0, beta0 = calibrate_prior(prior_equiv_tasks=20)

    expected_alpha0 = 1.0 + 20 / 2.0  # 1 + n / 2
    assert isclose(alpha0, expected_alpha0)
    assert isclose(beta0, 2.5)
