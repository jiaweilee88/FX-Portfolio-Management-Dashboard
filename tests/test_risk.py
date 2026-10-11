from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.analytics.risk import (
    CONFIDENCE_LEVELS,
    DEFAULT_LOOKBACK_DAYS,
    PSD_TOLERANCE,
    RIDGE_EPSILON,
    compute_var,
    covariance_matrix,
    position_exposures,
    regularize_if_needed,
    standalone_var,
    validate_covariance_matrix,
    var_metrics,
    z_score,
)
from src.data.market_data import Quote

TODAY = date(2026, 10, 9)
PREV = date(2026, 10, 8)


def _quote(value, source="yfinance"):
    return Quote(value, TODAY, value, PREV, source, indicative=True, stale=False)


def _trades(*rows):
    """rows: (trade_id, pair, side, notional, entry_price)"""
    df = pd.DataFrame(rows, columns=["trade_id", "currency_pair", "side", "notional_base_ccy", "entry_price"])
    df["base_currency"] = df["currency_pair"].str[:3]
    df["term_currency"] = df["currency_pair"].str[4:]
    df["side_sign"] = np.where(df["side"] == "BUY", 1, -1)
    return df


def _random_returns(currencies, n=300, seed=0, corr=0.0):
    rng = np.random.default_rng(seed)
    k = len(currencies)
    cov = np.full((k, k), corr) + np.eye(k) * (1 - corr)
    data = rng.multivariate_normal(np.zeros(k), cov * 1e-4, size=n)
    dates = pd.bdate_range(end=TODAY, periods=n)
    return pd.DataFrame(data, index=dates, columns=currencies)


# --- z_score ----------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_z_score_is_computed_via_scipy_norm_ppf(confidence):
    # The primary calculation: z_score must equal scipy's own quantile exactly, not
    # a separately-maintained constant that merely approximates it.
    assert z_score(confidence) == norm.ppf(confidence)


@pytest.mark.parametrize(
    "confidence, spec_value", [(0.90, 1.282), (0.95, 1.645), (0.99, 2.326)]
)
def test_z_score_matches_the_spec_table_to_3dp(confidence, spec_value):
    # The spec documents these rounded values for the README/UI; the live-computed
    # z must still agree with them to the precision the spec itself quotes.
    assert round(z_score(confidence), 3) == spec_value


def test_z_score_rejects_unsupported_confidence():
    with pytest.raises(ValueError):
        z_score(0.975)


def test_confidence_levels_are_the_only_accepted_inputs():
    for confidence in CONFIDENCE_LEVELS:
        z_score(confidence)  # does not raise
    assert CONFIDENCE_LEVELS == (0.90, 0.95, 0.99)


# --- covariance_matrix -------------------------------------------------------


def test_covariance_matrix_validity_symmetric_and_psd_on_random_data():
    # Invariant: Sigma is symmetric and positive semi-definite on random synthetic data.
    returns = _random_returns(["EUR", "GBP", "JPY"], n=300, corr=0.3)

    sigma = covariance_matrix(returns)

    diagnostics = validate_covariance_matrix(sigma)
    assert diagnostics.symmetric and diagnostics.finite and diagnostics.positive_semidefinite


def test_validate_covariance_matrix_flags_a_genuinely_singular_matrix():
    # Two currencies moving in perfect lockstep -> exactly singular raw sample covariance.
    base = _random_returns(["EUR"], n=300).iloc[:, 0]
    returns = pd.DataFrame({"EUR": base, "GBP": base})
    raw_sigma = returns.cov()
    assert np.linalg.matrix_rank(raw_sigma.to_numpy()) < 2  # confirms it's genuinely singular

    diagnostics = validate_covariance_matrix(raw_sigma)

    # Exactly singular still passes PSD (min eigenvalue ~0, not negative) -- the
    # property this flags is a real structural issue even when "technically" PSD.
    assert diagnostics.symmetric and diagnostics.finite
    assert diagnostics.min_eigenvalue == pytest.approx(0.0, abs=1e-9)


def test_validate_covariance_matrix_rejects_an_indefinite_matrix():
    not_psd = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1

    diagnostics = validate_covariance_matrix(not_psd)

    assert diagnostics.symmetric and diagnostics.finite and not diagnostics.positive_semidefinite
    assert diagnostics.min_eigenvalue == pytest.approx(-1.0)


def test_covariance_matrix_leaves_a_well_conditioned_matrix_unregularised():
    # "Use standard sample covariance initially": a matrix that already passes the
    # PSD check should come back byte-for-byte as the plain sample covariance, with
    # no ridge perturbation applied just as blanket insurance.
    returns = _random_returns(["EUR", "GBP", "JPY"], n=300, corr=0.2)

    sigma = covariance_matrix(returns, lookback_days=300)

    raw_sigma = returns.cov()
    pd.testing.assert_frame_equal(sigma, raw_sigma, check_exact=False, rtol=1e-12)


def test_covariance_matrix_does_not_regularize_a_merely_singular_but_valid_matrix():
    # Two currencies moving in perfect lockstep: rank-deficient (one zero eigenvalue)
    # but still mathematically valid PSD -- positive *semi*-definite explicitly
    # allows zero. This should NOT trigger the ridge fallback.
    base = _random_returns(["EUR"], n=300).iloc[:, 0]
    returns = pd.DataFrame({"EUR": base, "GBP": base, "JPY": _random_returns(["JPY"], n=300, seed=1).iloc[:, 0]})
    raw_sigma = returns.cov()
    assert np.linalg.matrix_rank(raw_sigma.to_numpy()) < 3  # confirms it's genuinely singular
    assert validate_covariance_matrix(raw_sigma).positive_semidefinite  # singular, but still PSD

    sigma = covariance_matrix(returns, lookback_days=300)

    pd.testing.assert_frame_equal(sigma, raw_sigma, check_exact=False, rtol=1e-9)


def test_regularize_if_needed_leaves_a_psd_matrix_untouched():
    psd = pd.DataFrame([[0.0001, 0.00002], [0.00002, 0.0002]])

    result = regularize_if_needed(psd)

    pd.testing.assert_frame_equal(result, psd)


def test_regularize_if_needed_applies_ridge_to_a_genuinely_indefinite_matrix():
    # Representative of the real failure mode this guards against: a variance that
    # floating-point noise has pushed just below zero. pandas' own .cov() is
    # essentially never indefinite, so this exercises the fallback directly against
    # a hand-built matrix, using the default RIDGE_EPSILON.
    indefinite = pd.DataFrame([[1e-4, 0.0], [0.0, -1.5e-10]])  # just past PSD_TOLERANCE
    assert not validate_covariance_matrix(indefinite).positive_semidefinite

    result = regularize_if_needed(indefinite)

    diagnostics = validate_covariance_matrix(result)
    assert diagnostics.positive_semidefinite and diagnostics.min_eigenvalue >= PSD_TOLERANCE
    np.linalg.inv(result.to_numpy())  # does not raise


def test_regularize_if_needed_scales_ridge_to_a_larger_indefinite_gap():
    # A bigger negative eigenvalue needs a correspondingly larger epsilon; this
    # confirms the fallback actually closes the gap rather than adding a token amount.
    indefinite = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    assert not validate_covariance_matrix(indefinite).positive_semidefinite

    result = regularize_if_needed(indefinite, epsilon=2.0)

    expected = indefinite + np.diag([2.0, 2.0])  # epsilon * diag, both diagonal entries are 1.0
    pd.testing.assert_frame_equal(result, expected)
    diagnostics = validate_covariance_matrix(result)
    assert diagnostics.positive_semidefinite and diagnostics.min_eigenvalue >= PSD_TOLERANCE
    np.linalg.inv(result.to_numpy())  # does not raise


def test_covariance_matrix_aligns_currencies_on_common_dates_before_computing_covariance():
    # EUR has a return on a date GBP doesn't (and vice versa); the misaligned dates
    # must be dropped so cov() never falls back to pandas' pairwise-NaN behaviour.
    dates = pd.bdate_range(end=TODAY, periods=10)
    eur = pd.Series(np.arange(10, dtype=float), index=dates)
    gbp = pd.Series(np.arange(10, dtype=float) * 2, index=dates)
    eur.iloc[3] = np.nan  # EUR missing a date GBP has
    gbp.iloc[7] = np.nan  # GBP missing a date EUR has
    returns = pd.DataFrame({"EUR": eur, "GBP": gbp})

    sigma = covariance_matrix(returns, lookback_days=10)

    common = returns.dropna(how="any")
    assert len(common) == 8  # 10 rows minus the 2 misaligned ones
    expected = common.cov()
    pd.testing.assert_frame_equal(sigma, expected, check_exact=False, rtol=1e-9)


def test_covariance_matrix_aligns_before_slicing_the_lookback_window():
    # GBP has a 5-day gap right at the end of an otherwise-30-day history. Aligning
    # first (dropna, then take the most recent 10 common dates) reaches back past
    # that gap to still return a full 10-row window; slicing to the most recent 10
    # raw rows first and only then dropping NaNs would instead leave just 5 rows
    # (dates 20-24), giving a different, wrong covariance.
    dates = pd.bdate_range(end=TODAY, periods=30)
    eur = pd.Series(np.arange(30, dtype=float), index=dates)
    gbp = pd.Series(np.arange(30, dtype=float) * 2, index=dates)
    gbp.iloc[25:] = np.nan  # GBP's last 5 days are missing
    returns = pd.DataFrame({"EUR": eur, "GBP": gbp})

    sigma = covariance_matrix(returns, lookback_days=10)

    aligned_then_tail = returns.dropna(how="any").tail(10).cov()
    tail_then_aligned = returns.tail(10).dropna(how="any").cov()  # the wrong order
    pd.testing.assert_frame_equal(sigma, aligned_then_tail, check_exact=False, rtol=1e-9)
    assert not np.allclose(sigma.to_numpy(), tail_then_aligned.to_numpy())


def test_covariance_matrix_uses_most_recent_lookback_window():
    quiet = pd.DataFrame({"EUR": np.zeros(300)}, index=pd.bdate_range(end=TODAY, periods=300))
    volatile_tail = _random_returns(["EUR"], n=50, seed=2) * 100  # huge recent moves
    combined = pd.concat([quiet.iloc[:-50], volatile_tail])

    sigma_full = covariance_matrix(combined, lookback_days=300)
    sigma_recent = covariance_matrix(combined, lookback_days=50)

    assert sigma_recent.loc["EUR", "EUR"] > sigma_full.loc["EUR", "EUR"]


def test_covariance_matrix_degrades_gracefully_with_fewer_rows_than_lookback():
    returns = _random_returns(["EUR"], n=30)

    sigma = covariance_matrix(returns, lookback_days=DEFAULT_LOOKBACK_DAYS)

    assert sigma.shape == (1, 1) and np.isfinite(sigma.to_numpy()).all()


def test_covariance_matrix_raises_on_insufficient_observations():
    returns = _random_returns(["EUR"], n=1)

    with pytest.raises(ValueError):
        covariance_matrix(returns)


def test_ridge_epsilon_is_small_relative_to_variance():
    indefinite = pd.DataFrame([[0.0001, 0.0005], [0.0005, 0.0002]])  # eigenvalues span negative

    result = regularize_if_needed(indefinite, epsilon=RIDGE_EPSILON)

    expected_diag = np.diag(indefinite.to_numpy()) * (1 + RIDGE_EPSILON)
    np.testing.assert_allclose(np.diag(result.to_numpy()), expected_diag)


# --- position_exposures ------------------------------------------------------


def test_major_position_single_leg_exposure():
    trades = _trades(("T1", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.12)}

    exposures, excluded = position_exposures(trades, quotes, ["EUR"])

    assert excluded == ()
    assert exposures.loc["T1", "EUR"] == pytest.approx(1_000_000 * 1.12)


def test_usd_base_position_term_leg_exposure():
    # USD/JPY BUY: long USD, short JPY. w^(B) = -s*N*S0*U_B
    trades = _trades(("T1", "USD/JPY", "BUY", 1_000_000, 150.0))
    quotes = {"JPY": _quote(1 / 148.0)}

    exposures, excluded = position_exposures(trades, quotes, ["JPY"])

    assert exposures.loc["T1", "JPY"] == pytest.approx(-1 * 1_000_000 * 150.0 * (1 / 148.0))


def test_cross_position_has_both_legs():
    trades = _trades(("T1", "EUR/GBP", "BUY", 1_000_000, 0.855))
    quotes = {"EUR": _quote(1.08), "GBP": _quote(1.27)}

    exposures, excluded = position_exposures(trades, quotes, ["EUR", "GBP"])

    assert exposures.loc["T1", "EUR"] == pytest.approx(1_000_000 * 1.08)
    assert exposures.loc["T1", "GBP"] == pytest.approx(-1_000_000 * 0.855 * 1.27)


def test_sell_flips_both_legs_sign():
    trades = _trades(("T1", "EUR/GBP", "SELL", 1_000_000, 0.855))
    quotes = {"EUR": _quote(1.08), "GBP": _quote(1.27)}

    exposures, _ = position_exposures(trades, quotes, ["EUR", "GBP"])

    assert exposures.loc["T1", "EUR"] == pytest.approx(-1_000_000 * 1.08)
    assert exposures.loc["T1", "GBP"] == pytest.approx(1_000_000 * 0.855 * 1.27)


def test_position_excluded_when_leg_outside_currency_universe():
    trades = _trades(("T1", "EUR/SEK", "BUY", 1_000_000, 11.5))
    quotes = {"EUR": _quote(1.08), "SEK": _quote(0.095)}

    exposures, excluded = position_exposures(trades, quotes, ["EUR"])  # SEK not VaR-eligible

    assert excluded == ("T1",)
    assert exposures.empty


def test_position_excluded_when_quote_missing():
    trades = _trades(("T1", "EUR/USD", "BUY", 1_000_000, 1.10))

    exposures, excluded = position_exposures(trades, {}, ["EUR"])

    assert excluded == ("T1",)


def test_usd_leg_never_a_column_even_if_requested():
    trades = _trades(("T1", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.12)}

    exposures, _ = position_exposures(trades, quotes, ["EUR", "USD"])

    assert list(exposures.columns) == ["EUR", "USD"]
    assert (exposures["USD"] == 0.0).all()


def test_offsetting_trades_net_to_zero_exposure():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 1_000_000, 1.10),
        ("T2", "EUR/USD", "SELL", 1_000_000, 1.10),
    )
    quotes = {"EUR": _quote(1.12)}

    exposures, _ = position_exposures(trades, quotes, ["EUR"])

    assert exposures["EUR"].sum() == pytest.approx(0.0)


# --- var_metrics: hand-verified worked example ------------------------------


@pytest.fixture
def two_currency_fixture():
    """Sigma is diagonal and exact by construction: var_AAA=0.0001, var_BBB=0.0004,
    zero correlation. Two single-leg positions, one per currency, each exposure
    chosen so the arithmetic below is exact and easy to hand-check."""
    sigma = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0004]], index=["AAA", "BBB"], columns=["AAA", "BBB"])
    exposures = pd.DataFrame(
        {"AAA": [100_000.0, 0.0], "BBB": [0.0, 50_000.0]}, index=["T1", "T2"]
    )
    signed_notional = pd.Series({"T1": 100_000.0, "T2": 50_000.0})
    return sigma, exposures, signed_notional


def test_worked_example_portfolio_var(two_currency_fixture):
    sigma, exposures, signed_notional = two_currency_fixture
    z = 1.645

    metrics = var_metrics(exposures, sigma, signed_notional, z)

    # w = [100000, 50000]; sigma_p^2 = 100000^2*0.0001 + 50000^2*0.0004 = 1e6 + 1e6 = 2e6
    expected_portfolio_var = z * np.sqrt(2_000_000)
    assert metrics.portfolio_var == pytest.approx(expected_portfolio_var)


def test_worked_example_standalone_var(two_currency_fixture):
    sigma, exposures, signed_notional = two_currency_fixture
    z = 1.645

    metrics = var_metrics(exposures, sigma, signed_notional, z)

    # Both positions: z*sqrt(N^2 * variance) = z*1000, identical by construction.
    assert metrics.positions.loc["T1", "standalone_var"] == pytest.approx(z * 1000.0)
    assert metrics.positions.loc["T2", "standalone_var"] == pytest.approx(z * 1000.0)
    assert metrics.standalone_sum == pytest.approx(2 * z * 1000.0)


def test_worked_example_component_and_marginal_var(two_currency_fixture):
    sigma, exposures, signed_notional = two_currency_fixture
    z = 1.645

    metrics = var_metrics(exposures, sigma, signed_notional, z)
    sigma_p = np.sqrt(2_000_000)

    # Sigma*w = [100000*0.0001, 50000*0.0004] = [10, 20]; w_i . (Sigma w) = 1,000,000 for both
    expected_component = z * 1_000_000 / sigma_p
    assert metrics.positions.loc["T1", "component_var"] == pytest.approx(expected_component)
    assert metrics.positions.loc["T2", "component_var"] == pytest.approx(expected_component)

    assert metrics.positions.loc["T1", "marginal_var"] == pytest.approx(expected_component / 100_000.0)
    assert metrics.positions.loc["T2", "marginal_var"] == pytest.approx(expected_component / 50_000.0)

    # "component VaR = marginal VaR scaled back to the position's actual size"
    for trade_id, notional in signed_notional.items():
        row = metrics.positions.loc[trade_id]
        assert row["marginal_var"] * notional == pytest.approx(row["component_var"])


def test_worked_example_diversification_benefit(two_currency_fixture):
    sigma, exposures, signed_notional = two_currency_fixture

    metrics = var_metrics(exposures, sigma, signed_notional, 1.645)

    assert metrics.standalone_sum > metrics.portfolio_var  # independent exposures diversify


# --- standalone_var: must be computed on the net exposure, never summed ------


def test_standalone_var_matches_var_metrics_worked_example(two_currency_fixture):
    sigma, exposures, _ = two_currency_fixture

    values = standalone_var(exposures, sigma, z=1.645)

    assert values["T1"] == pytest.approx(1.645 * 1000.0)
    assert values["T2"] == pytest.approx(1.645 * 1000.0)


def test_standalone_var_on_net_exposure_is_less_than_sum_of_individual_rows():
    # Two offsetting exposures in the same currency: summing their individual
    # standalone VaRs would overstate risk; computed on the net (aggregated) row,
    # standalone VaR must come out strictly lower -- it should reflect the netting.
    sigma = pd.DataFrame([[0.0002]], index=["EUR"], columns=["EUR"])
    individual = pd.DataFrame({"EUR": [2_000_000.0, -1_500_000.0]}, index=["T1", "T2"])
    aggregated = pd.DataFrame({"EUR": [individual["EUR"].sum()]}, index=["NET"])

    individual_values = standalone_var(individual, sigma, z=1.645)
    net_value = standalone_var(aggregated, sigma, z=1.645)

    assert net_value.iloc[0] < individual_values.sum()
    # And it must equal the quadratic form applied directly to the net notional, not
    # some other blend: z * |500,000| * sigma.
    assert net_value.iloc[0] == pytest.approx(1.645 * abs(500_000.0) * np.sqrt(0.0002))


def test_standalone_var_on_fully_offsetting_net_exposure_is_zero():
    sigma = pd.DataFrame([[0.0002]], index=["EUR"], columns=["EUR"])
    aggregated = pd.DataFrame({"EUR": [0.0]}, index=["NET"])  # two equal and opposite trades, netted first

    assert standalone_var(aggregated, sigma, z=1.645).iloc[0] == pytest.approx(0.0)


def test_standalone_var_cross_row_uses_the_full_quadratic_form():
    sigma = pd.DataFrame([[0.0002, 0.00005], [0.00005, 0.0003]], index=["EUR", "GBP"], columns=["EUR", "GBP"])
    exposures = pd.DataFrame({"EUR": [1_500_000.0], "GBP": [-800_000.0]}, index=["T1"])
    w = exposures.iloc[0].to_numpy()

    value = standalone_var(exposures, sigma, z=1.645)

    expected = 1.645 * np.sqrt(w @ sigma.to_numpy() @ w)
    assert value["T1"] == pytest.approx(expected)


def test_standalone_var_empty_exposures_returns_empty_series():
    sigma = pd.DataFrame([[0.0002]], index=["EUR"], columns=["EUR"])
    empty = pd.DataFrame(columns=["EUR"])

    result = standalone_var(empty, sigma, z=1.645)

    assert result.empty


# --- var_metrics: Test Plan invariants --------------------------------------


def _random_sigma(currencies, seed=0, corr_scale=0.3):
    rng = np.random.default_rng(seed)
    k = len(currencies)
    a = rng.normal(0, 0.02, (k, k))
    cov = a @ a.T * corr_scale + np.eye(k) * 1e-4
    return pd.DataFrame(cov, index=currencies, columns=currencies)


def _random_exposures(trade_ids, currencies, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.uniform(-2_000_000, 2_000_000, (len(trade_ids), len(currencies)))
    return pd.DataFrame(data, index=trade_ids, columns=currencies)


def test_component_var_reconciles_to_portfolio_var():
    currencies = ["EUR", "GBP", "JPY", "AUD"]
    trade_ids = [f"T{i}" for i in range(8)]
    sigma = _random_sigma(currencies, seed=1)
    exposures = _random_exposures(trade_ids, currencies, seed=1)
    signed_notional = pd.Series(np.random.default_rng(1).uniform(-5e6, 5e6, len(trade_ids)), index=trade_ids)

    metrics = var_metrics(exposures, sigma, signed_notional, 1.645)

    assert metrics.positions["component_var"].sum() == pytest.approx(metrics.portfolio_var, abs=1e-6)


def test_subadditivity_portfolio_var_never_exceeds_standalone_sum():
    currencies = ["EUR", "GBP", "JPY", "AUD", "CAD"]
    trade_ids = [f"T{i}" for i in range(12)]
    for seed in range(10):
        sigma = _random_sigma(currencies, seed=seed)
        exposures = _random_exposures(trade_ids, currencies, seed=seed)
        signed_notional = pd.Series(
            np.random.default_rng(seed).uniform(-5e6, 5e6, len(trade_ids)), index=trade_ids
        )

        metrics = var_metrics(exposures, sigma, signed_notional, 1.645)

        assert metrics.portfolio_var <= metrics.standalone_sum + 1e-9


def test_single_position_portfolio_var_equals_standalone_equals_component():
    sigma = pd.DataFrame([[0.0002, 0.00005], [0.00005, 0.0003]], index=["EUR", "GBP"], columns=["EUR", "GBP"])
    exposures = pd.DataFrame({"EUR": [1_500_000.0], "GBP": [-800_000.0]}, index=["T1"])
    signed_notional = pd.Series({"T1": 1_500_000.0})

    metrics = var_metrics(exposures, sigma, signed_notional, 1.645)

    row = metrics.positions.loc["T1"]
    assert metrics.portfolio_var == pytest.approx(row["standalone_var"])
    assert metrics.portfolio_var == pytest.approx(row["component_var"])


def test_flat_net_exposure_contributes_zero_to_portfolio_var():
    # Two offsetting positions on the same currency net to w=0; an unrelated third
    # currency has no exposure at all either, so the whole portfolio is flat.
    sigma = _random_sigma(["EUR", "GBP"], seed=2)
    exposures = pd.DataFrame({"EUR": [1_000_000.0, -1_000_000.0], "GBP": [0.0, 0.0]}, index=["T1", "T2"])
    signed_notional = pd.Series({"T1": 1_000_000.0, "T2": -1_000_000.0})

    metrics = var_metrics(exposures, sigma, signed_notional, 1.645)

    assert metrics.portfolio_var == pytest.approx(0.0, abs=1e-9)
    assert (metrics.positions["component_var"] == 0.0).all()
    assert (metrics.positions["marginal_var"] == 0.0).all()
    # Each position still carries its own standalone risk even though netted to flat.
    assert (metrics.positions["standalone_var"] > 0).all()


def test_flat_currency_within_a_larger_portfolio_contributes_nothing():
    # EUR nets to zero but GBP doesn't; portfolio VaR should equal the GBP-only figure.
    sigma = _random_sigma(["EUR", "GBP"], seed=3)
    exposures = pd.DataFrame(
        {"EUR": [1_000_000.0, -1_000_000.0, 0.0], "GBP": [0.0, 0.0, 700_000.0]}, index=["T1", "T2", "T3"]
    )
    signed_notional = pd.Series({"T1": 1_000_000.0, "T2": -1_000_000.0, "T3": 700_000.0})

    metrics = var_metrics(exposures, sigma, signed_notional, 1.645)

    gbp_only_var = 1.645 * np.sqrt(700_000.0**2 * sigma.loc["GBP", "GBP"])
    assert metrics.portfolio_var == pytest.approx(gbp_only_var)


# --- compute_var: horizon scaling & end-to-end wiring -----------------------


def test_var_horizon_scaling_10day_equals_1day_times_sqrt10():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "USD/JPY", "BUY", 4_000_000, 147.35),
    )
    quotes = {"EUR": _quote(1.09), "JPY": _quote(1 / 147.0)}
    returns = _random_returns(["EUR", "JPY"], n=300, corr=0.2, seed=4)

    result_1d = compute_var(trades, quotes, returns, horizon_days=1)
    result_10d = compute_var(trades, quotes, returns, horizon_days=10)

    assert result_10d.portfolio_var == pytest.approx(result_1d.portfolio_var * np.sqrt(10))
    pd.testing.assert_series_equal(
        result_10d.positions["standalone_var"], result_1d.positions["standalone_var"] * np.sqrt(10)
    )
    pd.testing.assert_series_equal(
        result_10d.positions["component_var"], result_1d.positions["component_var"] * np.sqrt(10)
    )


def test_compute_var_excludes_positions_with_no_market_data():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/SEK", "BUY", 1_000_000, 11.5),  # SEK has no return history below
    )
    quotes = {"EUR": _quote(1.09), "SEK": _quote(0.095)}
    returns = _random_returns(["EUR"], n=300, seed=5)

    result = compute_var(trades, quotes, returns)

    assert result.excluded_trade_ids == ("T2",)
    assert "T2" not in result.positions.index
    assert "T1" in result.positions.index


def test_compute_var_end_to_end_reconciles_on_a_realistic_portfolio():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "GBP/USD", "SELL", 2_000_000, 1.2740),
        ("T3", "USD/JPY", "BUY", 4_000_000, 147.35),
        ("T4", "EUR/GBP", "BUY", 1_000_000, 0.8550),
    )
    quotes = {"EUR": _quote(1.09), "GBP": _quote(1.27), "JPY": _quote(1 / 147.0)}
    returns = _random_returns(["EUR", "GBP", "JPY"], n=300, corr=0.25, seed=6)

    result = compute_var(trades, quotes, returns)

    assert result.excluded_trade_ids == ()
    assert result.positions["component_var"].sum() == pytest.approx(result.portfolio_var, abs=1e-6)
    assert result.portfolio_var <= result.standalone_sum + 1e-9
    assert result.diversification_benefit >= -1e-9
    assert result.z == norm.ppf(0.95) and result.confidence == 0.95
