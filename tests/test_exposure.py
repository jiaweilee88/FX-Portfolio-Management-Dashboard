from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analytics.exposure import (
    compute_currency_exposure,
    pair_diversification_benefit,
    pair_standalone_sum,
    pair_summary,
    total_component_var,
    total_pct_of_portfolio_var,
    var_concentration,
)
from src.analytics.pnl import compute_position_pnl
from src.analytics.risk import compute_var
from src.data.market_data import Quote

TODAY = date(2026, 10, 9)
PREV = date(2026, 10, 8)


def _quote(value, source="yfinance"):
    return Quote(value, TODAY, value, PREV, source, indicative=True, stale=False)


# pnl.compute_position_pnl (unlike compute_currency_exposure/risk.position_exposures)
# does not special-case USD: it must be a literal key in the quotes dict passed in.
USD = Quote(1.0, TODAY, 1.0, PREV, "USD (unit)", indicative=False, stale=False)


def _trades(*rows):
    """rows: (trade_id, pair, side, notional, entry_price)"""
    df = pd.DataFrame(rows, columns=["trade_id", "currency_pair", "side", "notional_base_ccy", "entry_price"])
    df["trade_date"] = pd.Timestamp("2026-09-01")  # well before TODAY/PREV: never a new_position
    df["base_currency"] = df["currency_pair"].str[:3]
    df["term_currency"] = df["currency_pair"].str[4:]
    df["side_sign"] = np.where(df["side"] == "BUY", 1, -1)
    return df


# --- compute_currency_exposure ----------------------------------------------


def test_major_position_contributes_both_legs_including_usd():
    trades = _trades(("T1", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.12)}

    result = compute_currency_exposure(trades, quotes)

    assert result.net_exposure["EUR"] == pytest.approx(1_000_000 * 1.12)
    assert result.net_exposure["USD"] == pytest.approx(-1_000_000 * 1.10)  # -s*N*S0*U_USD
    assert result.excluded_trade_ids == ()


def test_usd_base_position_contributes_both_legs():
    trades = _trades(("T1", "USD/JPY", "BUY", 1_000_000, 150.0))
    quotes = {"JPY": _quote(1 / 148.0)}

    result = compute_currency_exposure(trades, quotes)

    assert result.net_exposure["USD"] == pytest.approx(1_000_000 * 1.0)
    assert result.net_exposure["JPY"] == pytest.approx(-1_000_000 * 150.0 * (1 / 148.0))


def test_cross_position_contributes_both_non_usd_legs_no_usd_row():
    trades = _trades(("T1", "EUR/GBP", "BUY", 1_000_000, 0.855))
    quotes = {"EUR": _quote(1.08), "GBP": _quote(1.27)}

    result = compute_currency_exposure(trades, quotes)

    assert result.net_exposure["EUR"] == pytest.approx(1_000_000 * 1.08)
    assert result.net_exposure["GBP"] == pytest.approx(-1_000_000 * 0.855 * 1.27)
    assert "USD" not in result.net_exposure.index


def test_offsetting_trades_net_to_zero():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 1_000_000, 1.10),
        ("T2", "EUR/USD", "SELL", 1_000_000, 1.10),
    )
    quotes = {"EUR": _quote(1.12)}

    result = compute_currency_exposure(trades, quotes)

    assert result.net_exposure["EUR"] == pytest.approx(0.0)
    assert result.net_exposure["USD"] == pytest.approx(0.0)


def test_position_missing_a_leg_quote_is_excluded_entirely():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 1_000_000, 1.10),
        ("T2", "EUR/SEK", "BUY", 500_000, 11.5),  # SEK unpriced
    )
    quotes = {"EUR": _quote(1.12)}

    result = compute_currency_exposure(trades, quotes)

    assert result.excluded_trade_ids == ("T2",)
    # T2's EUR leg must not leak into the EUR total despite EUR being priced.
    assert result.net_exposure["EUR"] == pytest.approx(1_000_000 * 1.12)


def test_as_frame_percentages_sum_to_100():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "USD/JPY", "BUY", 4_000_000, 147.35),
        ("T3", "GBP/USD", "SELL", 2_000_000, 1.2740),
    )
    quotes = {"EUR": _quote(1.09), "JPY": _quote(1 / 147.0), "GBP": _quote(1.27)}

    frame = compute_currency_exposure(trades, quotes).as_frame()

    assert frame["pct_of_gross"].sum() == pytest.approx(100.0)
    assert (frame["pct_of_gross"] >= 0).all()
    assert frame["net_exposure"].abs().is_monotonic_decreasing


def test_as_frame_handles_zero_gross_exposure_without_dividing_by_zero():
    trades = _trades(("T1", "EUR/USD", "BUY", 1_000_000, 1.10), ("T2", "EUR/USD", "SELL", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.10)}  # at entry: both legs exactly zero

    frame = compute_currency_exposure(trades, quotes).as_frame()

    assert (frame["net_exposure"] == 0.0).all()
    assert (frame["pct_of_gross"] == 0.0).all()


def test_gross_exposure_property():
    trades = _trades(("T1", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.12)}

    result = compute_currency_exposure(trades, quotes)

    assert result.gross_exposure == pytest.approx(1_000_000 * 1.12 + 1_000_000 * 1.10)


# --- var_concentration / var_concentration_by_pair ---------------------------


def _random_returns(currencies, n=300, seed=0, corr=0.2):
    rng = np.random.default_rng(seed)
    k = len(currencies)
    cov = np.full((k, k), corr) + np.eye(k) * (1 - corr)
    data = rng.multivariate_normal(np.zeros(k), cov * 1e-4, size=n)
    dates = pd.bdate_range(end=TODAY, periods=n)
    return pd.DataFrame(data, index=dates, columns=currencies)


def test_var_concentration_percentages_sum_to_100():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "GBP/USD", "SELL", 2_000_000, 1.2740),
        ("T3", "USD/JPY", "BUY", 4_000_000, 147.35),
        ("T4", "EUR/GBP", "BUY", 1_000_000, 0.8550),
    )
    quotes = {"EUR": _quote(1.09), "GBP": _quote(1.27), "JPY": _quote(1 / 147.0)}
    returns = _random_returns(["EUR", "GBP", "JPY"], seed=7)

    result = compute_var(trades, quotes, returns)
    concentration = var_concentration(result, trades)

    assert concentration["pct_of_portfolio_var"].sum() == pytest.approx(100.0, abs=1e-6)
    assert set(concentration.index) == {"T1", "T2", "T3", "T4"}


def test_var_concentration_includes_currency_pair_and_sorts_by_magnitude():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "GBP/USD", "SELL", 100_000, 1.2740),
    )
    quotes = {"EUR": _quote(1.09), "GBP": _quote(1.27)}
    returns = _random_returns(["EUR", "GBP"], seed=8)

    result = compute_var(trades, quotes, returns)
    concentration = var_concentration(result, trades)

    assert list(concentration["currency_pair"]) == ["EUR/USD", "GBP/USD"]
    assert concentration["component_var"].abs().is_monotonic_decreasing


def test_var_concentration_handles_zero_portfolio_var_without_crashing():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 1_000_000, 1.10),
        ("T2", "EUR/USD", "SELL", 1_000_000, 1.10),
    )
    quotes = {"EUR": _quote(1.10)}
    returns = _random_returns(["EUR"], seed=10)

    result = compute_var(trades, quotes, returns)
    assert result.portfolio_var == pytest.approx(0.0, abs=1e-9)

    concentration = var_concentration(result, trades)

    assert (concentration["pct_of_portfolio_var"] == 0.0).all()


# --- pair_summary: the position table's pair-level primary rows -------------


def test_pair_summary_net_notional_sums_signed_notional_per_pair():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/USD", "SELL", 3_000_000, 1.0900),
        ("T3", "GBP/USD", "SELL", 2_000_000, 1.2740),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09), "GBP": _quote(1.27)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR", "GBP"], seed=11))

    summary = pair_summary(trades, quotes, pnl_df, result)

    assert summary.loc["EUR/USD", "net_notional"] == pytest.approx(5_000_000 - 3_000_000)
    assert summary.loc["GBP/USD", "net_notional"] == pytest.approx(-2_000_000)


def test_pair_summary_pnl_rows_equal_sum_of_drilldown_trades_exactly():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/USD", "SELL", 3_000_000, 1.0900),
        ("T3", "GBP/USD", "SELL", 2_000_000, 1.2740),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09), "GBP": _quote(1.27)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR", "GBP"], seed=12))

    summary = pair_summary(trades, quotes, pnl_df, result)

    eurusd_trades = pnl_df[pnl_df["currency_pair"] == "EUR/USD"]
    assert summary.loc["EUR/USD", "inception_pnl"] == pytest.approx(eurusd_trades["inception_pnl"].sum())
    assert summary.loc["EUR/USD", "daily_pnl"] == pytest.approx(eurusd_trades["daily_pnl"].sum())
    gbpusd_trades = pnl_df[pnl_df["currency_pair"] == "GBP/USD"]
    assert summary.loc["GBP/USD", "inception_pnl"] == pytest.approx(gbpusd_trades["inception_pnl"].sum())
    assert summary.loc["GBP/USD", "daily_pnl"] == pytest.approx(gbpusd_trades["daily_pnl"].sum())


def test_pair_summary_component_var_equals_sum_of_drilldown_trades_exactly():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/USD", "SELL", 3_000_000, 1.0900),
        ("T3", "GBP/USD", "SELL", 2_000_000, 1.2740),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09), "GBP": _quote(1.27)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR", "GBP"], seed=13))

    summary = pair_summary(trades, quotes, pnl_df, result)

    expected_eurusd = result.positions.loc[["T1", "T2"], "component_var"].sum()
    assert summary.loc["EUR/USD", "component_var"] == pytest.approx(expected_eurusd)
    expected_gbpusd = result.positions.loc[["T3"], "component_var"].sum()
    assert summary.loc["GBP/USD", "component_var"] == pytest.approx(expected_gbpusd)
    assert summary["pct_of_portfolio_var"].sum() == pytest.approx(100.0, abs=1e-6)


def test_pair_summary_standalone_var_is_not_summed_from_trades_when_offsetting():
    # The invariant this guards: naively summing the pair's trades' own standalone
    # VaR figures would overstate risk whenever they partly offset. Computed
    # correctly (on the net exposure), the pair's standalone VaR must be strictly
    # less than that sum.
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/USD", "SELL", 3_000_000, 1.0900),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR"], seed=14))

    summary = pair_summary(trades, quotes, pnl_df, result)

    sum_of_trade_standalone = result.positions.loc[["T1", "T2"], "standalone_var"].sum()
    assert summary.loc["EUR/USD", "standalone_var"] < sum_of_trade_standalone


def test_pair_summary_standalone_var_matches_direct_computation_on_net_exposure():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/USD", "SELL", 3_000_000, 1.0900),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR"], seed=15))

    summary = pair_summary(trades, quotes, pnl_df, result)

    net_eur_exposure = 5_000_000 * 1.09 - 3_000_000 * 1.09  # w^(A) for each trade, netted
    sigma_eur = result.sigma.loc["EUR", "EUR"]
    expected = result.z * abs(net_eur_exposure) * np.sqrt(sigma_eur) * result.scale
    assert summary.loc["EUR/USD", "standalone_var"] == pytest.approx(expected)


def test_pair_summary_cross_pair_standalone_var_uses_each_trades_own_entry_price():
    # A cross's net exposure can't be derived from summed notional alone -- each
    # trade's own entry price matters. Two EUR/GBP trades at different entry prices
    # must each contribute their own two-leg exposure before summing.
    trades = _trades(
        ("T1", "EUR/GBP", "BUY", 1_000_000, 0.8550),
        ("T2", "EUR/GBP", "SELL", 400_000, 0.8600),
    )
    quotes = {"USD": USD, "EUR": _quote(1.08), "GBP": _quote(1.27)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR", "GBP"], seed=16))

    summary = pair_summary(trades, quotes, pnl_df, result)

    net_eur = 1_000_000 * 1.08 - 400_000 * 1.08
    net_gbp = -1_000_000 * 0.8550 * 1.27 + 400_000 * 0.8600 * 1.27
    w = np.array([net_eur, net_gbp])
    sigma = result.sigma.loc[["EUR", "GBP"], ["EUR", "GBP"]].to_numpy()
    expected = result.z * np.sqrt(w @ sigma @ w) * result.scale
    assert summary.loc["EUR/GBP", "standalone_var"] == pytest.approx(expected)


def test_pair_summary_pair_with_no_market_data_gives_nan_not_zero():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/SEK", "BUY", 500_000, 11.5),  # SEK unpriced
    )
    quotes = {"USD": USD, "EUR": _quote(1.09)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR"], seed=17))

    summary = pair_summary(trades, quotes, pnl_df, result)

    assert np.isnan(summary.loc["EUR/SEK", "standalone_var"])
    assert np.isnan(summary.loc["EUR/SEK", "component_var"])
    assert not np.isnan(summary.loc["EUR/USD", "standalone_var"])


def test_pair_summary_sorted_by_absolute_component_var_descending():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "GBP/USD", "SELL", 100_000, 1.2740),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09), "GBP": _quote(1.27)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR", "GBP"], seed=18))

    summary = pair_summary(trades, quotes, pnl_df, result)

    assert list(summary.index) == ["EUR/USD", "GBP/USD"]
    assert summary["component_var"].abs().is_monotonic_decreasing


def test_pair_summary_handles_zero_portfolio_var_without_crashing():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 1_000_000, 1.10),
        ("T2", "EUR/USD", "SELL", 1_000_000, 1.10),
    )
    quotes = {"USD": USD, "EUR": _quote(1.10)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR"], seed=19))
    assert result.portfolio_var == pytest.approx(0.0, abs=1e-9)

    summary = pair_summary(trades, quotes, pnl_df, result)

    assert (summary["pct_of_portfolio_var"] == 0.0).all()
    assert summary.loc["EUR/USD", "standalone_var"] == pytest.approx(0.0, abs=1e-9)  # net exposure is flat too


# --- reconciliation totals & pair-level diversification benefit -------------


def _realistic_portfolio():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/USD", "SELL", 3_000_000, 1.0900),  # partly offsets T1 on the same pair
        ("T3", "GBP/USD", "SELL", 2_000_000, 1.2740),
        ("T4", "USD/JPY", "BUY", 4_000_000, 147.35),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09), "GBP": _quote(1.27), "JPY": _quote(1 / 147.0)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR", "GBP", "JPY"], seed=20))
    return trades, quotes, pnl_df, result


@pytest.mark.parametrize("lookback", [60, 150, 252, 400, 500])
def test_total_component_var_matches_portfolio_var_at_every_lookback(lookback):
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "GBP/USD", "SELL", 2_000_000, 1.2740),
        ("T3", "USD/JPY", "BUY", 4_000_000, 147.35),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09), "GBP": _quote(1.27), "JPY": _quote(1 / 147.0)}
    pnl_df = compute_position_pnl(trades, quotes)
    returns = _random_returns(["EUR", "GBP", "JPY"], n=600, seed=21)
    result = compute_var(trades, quotes, returns, lookback_days=lookback)

    pair_table = pair_summary(trades, quotes, pnl_df, result)
    trade_table = var_concentration(result, trades)

    assert total_component_var(pair_table) == pytest.approx(result.portfolio_var, abs=1e-6)
    assert total_component_var(trade_table) == pytest.approx(result.portfolio_var, abs=1e-6)
    assert total_pct_of_portfolio_var(pair_table) == pytest.approx(100.0, abs=1e-6)
    assert total_pct_of_portfolio_var(trade_table) == pytest.approx(100.0, abs=1e-6)


def test_pair_standalone_sum_excludes_nan_pairs_not_propagates():
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "EUR/SEK", "BUY", 500_000, 11.5),  # SEK unpriced -> no VaR-eligible trades on this pair
    )
    quotes = {"USD": USD, "EUR": _quote(1.09)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR"], seed=22))

    pair_table = pair_summary(trades, quotes, pnl_df, result)
    assert np.isnan(pair_table.loc["EUR/SEK", "standalone_var"])

    assert pair_standalone_sum(pair_table) == pytest.approx(pair_table.loc["EUR/USD", "standalone_var"])


def test_pair_diversification_benefit_is_less_than_trade_level_when_pair_has_offsetting_trades():
    # The methodology check: T1/T2 partly offset on EUR/USD. Summing their own gross
    # standalone VaRs (trade-level DB) double-counts that netting as diversification;
    # pair-level DB, built on the already-netted pair standalone, must be smaller.
    trades, quotes, pnl_df, result = _realistic_portfolio()

    pair_table = pair_summary(trades, quotes, pnl_df, result)

    trade_level_db = result.diversification_benefit  # risk.VaRResult's own, trade-level
    pair_level_db = pair_diversification_benefit(pair_table, result.portfolio_var)

    assert pair_level_db < trade_level_db
    assert pair_level_db >= -1e-9  # subadditivity still holds at pair level


def test_pair_diversification_benefit_matches_direct_formula():
    trades, quotes, pnl_df, result = _realistic_portfolio()
    pair_table = pair_summary(trades, quotes, pnl_df, result)

    expected = pair_table["standalone_var"].sum() - result.portfolio_var
    assert pair_diversification_benefit(pair_table, result.portfolio_var) == pytest.approx(expected)


def test_pair_diversification_benefit_equals_trade_level_when_no_pair_has_multiple_trades():
    # With exactly one trade per pair, there's no within-pair netting to double-count,
    # so pair-level and trade-level standalone sums -- and DB -- must coincide.
    trades = _trades(
        ("T1", "EUR/USD", "BUY", 5_000_000, 1.0865),
        ("T2", "GBP/USD", "SELL", 2_000_000, 1.2740),
    )
    quotes = {"USD": USD, "EUR": _quote(1.09), "GBP": _quote(1.27)}
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, _random_returns(["EUR", "GBP"], seed=23))

    pair_table = pair_summary(trades, quotes, pnl_df, result)

    assert pair_diversification_benefit(pair_table, result.portfolio_var) == pytest.approx(
        result.diversification_benefit
    )
