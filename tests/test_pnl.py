from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.pnl import compute_position_pnl, portfolio_pnl, position_value
from src.data.market_data import Quote
from src.data.portfolio_loader import load_portfolio

TODAY = date(2026, 10, 9)
MARK = date(2026, 10, 9)
PREV = date(2026, 10, 8)
SAMPLE_PORTFOLIO = Path(__file__).resolve().parent.parent / "data" / "sample_portfolio.csv"


def _quote(value, previous=None, as_of=MARK, previous_as_of=PREV, indicative=True, stale=False):
    return Quote(
        value=value,
        as_of=as_of,
        previous_value=previous if previous is not None else value,
        previous_as_of=previous_as_of,
        source="test",
        indicative=indicative,
        stale=stale,
    )


USD = Quote(1.0, TODAY, 1.0, PREV, "USD (unit)", indicative=False, stale=False)


def _trades(*rows):
    """rows: (trade_id, trade_date, pair, side, notional, entry_price)"""
    df = pd.DataFrame(
        rows, columns=["trade_id", "trade_date", "currency_pair", "side", "notional_base_ccy", "entry_price"]
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["base_currency"] = df["currency_pair"].str[:3]
    df["term_currency"] = df["currency_pair"].str[4:]
    df["side_sign"] = np.where(df["side"] == "BUY", 1, -1)
    return df


@pytest.fixture
def eurgbp():
    """The spec's Verification example: U_EUR 1.0830->1.0850, U_GBP 1.2710->1.2700, N=1,000,000."""
    u_eur0, u_gbp0, u_eur1, u_gbp1 = 1.0830, 1.2710, 1.0850, 1.2700
    return {
        "notional": 1_000_000,
        "entry": u_eur0 / u_gbp0,  # S_0 = U_A,0 / U_B,0
        "u_eur0": u_eur0,
        "u_gbp0": u_gbp0,
        "u_eur1": u_eur1,
        "u_gbp1": u_gbp1,
    }


# --- Triangulation consistency ----------------------------------------------


def test_eurgbp_matches_spec_verification_figure(eurgbp):
    p = position_value(1, eurgbp["notional"], eurgbp["entry"], eurgbp["u_eur1"], eurgbp["u_gbp1"])

    assert round(p) == 2852
    assert p == pytest.approx(2852.085, abs=0.01)


def test_eurgbp_matches_two_leg_decomposition_to_the_cent(eurgbp):
    n, s0 = eurgbp["notional"], eurgbp["entry"]
    eur_leg = n * (eurgbp["u_eur1"] - eurgbp["u_eur0"])
    gbp_leg = -n * s0 * (eurgbp["u_gbp1"] - eurgbp["u_gbp0"])

    p = position_value(1, n, s0, eurgbp["u_eur1"], eurgbp["u_gbp1"])

    assert eur_leg == pytest.approx(2000.00, abs=1e-6)
    assert gbp_leg == pytest.approx(852.085, abs=1e-3)
    assert p == pytest.approx(eur_leg + gbp_leg, abs=1e-6)


def test_eurgbp_value_is_zero_at_entry(eurgbp):
    p0 = position_value(1, eurgbp["notional"], eurgbp["entry"], eurgbp["u_eur0"], eurgbp["u_gbp0"])

    assert p0 == pytest.approx(0.0, abs=1e-9)


def test_eurgbp_through_compute_position_pnl(eurgbp):
    trades = _trades(("X1", "2026-09-01", "EUR/GBP", "BUY", eurgbp["notional"], eurgbp["entry"]))
    quotes = {
        "EUR": _quote(eurgbp["u_eur1"], previous=eurgbp["u_eur0"]),
        "GBP": _quote(eurgbp["u_gbp1"], previous=eurgbp["u_gbp0"]),
    }

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["inception_pnl"] == pytest.approx(2852.085, abs=0.01)
    # P_{t-1} sits exactly at entry rates, so the whole move is today's
    assert row["daily_pnl"] == pytest.approx(row["inception_pnl"], abs=1e-6)
    assert row["current_rate"] == pytest.approx(eurgbp["u_eur1"] / eurgbp["u_gbp1"])


def test_triangulation_identity_holds_algebraically_on_random_inputs():
    rng = np.random.default_rng(42)
    n = 1000
    s = rng.choice([-1, 1], n)
    notional = rng.uniform(1e5, 1e8, n)
    u_a0, u_b0 = rng.uniform(0.005, 2.0, n), rng.uniform(0.005, 2.0, n)
    u_a1, u_b1 = u_a0 * rng.uniform(0.9, 1.1, n), u_b0 * rng.uniform(0.9, 1.1, n)
    s0 = u_a0 / u_b0

    p = position_value(s, notional, s0, u_a1, u_b1)
    two_leg = s * notional * (u_a1 - u_a0) - s * notional * s0 * (u_b1 - u_b0)

    np.testing.assert_allclose(p, two_leg, rtol=1e-9, atol=1e-6)


def test_cross_entry_off_triangulated_rate_gives_small_basis(eurgbp):
    traded_entry = round(eurgbp["entry"], 4)  # real crosses trade at their own quoted rate

    p0 = position_value(1, eurgbp["notional"], traded_entry, eurgbp["u_eur0"], eurgbp["u_gbp0"])

    assert p0 != 0.0
    assert abs(p0) < 0.0001 * eurgbp["notional"]


# --- Inception P&L at trade date -------------------------------------------


@pytest.mark.parametrize("side_sign", [1, -1])
@pytest.mark.parametrize(
    "entry, u_base, u_term",
    [
        (1.0865, 1.0865, 1.0),  # EUR/USD
        (147.35, 1.0, 1 / 147.35),  # USD/JPY
        (0.8550, 0.8550 * 1.2710, 1.2710),  # EUR/GBP cross
    ],
    ids=["EUR/USD", "USD/JPY", "EUR/GBP"],
)
def test_inception_pnl_zero_when_current_equals_entry(side_sign, entry, u_base, u_term):
    assert position_value(side_sign, 5_000_000, entry, u_base, u_term) == pytest.approx(0.0, abs=1e-6)


# --- Type T vs Type B symmetry ---------------------------------------------


@pytest.mark.parametrize("u_jpy", [1 / 140.0, 1 / 150.0, 1 / 160.0])
def test_usd_jpy_and_jpy_usd_equivalents_match(u_jpy):
    # Long USD 1m vs JPY at 150, quoted both ways
    usd_jpy = position_value(1, 1_000_000, 150.0, 1.0, u_jpy)
    jpy_usd = position_value(-1, 150_000_000, 1 / 150.0, u_jpy, 1.0)

    assert np.sign(usd_jpy) == np.sign(jpy_usd)
    assert usd_jpy == pytest.approx(jpy_usd, abs=1e-6)


@pytest.mark.parametrize("u_eur", [1.05, 1.0865, 1.12])
def test_eur_usd_and_usd_eur_equivalents_match(u_eur):
    # Long EUR 1m vs USD at 1.0865, quoted both ways
    eur_usd = position_value(1, 1_000_000, 1.0865, u_eur, 1.0)
    usd_eur = position_value(-1, 1_086_500, 1 / 1.0865, 1.0, u_eur)

    assert np.sign(eur_usd) == np.sign(usd_eur)
    assert eur_usd == pytest.approx(usd_eur, abs=1e-6)


def test_usd_base_pnl_sign_direction():
    # BUY USD/JPY profits when USD/JPY rises, i.e. when U_JPY falls
    assert position_value(1, 1_000_000, 150.0, 1.0, 1 / 155.0) > 0
    assert position_value(1, 1_000_000, 150.0, 1.0, 1 / 145.0) < 0


# --- compute_position_pnl: daily vs inception and edge cases -----------------


def test_daily_pnl_is_change_in_position_value():
    trades = _trades(("T1", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.12, previous=1.115), "USD": USD}

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["inception_pnl"] == pytest.approx(1_000_000 * (1.12 - 1.10))
    assert row["daily_pnl"] == pytest.approx(1_000_000 * (1.12 - 1.115))
    assert not row["new_position"]
    assert row["mark_as_of"] == pd.Timestamp(MARK)  # USD leg's date never drives the mark


def test_same_day_trade_shows_identical_inception_and_daily_off_live_quote():
    # Trade dated today, live quote for today's (still-open) session
    trades = _trades(("T1", "2026-10-09", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.12, previous=1.05), "USD": USD}

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["inception_pnl"] == pytest.approx(20_000)  # P_t - 0, not blanked
    assert row["daily_pnl"] == row["inception_pnl"]  # entry stands in for P_{t-1}; yesterday's close ignored
    assert row["new_position"]  # informational badge only


def test_same_day_trade_is_included_in_portfolio_daily_total():
    trades = _trades(
        ("T1", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10),
        ("T2", "2026-10-09", "EUR/USD", "BUY", 1_000_000, 1.11),
    )
    quotes = {"EUR": _quote(1.12, previous=1.115), "USD": USD}

    pnl = compute_position_pnl(trades, quotes)
    totals = portfolio_pnl(pnl)

    assert pnl["new_position"].tolist() == [False, True]
    assert totals.daily_pnl == pytest.approx(5_000 + 10_000)
    assert totals.inception_pnl == pytest.approx(20_000 + 10_000)


def test_same_day_trade_without_any_previous_close_still_marks_off_entry():
    trades = _trades(("T1", "2026-10-09", "EUR/USD", "BUY", 1_000_000, 1.10))
    eur = Quote(1.12, MARK, None, None, "yfinance", indicative=True, stale=False)

    row = compute_position_pnl(trades, {"EUR": eur, "USD": USD}).iloc[0]

    assert row["new_position"]
    assert row["daily_pnl"] == row["inception_pnl"] == pytest.approx(20_000)


def test_trade_dated_after_its_latest_quote_still_marks_off_that_quote():
    # e.g. a cached close from before the trade: computed, badged, and (via the quote) stale
    trades = _trades(("T1", "2026-10-05", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {
        "EUR": _quote(1.12, previous=1.11, as_of=date(2026, 10, 2), previous_as_of=date(2026, 10, 1), stale=True),
        "USD": USD,
    }

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["new_position"] and row["stale"]
    assert row["daily_pnl"] == row["inception_pnl"] == pytest.approx(20_000)


def test_day_two_daily_diverges_from_inception_once_a_prior_close_exists():
    trades = _trades(("T1", "2026-10-08", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.12, previous=1.115), "USD": USD}  # previous close dated 10-08

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert not row["new_position"]
    assert row["inception_pnl"] == pytest.approx(20_000)
    assert row["daily_pnl"] == pytest.approx(5_000)


def test_missing_leg_quote_flags_no_market_data():
    trades = _trades(("T1", "2026-09-01", "USD/SEK", "BUY", 1_000_000, 10.5))

    row = compute_position_pnl(trades, {"USD": USD}).iloc[0]

    assert row["no_market_data"]
    assert np.isnan(row["inception_pnl"]) and np.isnan(row["daily_pnl"])
    assert not row["new_position"]


def test_existing_position_missing_previous_close_has_blank_daily_not_zero():
    trades = _trades(("T1", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10))
    eur = Quote(1.12, MARK, None, None, "yfinance", indicative=True, stale=False)

    row = compute_position_pnl(trades, {"EUR": eur, "USD": USD}).iloc[0]

    assert row["inception_pnl"] == pytest.approx(20_000)
    assert np.isnan(row["daily_pnl"])
    assert row["missing_previous_close"] and not row["new_position"]
    assert row["data_quality"] == "previous close missing (data gap), daily P&L blank"


def test_opposite_signed_trades_on_same_pair_net():
    trades = _trades(
        ("T1", "2026-09-01", "EUR/USD", "BUY", 5_000_000, 1.10),
        ("T2", "2026-09-02", "EUR/USD", "SELL", 5_000_000, 1.10),
    )
    quotes = {"EUR": _quote(1.13, previous=1.12), "USD": USD}

    totals = portfolio_pnl(compute_position_pnl(trades, quotes))

    assert totals.inception_pnl == pytest.approx(0.0, abs=1e-6)
    assert totals.daily_pnl == pytest.approx(0.0, abs=1e-6)


def test_labels_propagate_from_either_leg():
    trades = _trades(("T1", "2026-09-01", "EUR/GBP", "BUY", 1_000_000, 0.855))
    quotes = {
        "EUR": _quote(1.08, indicative=True, stale=False),
        "GBP": _quote(1.27, indicative=False, stale=True),
    }

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["indicative"] and row["stale"]


# --- mark date & leg staleness -------------------------------------------------


def test_cross_mark_is_earlier_leg_and_banner_names_the_lagging_leg():
    trades = _trades(("T1", "2026-09-01", "EUR/GBP", "BUY", 1_000_000, 0.855))
    quotes = {
        "EUR": _quote(1.08, as_of=date(2026, 10, 9), previous_as_of=date(2026, 10, 8)),
        "GBP": _quote(1.27, as_of=date(2026, 10, 2), previous_as_of=date(2026, 10, 1), stale=True),
    }

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["mark_as_of"] == pd.Timestamp("2026-10-02")
    assert row["previous_as_of"] == pd.Timestamp("2026-10-01")
    assert row["data_quality"] == (
        "GBP leg (as of 2026-10-02, stale) is 5 FX business days behind the EUR leg (as of 2026-10-09)"
    )


def test_leg_lag_counts_fx_business_days_across_a_closure():
    trades = _trades(("T1", "2026-09-01", "EUR/GBP", "BUY", 1_000_000, 0.855))
    quotes = {
        "EUR": _quote(1.08, as_of=date(2026, 12, 28), previous_as_of=date(2026, 12, 24)),
        "GBP": _quote(1.27, as_of=date(2026, 12, 24), previous_as_of=date(2026, 12, 23)),
    }

    row = compute_position_pnl(trades, quotes).iloc[0]

    # Dec 25 is an FX closure and the 26th/27th a weekend: one trading day apart
    assert "GBP leg (as of 2026-12-24) is 1 FX business day behind the EUR leg" in row["data_quality"]


def test_usd_leg_is_skipped_when_choosing_the_mark_date():
    # Even a USD quote dated *earlier* than the other leg must not drive the mark
    old_usd = Quote(1.0, date(2026, 10, 1), 1.0, date(2026, 9, 30), "USD (unit)", indicative=False, stale=False)
    trades = _trades(
        ("T1", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10),
        ("T2", "2026-09-01", "USD/JPY", "BUY", 1_000_000, 147.0),
    )
    quotes = {"EUR": _quote(1.12), "JPY": _quote(1 / 148.0), "USD": old_usd}

    pnl = compute_position_pnl(trades, quotes)

    assert (pnl["mark_as_of"] == pd.Timestamp(MARK)).all()
    assert (pnl["previous_as_of"] == pd.Timestamp(PREV)).all()
    assert (pnl["data_quality"] == "").all()


def test_same_dated_cross_legs_have_no_lag_banner():
    trades = _trades(("T1", "2026-09-01", "EUR/GBP", "BUY", 1_000_000, 0.855))
    quotes = {"EUR": _quote(1.08), "GBP": _quote(1.27)}

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["mark_as_of"] == pd.Timestamp(MARK)
    assert row["data_quality"] == ""


def test_stale_leg_is_named_not_blanket_tagged():
    trades = _trades(("T1", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10))
    quotes = {"EUR": _quote(1.12, as_of=date(2026, 10, 2), stale=True), "USD": USD}

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["stale"]
    assert row["data_quality"] == "EUR leg quote as of 2026-10-02 is stale"


def test_unquoted_leg_named_in_banner():
    trades = _trades(("T1", "2026-09-01", "EUR/SEK", "BUY", 1_000_000, 11.5))
    quotes = {"EUR": _quote(1.08)}

    row = compute_position_pnl(trades, quotes).iloc[0]

    assert row["no_market_data"] and not row["new_position"]
    assert row["data_quality"] == "no market data for SEK leg"


def test_portfolio_totals_exclude_and_count_missing():
    trades = _trades(
        ("T1", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10),
        ("T2", "2026-09-01", "USD/SEK", "BUY", 1_000_000, 10.5),
    )
    quotes = {"EUR": _quote(1.12, previous=1.11), "USD": USD}

    totals = portfolio_pnl(compute_position_pnl(trades, quotes))

    assert totals.inception_pnl == pytest.approx(20_000)
    assert totals.daily_pnl == pytest.approx(10_000)
    assert totals.no_market_data == ("T2",)
    assert totals.inception_footnote == "Excludes 1 of 2 positions with blank inception P&L. no market data: T2."
    assert totals.daily_footnote == "Excludes 1 of 2 positions with blank daily P&L. no market data: T2."


# --- blank daily P&L on existing positions: portfolio total & footnote ---------


def _gap_portfolio():
    trades = _trades(
        ("T1", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10),  # fine
        ("T2", "2026-09-15", "USD/CNH", "BUY", 1_000_000, 7.10),  # existing, CNH has no previous close
        ("T3", "2026-10-09", "USD/CNH", "SELL", 500_000, 6.69),  # same-day trade on the same gappy quote
    )
    cnh = Quote(1 / 6.692, MARK, None, None, "yfinance", indicative=True, stale=False)
    return trades, {"EUR": _quote(1.12, previous=1.115), "CNH": cnh, "USD": USD}


def test_blank_daily_is_excluded_from_total_not_nan_and_not_zero():
    trades, quotes = _gap_portfolio()
    pnl = compute_position_pnl(trades, quotes)

    totals = portfolio_pnl(pnl)

    assert pnl["missing_previous_close"].tolist() == [False, True, False]  # T3 is new, not a gap
    expected_daily = pnl.loc[[0, 2], "daily_pnl"].sum()
    assert not np.isnan(totals.daily_pnl)  # one blank position doesn't blank the total
    assert totals.daily_pnl == pytest.approx(expected_daily)
    assert totals.missing_previous_close == ("T2",)
    assert totals.inception_pnl == pytest.approx(pnl["inception_pnl"].sum())  # inception unaffected
    assert totals.inception_footnote is None


def test_daily_footnote_names_count_trades_and_reason():
    trades, quotes = _gap_portfolio()

    totals = portfolio_pnl(compute_position_pnl(trades, quotes))

    assert totals.daily_footnote == (
        "Excludes 1 of 3 positions with blank daily P&L. "
        "existing position missing its previous close (data gap): T2."
    )


def test_daily_footnote_combines_reasons():
    trades = _trades(
        ("T1", "2026-09-15", "USD/CNH", "BUY", 1_000_000, 7.10),
        ("T2", "2026-09-01", "USD/SEK", "BUY", 1_000_000, 10.5),
        ("T3", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10),
    )
    cnh = Quote(1 / 6.692, MARK, None, None, "yfinance", indicative=True, stale=False)
    quotes = {"CNH": cnh, "EUR": _quote(1.12, previous=1.115), "USD": USD}

    totals = portfolio_pnl(compute_position_pnl(trades, quotes))

    assert totals.daily_footnote == (
        "Excludes 2 of 3 positions with blank daily P&L. "
        "existing position missing its previous close (data gap): T1; no market data: T2."
    )
    assert totals.daily_pnl == pytest.approx(5_000)


def test_total_is_blank_only_when_every_position_is_blank():
    trades = _trades(("T1", "2026-09-15", "USD/CNH", "BUY", 1_000_000, 7.10))
    cnh = Quote(1 / 6.692, MARK, None, None, "yfinance", indicative=True, stale=False)

    totals = portfolio_pnl(compute_position_pnl(trades, {"CNH": cnh, "USD": USD}))

    assert np.isnan(totals.daily_pnl)
    assert totals.daily_footnote.startswith("Excludes 1 of 1 position with blank daily P&L.")


def test_no_footnotes_when_nothing_is_excluded():
    trades = _trades(("T1", "2026-09-01", "EUR/USD", "BUY", 1_000_000, 1.10))

    totals = portfolio_pnl(compute_position_pnl(trades, {"EUR": _quote(1.12, previous=1.115), "USD": USD}))

    assert totals.daily_footnote is None and totals.inception_footnote is None


def test_sample_portfolio_at_entry_like_rates_runs_end_to_end():
    trades = load_portfolio(SAMPLE_PORTFOLIO).trades
    quotes = {
        "USD": USD,
        "EUR": _quote(1.09), "GBP": _quote(1.27), "AUD": _quote(0.66), "NZD": _quote(0.59),
        "JPY": _quote(1 / 147.0), "CHF": _quote(1 / 0.80), "CAD": _quote(1 / 1.39),
    }

    pnl = compute_position_pnl(trades, quotes)

    assert len(pnl) == len(trades) == 17
    assert pnl["inception_pnl"].notna().all() and pnl["daily_pnl"].notna().all()
    assert not pnl["no_market_data"].any()
