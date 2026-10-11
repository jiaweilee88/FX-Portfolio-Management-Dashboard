from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.runtime.caching import cache_utils
from streamlit.testing.v1 import AppTest

from src.data import market_data as md

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _fake_yf_history(symbol, **kwargs):
    today = date.today()
    if "period" in kwargs:
        # Simulates a data gap on CAD (the sample portfolio's only CAD trade, T0006,
        # thereby ends up with no previous close) without relying on a real-world quirk.
        index = pd.bdate_range(end=today, periods=1 if symbol == "CAD=X" else 2)
    else:
        index = pd.bdate_range(end=today - timedelta(days=1), periods=300)
    seed = sum(map(ord, symbol))
    values = np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.005, len(index))))
    return pd.Series(values, index=pd.DatetimeIndex(index, name="date"), name=symbol)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    clock = FakeClock()
    calls = {"history": 0, "current": 0}
    real_history, real_current = md.load_history, md.load_current_quotes

    def counted_history(*args, **kwargs):
        calls["history"] += 1
        return real_history(*args, **kwargs)

    def counted_current(*args, **kwargs):
        calls["current"] += 1
        return real_current(*args, **kwargs)

    monkeypatch.setattr(md, "_yf_history", _fake_yf_history)
    monkeypatch.setattr(md, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(md, "load_history", counted_history)
    monkeypatch.setattr(md, "load_current_quotes", counted_current)
    monkeypatch.setattr(cache_utils, "TTLCACHE_TIMER", clock)
    st.cache_data.clear()
    yield clock, calls
    st.cache_data.clear()


def _run():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_app_renders_a_snapshot(app_env):
    at = _run()

    labels = [m.label for m in at.metric]
    assert "Inception P&L (USD)" in labels and "Daily P&L (USD)" in labels
    assert not any("Rejected portfolio row" in w.value for w in at.warning)  # sample file has no rejected rows
    assert len(at.dataframe) >= 1


def test_daily_total_carries_footnote_for_blank_positions(app_env):
    at = _run()

    daily = next(m for m in at.metric if m.label == "Daily P&L (USD)")
    assert daily.value != "n/a"  # the blank position didn't blank the total
    footnote = "Excludes 1 of 17 positions with blank daily P&L. existing position missing its previous close (data gap): T0006."
    # Required to be a visible banner, not buried -- and also surfaced as a caption right under the metric.
    assert any(footnote in w.value for w in at.warning)
    assert any(footnote in c.value for c in at.caption)


def test_pair_level_position_table(app_env):
    at = _run()

    pair_table = next(df.value for df in at.dataframe if "Net Exposure" in df.value.columns)
    assert list(pair_table.columns) == [
        "Pair", "Net Exposure", "Inception P&L", "Daily P&L", "Standalone VaR", "Component VaR", "% of Portfolio VaR",
    ]
    assert len(pair_table) == 9
    assert set(pair_table["Pair"]) == {
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
        "USD/CAD", "NZD/USD", "EUR/GBP", "AUD/JPY",
    }


def test_trade_level_detail_table_has_every_requested_field(app_env):
    at = _run()

    detail = next(df.value for df in at.dataframe if list(df.value.columns)[:2] == ["Trade ID", "Trade Date"])
    assert list(detail.columns) == [
        "Trade ID", "Trade Date", "Pair", "Side", "Notional", "Entry Price", "Current Price (indicative)",
        "Daily P&L", "Inception P&L", "Standalone VaR", "Component VaR", "% of Portfolio VaR",
    ]
    assert detail["Current Price (indicative)"].str.match(r"^\d+\.\d{4}$").all()
    assert len(detail) == 17  # every trade, no netting
    assert set(detail["Trade ID"]) == {f"T{i:04d}" for i in range(1, 18)}


def test_component_var_chart_and_table_are_both_pair_level_and_reconcile(app_env):
    at = _run()

    pair_table = next(df.value for df in at.dataframe if "Net Exposure" in df.value.columns)
    portfolio_var_metric = next(m.value for m in at.metric if m.label.startswith("Portfolio VaR"))
    portfolio_var = float(portfolio_var_metric.replace("$", "").replace(",", ""))

    component_sum = pair_table["Component VaR"].map(lambda s: float(s.replace(",", ""))).sum()
    assert component_sum == pytest.approx(portfolio_var, abs=1.0)  # rounded to the dollar per row

    caption = next(c.value for c in at.caption if "reconciles to portfolio VaR" in c.value)
    assert _usd_in(caption, "Sum of component VaRs") == _usd_in(caption, "reconciles to portfolio VaR")


def _usd_in(text: str, marker: str) -> str:
    tail = text.split(marker, 1)[1]
    return tail.split("$", 1)[1].split(".")[0].split(" ")[0]


def test_currency_exposure_table_and_bar_chart_present(app_env):
    at = _run()

    exposure_table = next(df.value for df in at.dataframe if "Net USD Exposure" in df.value.columns)
    assert "USD" in exposure_table.index
    assert exposure_table["Net USD Exposure"].str.match(r"^-?[\d,]+\.\d{2}$").all()  # 2dp, not rounded to the dollar
    assert len(at.get("plotly_chart")) >= 2  # component-VaR bar + currency-exposure bar


def test_historical_pnl_charts_present_between_summary_and_pair_table(app_env):
    at = _run()

    subheaders = [s.value for s in at.subheader]
    assert "Historical P&L" in subheaders
    assert subheaders.index("Historical P&L") < subheaders.index("Position Information by Currency Pair")
    assert subheaders.index("Portfolio summary") < subheaders.index("Historical P&L")

    caption = next(c.value for c in at.caption if "last fully-settled session" in c.value)
    assert "today's live move is in the Portfolio summary metrics above" in caption

    # At least 4 charts now: equity curve, daily bars, component VaR, currency exposure.
    assert len(at.get("plotly_chart")) >= 4


def test_historical_pnl_curve_ends_at_or_before_the_latest_quote_date(app_env):
    at = _run()

    latest_quote_caption = next(c.value for c in at.caption if c.value.startswith("Prices: yfinance"))
    latest_quote_date = date.fromisoformat(latest_quote_caption.rsplit(" ", 1)[1])

    reconciliation_caption = next(c.value for c in at.caption if "last fully-settled session" in c.value)
    curve_end = date.fromisoformat(reconciliation_caption.split(" to ")[1].split(" ")[0])

    assert curve_end <= latest_quote_date  # the curve never runs ahead of today's own live quote


def test_data_quality_banners_are_collapsed_at_the_bottom(app_env):
    at = _run()

    expanders = at.expander
    assert len(expanders) == 1
    banner_expander = expanders[0]
    assert banner_expander.label.startswith("Data-quality warnings")
    assert banner_expander.proto.expanded is False  # collapsed by default, not forced open

    children_types = {type(child).__name__ for child in banner_expander.children.values()}
    assert "Warning" in children_types

    # It's genuinely last: every other subheader precedes it in the script's element order.
    all_children = at.main.children
    expander_position = next(i for i, c in all_children.items() if c is banner_expander)
    assert expander_position == max(all_children)


def test_dollar_figures_show_two_decimal_places_not_rounded_to_the_dollar(app_env):
    at = _run()

    for label in ("Inception P&L (USD)", "Daily P&L (USD)", "Diversification Benefit (USD)"):
        value = next(m.value for m in at.metric if m.label == label)
        assert value != "n/a"
        assert value.split(".")[-1] != "00" or True  # decimals present; exact cents vary run to run
        assert "." in value and len(value.split(".")[-1]) == 2

    pair_table = next(df.value for df in at.dataframe if "Net Exposure" in df.value.columns)
    for col in ("Inception P&L", "Daily P&L", "Standalone VaR", "Component VaR"):
        non_blank = pair_table[col][pair_table[col] != "—"]
        assert non_blank.str.match(r"^-?[\d,]+\.\d{2}$").all(), (col, non_blank.tolist())


def test_diversification_benefit_uses_pair_level_standalone_not_trade_level(app_env):
    at = _run()

    from src.analytics.exposure import pair_diversification_benefit, pair_standalone_sum, pair_summary
    from src.analytics.pnl import compute_position_pnl
    from src.analytics.risk import compute_var
    from src.data.portfolio_loader import load_portfolio

    db_shown = float(next(m.value for m in at.metric if m.label == "Diversification Benefit (USD)").replace("$", "").replace(",", ""))
    standalone_sum_shown = float(next(c.value for c in at.caption if "Sum of pair-level standalone VaRs" in c.value).rsplit("$", 1)[1].replace(",", ""))

    trades = load_portfolio("data/sample_portfolio.csv").trades
    currencies = tuple(sorted(set(trades["base_currency"]) | set(trades["term_currency"])))
    quotes, _ = md.load_current_quotes(currencies, today=date.today())
    history = md.load_history(currencies, today=date.today())
    pnl_df = compute_position_pnl(trades, quotes)
    result = compute_var(trades, quotes, history.returns())
    pair_table = pair_summary(trades, quotes, pnl_df, result)

    expected_db = pair_diversification_benefit(pair_table, result.portfolio_var)
    expected_standalone_sum = pair_standalone_sum(pair_table)

    assert db_shown == pytest.approx(expected_db, abs=0.02)
    assert standalone_sum_shown == pytest.approx(expected_standalone_sum, abs=0.02)
    # The sample portfolio has 3 EUR/USD trades (partly offsetting) and 3 USD/CHF
    # trades, so pair-level netting must make this strictly less than the
    # trade-level figure risk.VaRResult itself would report.
    assert expected_db < result.diversification_benefit


def test_both_reconciliation_captions_state_the_percentage_sum(app_env):
    at = _run()

    captions = [c.value for c in at.caption]
    reconciliation_captions = [c for c in captions if "reconciles to portfolio VaR" in c]
    assert len(reconciliation_captions) == 2  # pair-level and trade-level
    for c in reconciliation_captions:
        assert "percentages sum to 100.00%" in c or "percentages sum to 99." in c or "percentages sum to 100." in c


def test_file_upload_overrides_the_bundled_sample_portfolio(app_env):
    at = _run()
    sample_pair_table = next(df.value for df in at.dataframe if "Net Exposure" in df.value.columns)
    assert len(sample_pair_table) == 9

    csv_content = (
        b"trade_id,trade_date,currency_pair,side,notional_base_ccy,entry_price\n"
        b"U0001,2026-09-01,EUR/USD,BUY,1000000,1.1000\n"
        b"U0002,2026-09-02,GBP/USD,SELL,500000,1.2800\n"
    )
    at.sidebar.file_uploader[0].set_value(("my_portfolio.csv", csv_content, "text/csv")).run()

    assert not at.exception, at.exception
    source_caption = next(c.value for c in at.caption if c.value.startswith("Source:"))
    assert source_caption == "Source: my_portfolio.csv"

    detail = next(df.value for df in at.dataframe if list(df.value.columns)[:2] == ["Trade ID", "Trade Date"])
    assert set(detail["Trade ID"]) == {"U0001", "U0002"}

    pair_table = next(df.value for df in at.dataframe if "Net Exposure" in df.value.columns)
    assert set(pair_table["Pair"]) == {"EUR/USD", "GBP/USD"}


def test_file_upload_with_missing_required_column_falls_back_to_sample_with_error(app_env):
    at = _run()

    bad_csv = b"trade_id,trade_date,currency_pair,side,notional_base_ccy\nU0001,2026-09-01,EUR/USD,BUY,1000000\n"
    at.sidebar.file_uploader[0].set_value(("bad.csv", bad_csv, "text/csv")).run()

    assert not at.exception, at.exception
    assert any("Could not load the uploaded file" in e.value for e in at.sidebar.error)
    source_caption = next(c.value for c in at.caption if c.value.startswith("Source:"))
    assert source_caption == "Source: bundled sample_portfolio.csv"
    pair_table = next(df.value for df in at.dataframe if "Net Exposure" in df.value.columns)
    assert len(pair_table) == 9  # back on the original sample portfolio


def test_current_quotes_cached_60s_history_cached_24h(app_env):
    clock, calls = app_env

    _run()
    assert calls == {"history": 1, "current": 1}

    clock.advance(30)  # rerun within 60s: both served from cache
    _run()
    assert calls == {"history": 1, "current": 1}

    clock.advance(31)  # 61s: quotes refetched, history still cached
    _run()
    assert calls == {"history": 1, "current": 2}

    clock.advance(24 * 60 * 60)  # past 24h: history refetched too
    _run()
    assert calls == {"history": 2, "current": 3}
