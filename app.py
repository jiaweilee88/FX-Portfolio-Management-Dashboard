"""FX Portfolio Dashboard, Streamlit entry point.

Each page load recomputes one point-in-time snapshot: the 800-day history
(st.cache_data, 24h) plus current quotes (st.cache_data, 60s), then P&L off
those quotes. No intraday P&L series is stored, tracked or charted.
"""

import math
from datetime import date
from pathlib import Path

import streamlit as st

from src.analytics.pnl import compute_position_pnl, portfolio_pnl
from src.data import market_data as md
from src.data.portfolio_loader import load_portfolio

PORTFOLIO_PATH = Path(__file__).resolve().parent / "data" / "sample_portfolio.csv"
HISTORY_TTL_S = 24 * 60 * 60
CURRENT_PRICE_TTL_S = 60


@st.cache_data(ttl=HISTORY_TTL_S, show_spinner="Loading 800-day price history...")
def cached_history(currencies: tuple[str, ...], today: date) -> md.HistoryData:
    return md.load_history(currencies, today=today)


@st.cache_data(ttl=CURRENT_PRICE_TTL_S, show_spinner="Fetching current quotes...")
def cached_current_quotes(currencies: tuple[str, ...], today: date) -> tuple[dict, dict]:
    return md.load_current_quotes(currencies, today=today)


def main() -> None:
    st.set_page_config(page_title="FX Portfolio Dashboard", layout="wide")
    st.title("FX Portfolio Dashboard")

    today = date.today()
    loaded = load_portfolio(PORTFOLIO_PATH)
    trades = loaded.trades
    for rejected in loaded.report.rejected_rows:
        st.warning(f"Rejected portfolio row {rejected.row_number} ({rejected.trade_id}): {rejected.reason}")

    currencies = tuple(sorted(set(trades["base_currency"]) | set(trades["term_currency"])))
    history = cached_history(currencies, today)
    quotes, quote_errors = cached_current_quotes(currencies, today)
    market = md.MarketData(history=history, current=quotes, current_errors=quote_errors)

    latest = max((q.as_of for c, q in quotes.items() if c != "USD"), default=None)
    st.caption(f"Prices: yfinance, {md.INDICATIVE_LABEL} · latest quote as of {latest}")

    warnings = market.warnings()
    if warnings:
        with st.expander(f"Data-quality warnings ({len(warnings)})"):
            for w in warnings:
                st.write(f"- {w}")

    pnl = compute_position_pnl(trades, quotes)
    totals = portfolio_pnl(pnl)
    col_inception, col_daily = st.columns(2)
    col_inception.metric("Inception P&L (USD)", _usd(totals.inception_pnl))
    if totals.inception_footnote:
        col_inception.caption(totals.inception_footnote)
    col_daily.metric("Daily P&L (USD)", _usd(totals.daily_pnl))
    if totals.daily_footnote:
        col_daily.caption(totals.daily_footnote)

    st.dataframe(pnl, hide_index=True, width="stretch")


def _usd(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:,.0f}"


main()
