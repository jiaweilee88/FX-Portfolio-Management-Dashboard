"""FX Portfolio Dashboard, Streamlit entry point.

Every figure on screen comes from src/: portfolio_loader for trades, market_data
for prices, pnl.py for P&L, risk.py for VaR, exposure.py for the currency,
concentration and reconciliation figures. This file only loads, caches, formats
and lays out what those functions return -- it computes nothing financial itself;
every sum, ratio or difference shown (reconciliation totals, diversification
benefit) is itself a named function in src/, not a pandas call inline here.

Each page load recomputes one point-in-time snapshot: the 800-day history
(st.cache_data, 24h) plus current quotes (st.cache_data, 60s), then P&L/VaR off
those quotes, fully refreshed from the sidebar's confidence/horizon/lookback on
every rerun -- nothing VaR-related is cached, so a sidebar change always recomputes
the real thing, never a stale figure. No intraday (sub-daily) series is stored or
tracked; the historical P&L chart is the current, static holdings replayed against
each past day's settled price, not a day-by-day record of P&L as it actually
happened -- there is no day-by-day holdings history to replay, since this
dashboard only ever sees the current trade list.

Page order: portfolio summary, historical P&L, positions by pair, component VaR
breakdown (pair level), currency exposure, trade-level detail, then data-quality
banners last in a collapsed expander -- visible to anyone who opens it, but not
competing with the numbers above for attention.
"""

import math
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.exposure import (
    compute_currency_exposure,
    pair_diversification_benefit,
    pair_standalone_sum,
    pair_summary,
    total_component_var,
    total_pct_of_portfolio_var,
    var_concentration,
)
from src.analytics.pnl import compute_position_pnl, historical_portfolio_pnl, portfolio_pnl
from src.analytics.risk import (
    CONFIDENCE_LEVELS,
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    MAX_LOOKBACK_DAYS,
    MIN_LOOKBACK_DAYS,
    compute_var,
)
from src.data import market_data as md
from src.data.portfolio_loader import load_portfolio
from src.models.schema import PortfolioSchemaError

PORTFOLIO_PATH = Path(__file__).resolve().parent / "data" / "sample_portfolio.csv"
HISTORY_TTL_S = 24 * 60 * 60
CURRENT_PRICE_TTL_S = 60

# Diverging pair (dataviz skill palette.md): red = positive (adds to portfolio VaR /
# net long), blue = negative (diversifies / net short). Flat two-color usage, not a
# ramp, so the categorical adjacency validator doesn't apply (color-formula.md's
# Scope note); both hexes clear >=3:1 against their own mode's chart surface.
_DIVERGING_COLORS = {
    "light": {"positive": "#e34948", "negative": "#2a78d6", "text": "#0b0b0b", "grid": "#e1e0d9", "muted": "#898781"},
    "dark": {"positive": "#e66767", "negative": "#3987e5", "text": "#f0efec", "grid": "#2c2c2a", "muted": "#898781"},
}

# Status colors (dataviz skill palette.md, fixed, never themed -- same hex in both
# modes). P&L sign is a genuine good/bad state, not an arbitrary polarity like the
# VaR/exposure charts above, so it earns the reserved status tokens instead of the
# diverging pair. The equity-curve line uses the categorical slot-1 blue: a single
# series, so plain identity color, not a sign encoding.
_STATUS_COLORS = {"good": "#0ca30c", "critical": "#d03b3b"}
_LINE_COLOR = {"light": "#2a78d6", "dark": "#3987e5"}


def _theme() -> str:
    theme = st.context.theme.type if st.context.theme else None
    return "dark" if theme == "dark" else "light"


@st.cache_data(ttl=HISTORY_TTL_S, show_spinner="Loading 800-day price history...")
def cached_history(currencies: tuple[str, ...], today: date) -> md.HistoryData:
    return md.load_history(currencies, today=today)


@st.cache_data(ttl=CURRENT_PRICE_TTL_S, show_spinner="Fetching current quotes...")
def cached_current_quotes(currencies: tuple[str, ...], today: date) -> tuple[dict, dict]:
    return md.load_current_quotes(currencies, today=today)


# --- formatting helpers (display only -- no figure below is derived here) -----


def _usd(value: float) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"${value:,.2f}"


def _pnl_cell(value: float) -> str:
    """Blank for a genuine data gap (NaN), never 0."""
    return "—" if pd.isna(value) else f"{value:,.2f}"


def _var_cell(value: float) -> str:
    return "excluded" if pd.isna(value) else f"{value:,.2f}"


def _pct_cell(value: float) -> str:
    return "excluded" if pd.isna(value) else f"{value:.2f}%"


def _rate_cell(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.4f}"


def _build_pair_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Primary table: one row per net currency pair (src/analytics/exposure.py's
    pair_summary). Net exposure, daily P&L, inception P&L and component VaR are
    all exact sums of their trade-level figures; standalone VaR is computed
    directly on the pair's net exposure, never summed from the trades' own
    standalone VaR (see pair_summary's docstring for why that would overstate risk)."""
    df = summary.reset_index()
    base_ccy = df["currency_pair"].str.split("/").str[0]
    return pd.DataFrame(
        {
            "Pair": df["currency_pair"],
            "Net Exposure": [f"{v:+,.2f} {c}" for v, c in zip(df["net_notional"], base_ccy)],
            "Inception P&L": df["inception_pnl"].map(_pnl_cell),
            "Daily P&L": df["daily_pnl"].map(_pnl_cell),
            "Standalone VaR": df["standalone_var"].map(_var_cell),
            "Component VaR": df["component_var"].map(_var_cell),
            "% of Portfolio VaR": df["pct_of_portfolio_var"].map(_pct_cell),
        }
    )


def _build_trade_detail_table(pnl_df: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    """Trade-level detail: every number pair_summary's rows netted away, named back
    to the individual trade. Current Price is the literal U_A,t/U_B,t-derived rate
    pnl.py marked this trade's P&L against (pnl_df's own current_rate column), not a
    separately fetched display value. standalone_var/component_var/pct_of_portfolio_var
    come from risk.py's own trade-level output (var_concentration) -- the same numbers
    the pair-level and chart sections are built from, just unaggregated."""
    merged = pnl_df.merge(
        concentration[["standalone_var", "component_var", "pct_of_portfolio_var"]],
        left_on="trade_id",
        right_index=True,
        how="left",
    )
    return pd.DataFrame(
        {
            "Trade ID": merged["trade_id"],
            "Trade Date": merged["trade_date"].dt.strftime("%Y-%m-%d"),
            "Pair": merged["currency_pair"],
            "Side": merged["side"],
            "Notional": merged["notional_base_ccy"].map(lambda v: f"{v:,.0f}"),
            "Entry Price": merged["entry_price"].map(_rate_cell),
            "Current Price (indicative)": merged["current_rate"].map(_rate_cell),
            "Daily P&L": merged["daily_pnl"].map(_pnl_cell),
            "Inception P&L": merged["inception_pnl"].map(_pnl_cell),
            "Standalone VaR": merged["standalone_var"].map(_var_cell),
            "Component VaR": merged["component_var"].map(_var_cell),
            "% of Portfolio VaR": merged["pct_of_portfolio_var"].map(_pct_cell),
        }
    )


def _build_currency_exposure_table(exposure_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Net USD Exposure": exposure_frame["net_exposure"].map(lambda v: f"{v:,.2f}"),
            "% of Gross": exposure_frame["pct_of_gross"].map(lambda v: f"{v:.2f}%"),
        },
        index=exposure_frame.index.rename("Currency"),
    )


def _equity_curve_chart(frame: pd.DataFrame) -> go.Figure:
    """Line chart of pnl.historical_portfolio_pnl's inception_pnl column: the
    current, static holdings marked against each past settled session's price."""
    palette = _DIVERGING_COLORS[_theme()]
    fig = go.Figure(
        go.Scatter(
            x=frame.index,
            y=frame["inception_pnl"],
            mode="lines",
            line=dict(color=_LINE_COLOR[_theme()], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Inception P&L: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=palette["text"],
        margin=dict(l=0, r=10, t=10, b=0),
        height=280,
        xaxis=dict(title=None, gridcolor=palette["grid"]),
        yaxis=dict(
            title="Inception P&L (USD)", gridcolor=palette["grid"], zeroline=True, zerolinecolor=palette["muted"]
        ),
        showlegend=False,
    )
    return fig


def _daily_pnl_bar_chart(frame: pd.DataFrame) -> go.Figure:
    """Bar chart of pnl.historical_portfolio_pnl's daily_pnl column. Green/red
    (status colors): P&L sign is a good/bad state, not a polarity."""
    palette = _DIVERGING_COLORS[_theme()]
    daily = frame["daily_pnl"].dropna()
    colors = [_STATUS_COLORS["good"] if v >= 0 else _STATUS_COLORS["critical"] for v in daily]

    fig = go.Figure(
        go.Bar(
            x=daily.index,
            y=daily,
            marker_color=colors,
            hovertemplate="%{x|%Y-%m-%d}<br>Daily P&L: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=palette["text"],
        margin=dict(l=0, r=10, t=10, b=0),
        height=220,
        xaxis=dict(title=None, gridcolor=palette["grid"]),
        yaxis=dict(
            title="Daily P&L (USD)", gridcolor=palette["grid"], zeroline=True, zerolinecolor=palette["muted"]
        ),
        showlegend=False,
    )
    return fig


def _component_var_chart(summary: pd.DataFrame) -> go.Figure:
    """Horizontal bar of component VaR by currency pair (summary = pair_summary's
    output: index is the pair, so the chart's bars line up one-to-one with the
    Positions table above it -- no separate trade-level breakdown competing for
    the same 'component VaR' label at a different grain)."""
    palette = _DIVERGING_COLORS[_theme()]

    ordered = summary.iloc[::-1]  # largest |contribution| at the top of the bar chart
    labels = list(ordered.index.astype(str))
    values = ordered["component_var"]
    colors = [palette["positive"] if v >= 0 else palette["negative"] for v in values]
    text = [f"${v:,.2f} ({pct:.2f}%)" for v, pct in zip(values, ordered["pct_of_portfolio_var"])]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="outside",
            hovertemplate="%{y}<br>Component VaR: $%{x:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=palette["text"],
        margin=dict(l=0, r=10, t=10, b=0),
        height=max(220, 44 * len(labels)),
        xaxis=dict(title="Component VaR (USD)", gridcolor=palette["grid"], zeroline=True, zerolinecolor=palette["muted"]),
        yaxis=dict(title=None),
        showlegend=False,
    )
    return fig


def _currency_exposure_chart(exposure_frame: pd.DataFrame) -> go.Figure:
    """Diverging horizontal bar of net USD exposure by currency. Above/below a zero
    baseline is exactly the diverging-bar job per the dataviz skill's
    choosing-a-form.md; a pie wedge's angle has no sign, so an earlier pie version
    of this chart could only show magnitude, not long/short direction."""
    palette = _DIVERGING_COLORS[_theme()]

    ordered = exposure_frame.reindex(exposure_frame["net_exposure"].abs().sort_values().index)
    labels = list(ordered.index.astype(str))
    values = ordered["net_exposure"]
    colors = [palette["positive"] if v >= 0 else palette["negative"] for v in values]
    text = [f"${v:,.2f} ({pct:.1f}% of gross)" for v, pct in zip(values, ordered["pct_of_gross"])]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="outside",
            hovertemplate="%{y}<br>Net exposure: $%{x:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=palette["text"],
        margin=dict(l=0, r=10, t=10, b=0),
        height=max(220, 40 * len(labels)),
        xaxis=dict(title="Net USD Exposure", gridcolor=palette["grid"], zeroline=True, zerolinecolor=palette["muted"]),
        yaxis=dict(title=None),
        showlegend=False,
    )
    return fig


# --- page ----------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="FX Portfolio Dashboard", layout="wide")
    st.title("FX Portfolio Dashboard")

    today = date.today()

    st.sidebar.header("Portfolio source")
    uploaded = st.sidebar.file_uploader(
        "Upload a portfolio CSV (optional)",
        type=["csv"],
        help="Required columns: trade_id, trade_date, currency_pair, side, notional_base_ccy, entry_price.",
    )
    try:
        loaded = load_portfolio(uploaded if uploaded is not None else PORTFOLIO_PATH)
    except PortfolioSchemaError as exc:
        st.sidebar.error(f"Could not load the uploaded file ({exc}). Showing the bundled sample portfolio instead.")
        uploaded = None
        loaded = load_portfolio(PORTFOLIO_PATH)
    st.sidebar.caption(f"Source: {uploaded.name if uploaded is not None else 'bundled sample_portfolio.csv'}")

    trades = loaded.trades
    banners: list[str] = [
        f"Rejected portfolio row {r.row_number} ({r.trade_id}): {r.reason}" for r in loaded.report.rejected_rows
    ]

    st.sidebar.header("VaR parameters")
    confidence = st.sidebar.selectbox(
        "Confidence level",
        options=CONFIDENCE_LEVELS,
        index=CONFIDENCE_LEVELS.index(DEFAULT_CONFIDENCE),
        format_func=lambda c: f"{c:.0%}",
    )
    horizon_days = int(
        st.sidebar.number_input("Horizon (days)", min_value=1, max_value=30, value=DEFAULT_HORIZON_DAYS, step=1)
    )
    lookback_days = int(
        st.sidebar.slider(
            "Lookback window (trading days)",
            min_value=MIN_LOOKBACK_DAYS,
            max_value=MAX_LOOKBACK_DAYS,
            value=DEFAULT_LOOKBACK_DAYS,
            step=1,
        )
    )
    st.sidebar.caption(
        "Confidence and horizon rescale every $ VaR figure uniformly, so each "
        "position's % share of portfolio VaR is unchanged by them -- it only moves "
        "with the lookback window, which changes the underlying correlation structure."
    )

    if trades.empty:
        st.warning("No valid trades to analyze -- see the data-quality warnings below.")
        with st.expander(f"Data-quality warnings ({len(banners)})", expanded=True):
            for b in banners:
                st.warning(b)
        return

    currencies = tuple(sorted(set(trades["base_currency"]) | set(trades["term_currency"])))
    history = cached_history(currencies, today)
    quotes, quote_errors = cached_current_quotes(currencies, today)
    market = md.MarketData(history=history, current=quotes, current_errors=quote_errors)

    latest = max((q.as_of for c, q in quotes.items() if c != "USD"), default=None)
    st.caption(f"Prices: yfinance, indicative - latest quote as of {latest}")

    banners += market.warnings()

    # --- analytics: fully recomputed every rerun from the sidebar's current values,
    # nothing VaR-related is wrapped in st.cache_data -- every number below traces
    # to a src/ function call made right here, with this run's actual parameters. --
    pnl_df = compute_position_pnl(trades, quotes)
    totals = portfolio_pnl(pnl_df)
    returns = history.returns()
    var_result = compute_var(
        trades, quotes, returns, confidence=confidence, horizon_days=horizon_days, lookback_days=lookback_days
    )
    concentration = var_concentration(var_result, trades)
    pair_table = pair_summary(trades, quotes, pnl_df, var_result)
    currency_exposure = compute_currency_exposure(trades, quotes)
    historical_frame, historical_excluded = historical_portfolio_pnl(trades, history.levels)

    if totals.daily_footnote:
        banners.append(totals.daily_footnote)
    if var_result.excluded_trade_ids:
        banners.append(f"Excluded from VaR, no market data: {', '.join(var_result.excluded_trade_ids)}")
    if currency_exposure.excluded_trade_ids:
        banners.append(f"Excluded from currency exposure, no market data: {', '.join(currency_exposure.excluded_trade_ids)}")
    if historical_excluded:
        banners.append(f"Excluded from historical P&L, no market data: {', '.join(historical_excluded)}")

    # --- 1. portfolio summary ---------------------------------------------------
    st.subheader("Portfolio summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inception P&L (USD)", _usd(totals.inception_pnl))
    if totals.inception_footnote:
        c1.caption(totals.inception_footnote)
    c2.metric("Daily P&L (USD)", _usd(totals.daily_pnl))
    if totals.daily_footnote:
        c2.caption(totals.daily_footnote)
    c3.metric(f"Portfolio VaR ({confidence:.0%}, {horizon_days}d)", _usd(var_result.portfolio_var))
    c3.caption(f"{lookback_days}-day lookback, z={var_result.z:.3f}")
    # Diversification benefit, computed at PAIR (net) level, not trade level: summing
    # two offsetting trades' own gross standalone VaRs would double-count trivial
    # within-pair netting as if it were genuine cross-currency diversification -- see
    # exposure.pair_diversification_benefit's docstring.
    pair_db = pair_diversification_benefit(pair_table, var_result.portfolio_var)
    c4.metric("Diversification Benefit (USD)", _usd(pair_db))
    c4.caption(f"Sum of pair-level standalone VaRs: {_usd(pair_standalone_sum(pair_table))}")

    # --- 2. historical P&L: the current holdings replayed against past prices --
    st.subheader("Historical P&L")
    if historical_frame.empty:
        st.caption("Not enough price history to build a historical P&L curve for this portfolio.")
    else:
        st.plotly_chart(_equity_curve_chart(historical_frame), width="stretch")
        st.plotly_chart(_daily_pnl_bar_chart(historical_frame), width="stretch")
        st.caption(
            f"Mark-to-market of the current portfolio against historical prices, "
            f"{historical_frame.index.min():%Y-%m-%d} to {historical_frame.index.max():%Y-%m-%d} "
            "(last fully-settled session) -- today's live move is in the Portfolio summary metrics "
            "above, not yet in this curve. Green/red = gain/loss."
        )

    # --- 3. positions by pair ----------------------------------------------------
    st.subheader("Position Information by Currency Pair")
    st.dataframe(_build_pair_table(pair_table), hide_index=True, width="stretch")
    st.caption(
        f"Sum of component VaRs {_usd(total_component_var(pair_table))} reconciles to portfolio VaR "
        f"{_usd(var_result.portfolio_var)}; percentages sum to {total_pct_of_portfolio_var(pair_table):.2f}%."
    )

    # --- 4. component VaR breakdown, at pair level (matches the table above) ----
    st.subheader("Component VaR breakdown")
    st.plotly_chart(_component_var_chart(pair_table), width="stretch")
    st.caption("Red = adds to portfolio VaR. Blue = diversifies it.")

    # --- 5. currency exposure ----------------------------------------------------
    st.subheader("Currency exposure")
    exposure_frame = currency_exposure.as_frame()
    col_table, col_chart = st.columns([3, 2])
    col_table.dataframe(_build_currency_exposure_table(exposure_frame), width="stretch")
    col_chart.plotly_chart(_currency_exposure_chart(exposure_frame), width="stretch")
    col_chart.caption("Red = net long. Blue = net short.")

    # --- 6. trade-level detail ----------------------------------------------------
    st.subheader("Trade-level detail")
    st.dataframe(_build_trade_detail_table(pnl_df, concentration), hide_index=True, width="stretch")
    st.caption(
        f"Sum of component VaRs {_usd(total_component_var(concentration))} reconciles to portfolio VaR "
        f"{_usd(var_result.portfolio_var)}; percentages sum to {total_pct_of_portfolio_var(concentration):.2f}%."
    )

    # --- data-quality banners: visible, but collapsed, at the very bottom -------
    with st.expander(f"Data-quality warnings ({len(banners)})", expanded=False):
        if banners:
            for b in banners:
                st.warning(b)
        else:
            st.caption("No data-quality issues flagged for this snapshot.")


main()
