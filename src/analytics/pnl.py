"""USD P&L per the spec's 'P&L (one formula, every pair)' section.

P_{i,t} = s_i N_i (U_{A,t} - S_{i,0} U_{B,t}),  PnL_inception = P_{i,t},  PnL_daily = P_{i,t} - P_{i,t-1}

U_C comes from market_data's Quote objects (yfinance, parquet-cache fallback),
with USD present as a unit quote, so majors, USD/XXX pairs and crosses all go
through position_value unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd

from src.data.market_data import FX_BUSINESS_DAY, Quote

MISSING_PREVIOUS_MESSAGE = "previous close missing (data gap), daily P&L blank"


def position_value(side_sign, notional, entry_price, u_base, u_term):
    """P_{i,t}: USD mark-to-market of a spot position on A/B. Elementwise on arrays/Series."""
    return side_sign * notional * (u_base - entry_price * u_term)


def compute_position_pnl(trades: pd.DataFrame, quotes: Mapping[str, Quote]) -> pd.DataFrame:
    """Per-trade inception and daily USD P&L.

    - A trade with no prior close since execution is on its first mark: P_{t-1} = 0
      (entry_price stands in for the missing prior close), so its daily P&L equals
      its inception P&L. It is computed off the live quote, session open or not,
      included in every total, and flagged new_position as an informational badge only.
    - A trade missing a quote for either leg has NaN P&L and is flagged no_market_data.
    - An existing position (not on its first mark) whose quote lacks a previous close
      has a blank (NaN) daily P&L, never 0, and is flagged missing_previous_close.
    - mark_as_of is the earlier of the non-USD legs' quote dates (a position is only
      as current as its stalest input; USD is always current). data_quality names
      any leg that lags the other, by how many FX business days, and any stale leg.
    """
    base, term = trades["base_currency"], trades["term_currency"]
    trade_date = pd.to_datetime(trades["trade_date"])

    u_base = _leg(base, quotes, "value")
    u_term = _leg(term, quotes, "value")
    u_base_prev = _leg(base, quotes, "previous_value")
    u_term_prev = _leg(term, quotes, "previous_value")
    mark_as_of = _earliest(_leg_date(base, quotes, "as_of"), _leg_date(term, quotes, "as_of"))
    previous_as_of = _earliest(
        _leg_date(base, quotes, "previous_as_of"), _leg_date(term, quotes, "previous_as_of")
    )
    no_market_data = u_base.isna() | u_term.isna()

    new_position = ~no_market_data & (
        (previous_as_of < trade_date) | (previous_as_of.isna() & (trade_date >= mark_as_of))
    )

    s, n, s0 = trades["side_sign"], trades["notional_base_ccy"], trades["entry_price"]
    p_t = position_value(s, n, s0, u_base, u_term)
    p_prev = position_value(s, n, s0, u_base_prev, u_term_prev).mask(new_position, 0.0)
    missing_previous_close = ~no_market_data & p_prev.isna()

    out = trades[
        ["trade_id", "trade_date", "currency_pair", "side", "notional_base_ccy", "entry_price",
         "base_currency", "term_currency"]
    ].copy()
    out["current_rate"] = u_base / u_term
    out["u_base"] = u_base  # U_{A,t}: the literal base-currency quote this position's P&L is marked against
    out["u_term"] = u_term  # U_{B,t}
    out["u_base_as_of"] = _leg_date(base, quotes, "as_of")
    out["u_term_as_of"] = _leg_date(term, quotes, "as_of")
    out["mark_as_of"] = mark_as_of
    out["previous_as_of"] = previous_as_of
    out["inception_pnl"] = p_t
    out["daily_pnl"] = p_t - p_prev
    out["new_position"] = new_position
    out["no_market_data"] = no_market_data
    out["missing_previous_close"] = missing_previous_close
    out["indicative"] = _leg_flag(base, quotes, "indicative") | _leg_flag(term, quotes, "indicative")
    out["stale"] = _leg_flag(base, quotes, "stale") | _leg_flag(term, quotes, "stale")
    out["data_quality"] = [
        "; ".join(_leg_messages(b, t, quotes) + ([MISSING_PREVIOUS_MESSAGE] if gap else []))
        for b, t, gap in zip(base, term, missing_previous_close)
    ]
    return out


@dataclass(frozen=True)
class PortfolioPnL:
    """Portfolio totals. Positions with a blank figure are excluded from that sum
    (neither propagated as NaN nor treated as 0) and named in its footnote."""

    n_positions: int
    inception_pnl: float  # NaN only if every position is blank
    daily_pnl: float
    no_market_data: tuple[str, ...]  # trade_ids: blank inception and daily
    missing_previous_close: tuple[str, ...]  # trade_ids: blank daily only

    @property
    def inception_footnote(self) -> Optional[str]:
        return _footnote(self.n_positions, "inception", [("no market data", self.no_market_data)])

    @property
    def daily_footnote(self) -> Optional[str]:
        return _footnote(
            self.n_positions,
            "daily",
            [
                ("existing position missing its previous close (data gap)", self.missing_previous_close),
                ("no market data", self.no_market_data),
            ],
        )


def portfolio_pnl(position_pnl: pd.DataFrame) -> PortfolioPnL:
    ids = position_pnl["trade_id"]
    return PortfolioPnL(
        n_positions=len(position_pnl),
        inception_pnl=float(position_pnl["inception_pnl"].sum(min_count=1)),
        daily_pnl=float(position_pnl["daily_pnl"].sum(min_count=1)),
        no_market_data=tuple(ids[position_pnl["no_market_data"]]),
        missing_previous_close=tuple(ids[position_pnl["missing_previous_close"]]),
    )


def _footnote(n_positions: int, metric: str, groups: list[tuple[str, tuple[str, ...]]]) -> Optional[str]:
    parts = [f"{reason}: {', '.join(trade_ids)}" for reason, trade_ids in groups if trade_ids]
    if not parts:
        return None
    n_excluded = sum(len(trade_ids) for _, trade_ids in groups)
    noun = "position" if n_positions == 1 else "positions"
    return (
        f"Excludes {n_excluded} of {n_positions} {noun} with blank {metric} P&L. "
        + "; ".join(parts) + "."
    )


def historical_portfolio_pnl(
    trades: pd.DataFrame, levels: pd.DataFrame
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Portfolio inception and daily USD P&L at every date in `levels`
    (market_data.HistoryData.levels: U_C by FX business date, USD = 1.0), using the
    same position_value formula as compute_position_pnl -- an equity curve, not a
    new model. A trade contributes exactly 0 before its own trade_date (P_{i,0} = 0
    by construction, same as the point-in-time formula); it is excluded entirely,
    not zeroed, on every date if either of its own legs has no market data at all.

    This is a past-looking mark-to-market of the portfolio's current (static)
    holdings against historical prices, not a record of P&L as it was actually
    reported day by day -- there is no day-by-day holdings history to replay, since
    this dashboard only ever sees the current trade list. The curve stops at the
    last fully-settled session in `levels`; it does not include today's still-live
    quote (compute_position_pnl/portfolio_pnl do, for the headline metrics).

    Returns (frame, excluded_trade_ids), frame indexed by date with columns
    inception_pnl and daily_pnl (daily_pnl's first row is NaN: there is no prior
    date in the window to diff against, which is a window boundary, not a gap)."""
    marks: dict[str, pd.Series] = {}
    excluded: list[str] = []

    for _, t in trades.iterrows():
        base, term = t["base_currency"], t["term_currency"]
        if base not in levels.columns or term not in levels.columns:
            excluded.append(t["trade_id"])
            continue
        mark = position_value(t["side_sign"], t["notional_base_ccy"], t["entry_price"], levels[base], levels[term])
        marks[t["trade_id"]] = mark.mask(mark.index < pd.Timestamp(t["trade_date"]), 0.0)

    if not marks:
        return pd.DataFrame(columns=["inception_pnl", "daily_pnl"]), tuple(excluded)

    inception = pd.DataFrame(marks).sum(axis=1, min_count=1)
    frame = pd.DataFrame({"inception_pnl": inception, "daily_pnl": inception.diff()})
    earliest_trade = pd.Timestamp(trades["trade_date"].min())
    return frame.loc[frame.index >= earliest_trade], tuple(excluded)


def _leg(currencies: pd.Series, quotes: Mapping[str, Quote], attr: str) -> pd.Series:
    return currencies.map(lambda c: getattr(quotes[c], attr) if c in quotes else None).astype(float)


def _leg_date(currencies: pd.Series, quotes: Mapping[str, Quote], attr: str) -> pd.Series:
    """A leg's quote date; NaT for USD (U_USD = 1 is always current) and for unquoted legs."""
    return pd.to_datetime(
        currencies.map(lambda c: getattr(quotes[c], attr) if c != "USD" and c in quotes else None)
    )


def _leg_flag(currencies: pd.Series, quotes: Mapping[str, Quote], attr: str) -> pd.Series:
    return currencies.map(lambda c: bool(getattr(quotes[c], attr)) if c in quotes else False).astype(bool)


def _earliest(a: pd.Series, b: pd.Series) -> pd.Series:
    """Elementwise earlier of two leg dates, ignoring legs with no date."""
    return pd.concat([a, b], axis=1).min(axis=1)


def _leg_messages(base: str, term: str, quotes: Mapping[str, Quote]) -> list[str]:
    legs = [c for c in (base, term) if c != "USD"]
    messages = [f"no market data for {c} leg" for c in legs if c not in quotes]
    priced = [c for c in legs if c in quotes]

    named: set[str] = set()
    if len(priced) == 2 and quotes[priced[0]].as_of != quotes[priced[1]].as_of:
        behind, ahead = sorted(priced, key=lambda c: quotes[c].as_of)
        lag = _fx_business_days_between(quotes[behind].as_of, quotes[ahead].as_of)
        stale = ", stale" if quotes[behind].stale else ""
        messages.append(
            f"{behind} leg (as of {quotes[behind].as_of}{stale}) is {lag} FX business "
            f"day{'' if lag == 1 else 's'} behind the {ahead} leg (as of {quotes[ahead].as_of})"
        )
        named.add(behind)

    messages += [
        f"{c} leg quote as of {quotes[c].as_of} is stale"
        for c in priced
        if quotes[c].stale and c not in named
    ]
    return messages


def _fx_business_days_between(earlier, later) -> int:
    return len(pd.date_range(earlier, later, freq=FX_BUSINESS_DAY)) - 1
