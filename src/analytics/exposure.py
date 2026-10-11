"""Currency exposure breakdown and VaR concentration, the Requirements Checklist's
'Currency exposure breakdown (net USD exposure by currency)' and 'Risk concentration
view (% of portfolio VaR by position/pair)'.

Currency exposure extends risk.py's w^(A)_i/w^(B)_i formula to every currency a
position touches, USD included. risk.py deliberately drops USD from its own w
(zero variance, irrelevant to VaR), but an exposure breakdown has no such reason to
omit it -- a PM looking at "how much USD am I net long/short" is a legitimate,
separate question from "what's driving my VaR". VaR concentration has no new math
at all: it is risk.py's own component VaR (which already reconciles exactly to
portfolio_var by Euler's theorem), simply read off as a percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.analytics.risk import VaRResult, position_exposures, standalone_var
from src.data.market_data import Quote


def _u(currency: str, quotes: Mapping[str, Quote]) -> float | None:
    """U_C: 1.0 for USD by construction, otherwise the quote's value if priced."""
    if currency == "USD":
        return 1.0
    quote = quotes.get(currency)
    return quote.value if quote is not None else None


@dataclass(frozen=True)
class CurrencyExposure:
    net_exposure: pd.Series  # currency -> net USD exposure, USD included
    excluded_trade_ids: tuple[str, ...]  # no market data for a leg; contribute to no currency

    @property
    def gross_exposure(self) -> float:
        return float(self.net_exposure.abs().sum())

    def as_frame(self) -> pd.DataFrame:
        """net_exposure and pct_of_gross, sorted largest absolute exposure first."""
        gross = self.gross_exposure
        pct = self.net_exposure.abs() / gross * 100 if gross > 0 else self.net_exposure * 0.0
        df = pd.DataFrame({"net_exposure": self.net_exposure, "pct_of_gross": pct})
        return df.reindex(df["net_exposure"].abs().sort_values(ascending=False).index)


def compute_currency_exposure(trades: pd.DataFrame, quotes: Mapping[str, Quote]) -> CurrencyExposure:
    """Net USD exposure per currency: w^(A)_i = s_i*N_i*U_A,t summed over every
    position's base-currency leg, plus w^(B)_i = -s_i*N_i*S_i,0*U_B,t over every
    term-currency leg (USD included in both, unlike risk.position_exposures). A
    position missing a quote for either leg contributes to no currency at all and
    is reported separately, consistent with pnl.py/risk.py's no-market-data handling."""
    rows: dict[str, dict[str, float]] = {}
    excluded: list[str] = []

    for _, t in trades.iterrows():
        u_base, u_term = _u(t["base_currency"], quotes), _u(t["term_currency"], quotes)
        if u_base is None or u_term is None:
            excluded.append(t["trade_id"])
            continue
        legs: dict[str, float] = {}
        legs[t["base_currency"]] = legs.get(t["base_currency"], 0.0) + t["side_sign"] * t["notional_base_ccy"] * u_base
        legs[t["term_currency"]] = legs.get(t["term_currency"], 0.0) - (
            t["side_sign"] * t["notional_base_ccy"] * t["entry_price"] * u_term
        )
        rows[t["trade_id"]] = legs

    if not rows:
        return CurrencyExposure(net_exposure=pd.Series(dtype=float), excluded_trade_ids=tuple(excluded))

    exposures = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    net = exposures.sum(axis=0)
    net.index.name = "currency"
    return CurrencyExposure(net_exposure=net, excluded_trade_ids=tuple(excluded))


def var_concentration(result: VaRResult, trades: pd.DataFrame) -> pd.DataFrame:
    """Position-level % of portfolio VaR, from risk.py's component VaR. Sorted by
    absolute contribution, largest first. Positions risk.py excluded (no market
    data) have no row here -- see result.excluded_trade_ids."""
    df = result.positions.copy()
    df["pct_of_portfolio_var"] = (
        df["component_var"] / result.portfolio_var * 100 if result.portfolio_var > 0 else 0.0
    )
    df = df.join(trades.set_index("trade_id")["currency_pair"])
    return df.reindex(df["component_var"].abs().sort_values(ascending=False).index)


def pair_summary(
    trades: pd.DataFrame, quotes: Mapping[str, Quote], pnl_df: pd.DataFrame, result: VaRResult
) -> pd.DataFrame:
    """Primary (pair-level) view of the position table: one row per currency_pair with
    net_notional, inception_pnl, daily_pnl, standalone_var and component_var.

    net_notional, inception_pnl, daily_pnl and component_var are all exact sums of
    their trade-level figures -- net_notional because every trade sharing a pair
    shares its base currency, so signed notionals (s_i*N_i) add directly; P&L
    because inception/daily P&L are linear in each trade's own mark; component_var
    because it is already built from marginal VaR, which reflects the whole
    portfolio, so summing it over a subset is exact (Euler's theorem restricted to
    that subset), not an approximation.

    standalone_var is NOT a sum of the pair's trades' own standalone_var figures --
    that would overstate risk whenever two trades on the same pair partly offset,
    since it throws away exactly the netting "position VaR" is meant to capture.
    Each trade's own USD exposure vector is computed first (entry price and all,
    via risk.position_exposures -- required for a correct exposure even on a
    USD-base or cross pair), the vectors are summed per pair, and only then is
    risk.standalone_var's quadratic form applied to that net vector directly.

    A pair with no VaR-eligible trades (every trade on it lacking market data) gets
    NaN for standalone_var and component_var, not 0 and not an exception."""
    pair_of_trade = trades.set_index("trade_id")["currency_pair"]

    net_notional = (trades["side_sign"] * trades["notional_base_ccy"]).groupby(trades["currency_pair"]).sum()
    pnl_sums = pnl_df.groupby("currency_pair")[["inception_pnl", "daily_pnl"]].sum(min_count=1)

    exposures, _ = position_exposures(trades, quotes, result.currencies)
    if exposures.empty:
        pair_exposures = pd.DataFrame(columns=result.currencies)
    else:
        pair_exposures = exposures.groupby(pair_of_trade.reindex(exposures.index)).sum()
    standalone = standalone_var(pair_exposures, result.sigma, result.z, result.scale)

    component_by_trade = result.positions["component_var"]
    component = component_by_trade.groupby(pair_of_trade.reindex(component_by_trade.index)).sum()

    summary = pd.DataFrame(index=pd.Index(trades["currency_pair"].unique(), name="currency_pair"))
    summary["net_notional"] = net_notional
    summary["inception_pnl"] = pnl_sums["inception_pnl"]
    summary["daily_pnl"] = pnl_sums["daily_pnl"]
    summary["standalone_var"] = standalone.reindex(summary.index)
    summary["component_var"] = component.reindex(summary.index)
    summary["pct_of_portfolio_var"] = (
        summary["component_var"] / result.portfolio_var * 100 if result.portfolio_var > 0 else 0.0
    )
    return summary.reindex(summary["component_var"].abs().sort_values(ascending=False).index)


def total_component_var(table: pd.DataFrame) -> float:
    """Sum of a position/pair table's component_var column. By Euler's theorem this
    must equal the portfolio VaR it was derived from (var_concentration's trade-level
    sum and pair_summary's pair-level sum both do); computed here, not inline where
    a page displays it, so a reconciliation check is never silently faked."""
    return float(table["component_var"].sum())


def total_pct_of_portfolio_var(table: pd.DataFrame) -> float:
    """Sum of a position/pair table's pct_of_portfolio_var column -- must equal
    100% for the same reason total_component_var must equal portfolio_var; the two
    are the same identity, one normalised."""
    return float(table["pct_of_portfolio_var"].sum())


def pair_standalone_sum(pair_table: pd.DataFrame) -> float:
    """Sum of pair_summary's own (net, per-pair) standalone_var column -- the
    undiversified baseline for a diversification-benefit figure that only counts
    genuine cross-currency diversification, not trivial netting within a pair.
    NaN rows (a pair with no VaR-eligible trades) are excluded, not propagated."""
    return float(pair_table["standalone_var"].sum(min_count=1))


def pair_diversification_benefit(pair_table: pd.DataFrame, portfolio_var: float) -> float:
    """Diversification benefit computed at pair (net) level: the sum of each pair's
    own net standalone VaR, minus portfolio VaR.

    This is deliberately NOT risk.VaRResult.diversification_benefit, which sums
    TRADE-level standalone VaR. That trade-level sum is the right baseline for the
    spec's own position-level subadditivity invariant (portfolio VaR <= sum of
    trade standalones), but it is the wrong baseline for a headline "diversification
    benefit" figure: summing two offsetting trades' own gross standalone VaRs
    double-counts trivial within-pair netting as if it were genuine diversification
    across currencies. pair_summary's standalone_var is already netted per pair (see
    its own docstring), so summing it isolates the benefit that actually comes from
    holding multiple, imperfectly-correlated currency exposures."""
    return pair_standalone_sum(pair_table) - portfolio_var
