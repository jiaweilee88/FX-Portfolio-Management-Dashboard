"""Loads and validates the portfolio CSV described in
'Portfolio data schema (CSV)', applying the loader-relevant rows of the
'Edge cases & data-quality risks' table:

- Zero or negative notional/entry_price -> row rejected, rest of file still loads.
- Duplicate trade_id -> rejected and listed in the validation report.
- Malformed rows (missing column, bad date, negative notional) -> rejected with
  a clear error, never a silent crash.

Netting of opposite-signed trades on the same pair, "new position" P&L
flagging, and market-data-driven edge cases (stale rates, missing coverage,
outliers) are handled downstream in the analytics/risk layer and market-data
layer respectively, not here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.schema import (
    ALL_COLUMNS,
    CURRENCY_PAIR_RE,
    DATE_FORMAT,
    REQUIRED_COLUMNS,
    VALID_SIDES,
    PortfolioSchemaError,
    RejectedRow,
    Trade,
    ValidationReport,
)


@dataclass
class PortfolioLoadResult:
    """Return value of load_portfolio: the accepted trades plus a report of
    anything rejected along the way."""

    trades: pd.DataFrame
    report: ValidationReport


def load_portfolio(path: str | Path) -> PortfolioLoadResult:
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            if header is None:
                raise PortfolioSchemaError(f"{path}: file is empty, no header row found")

            missing_columns = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing_columns:
                raise PortfolioSchemaError(
                    f"{path}: missing required column(s): {', '.join(missing_columns)}"
                )

            rows = list(reader)
    except OSError as exc:
        raise PortfolioSchemaError(f"{path}: could not be read ({exc})") from exc

    accepted: list[Trade] = []
    rejected: list[RejectedRow] = []
    seen_trade_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=1):
        trade_id = (row.get("trade_id") or "").strip() or None
        try:
            trade = _parse_row(row, seen_trade_ids)
        except ValueError as exc:
            rejected.append(RejectedRow(row_number=row_number, trade_id=trade_id, reason=str(exc)))
            continue

        seen_trade_ids.add(trade.trade_id)
        accepted.append(trade)

    report = ValidationReport(total_rows=len(rows), rejected_rows=rejected)
    trades_df = _to_dataframe(accepted)
    return PortfolioLoadResult(trades=trades_df, report=report)


def _parse_row(row: dict[str, Optional[str]], seen_trade_ids: set[str]) -> Trade:
    values: dict[str, str] = {}
    for col in ALL_COLUMNS:
        raw = row.get(col)
        if raw is None and col in REQUIRED_COLUMNS:
            raise ValueError(f"missing required field: {col}")
        values[col] = (raw or "").strip()

    for col in REQUIRED_COLUMNS:
        if not values[col]:
            raise ValueError(f"missing required field: {col}")

    trade_id = values["trade_id"]
    if trade_id in seen_trade_ids:
        raise ValueError(f"duplicate trade_id: {trade_id}")

    trade_date = _parse_date(values["trade_date"], "trade_date")

    currency_pair = values["currency_pair"].upper()
    if not CURRENCY_PAIR_RE.match(currency_pair):
        raise ValueError(f"invalid currency_pair format: {values['currency_pair']!r}")
    base, term = currency_pair.split("/")
    if base == term:
        raise ValueError(f"currency_pair base and term must differ: {currency_pair!r}")

    side = values["side"].upper()
    if side not in VALID_SIDES:
        raise ValueError(f"invalid side: {values['side']!r} (must be BUY or SELL)")

    notional = _parse_positive_float(values["notional_base_ccy"], "notional_base_ccy")
    entry_price = _parse_positive_float(values["entry_price"], "entry_price")

    settlement_date = (
        _parse_date(values["settlement_date"], "settlement_date")
        if values["settlement_date"]
        else None
    )

    return Trade(
        trade_id=trade_id,
        trade_date=trade_date,
        currency_pair=currency_pair,
        side=side,  # type: ignore[arg-type]
        notional_base_ccy=notional,
        entry_price=entry_price,
        settlement_date=settlement_date,
        counterparty=values["counterparty"] or None,
        book=values["book"] or None,
        trader=values["trader"] or None,
        notes=values["notes"] or None,
    )


def _parse_date(value: str, field_name: str) -> date:
    try:
        return datetime.strptime(value, DATE_FORMAT).date()
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: {value!r} (expected YYYY-MM-DD)") from exc


def _parse_positive_float(value: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: {value!r} (expected a number)") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0, got {parsed}")
    return parsed


def _to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    columns = list(ALL_COLUMNS) + ["base_currency", "term_currency", "side_sign", "is_cross"]
    if not trades:
        return pd.DataFrame(columns=columns)

    records = [
        {
            **{col: getattr(t, col) for col in ALL_COLUMNS},
            "base_currency": t.base_currency,
            "term_currency": t.term_currency,
            "side_sign": t.side_sign,
            "is_cross": t.is_cross,
        }
        for t in trades
    ]
    df = pd.DataFrame(records, columns=columns)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df
