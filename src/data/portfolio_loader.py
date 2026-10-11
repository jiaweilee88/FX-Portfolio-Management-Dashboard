"""Loads and validates the portfolio CSV described in
'Portfolio data schema (CSV)', applying the loader-relevant rows of the
'Edge cases & data-quality risks' table. load_portfolio accepts a filesystem path
or any already-open readable source (e.g. Streamlit's st.file_uploader() result),
so a dashboard can read either the bundled sample file or a user-uploaded one
through the exact same validation path:

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
import io
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.models.schema import (
    ALL_COLUMNS,
    CURRENCY_PAIR_RE,
    DATE_FORMATS,
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


# A filesystem path, or any already-open readable source with the portfolio's CSV
# text/bytes -- e.g. Streamlit's st.file_uploader() return value. This module has
# no dependency on Streamlit; it only needs .read() and, optionally, .name.
PortfolioSource = Union[str, Path, object]


def _read_portfolio_text(source: PortfolioSource) -> tuple[str, str]:
    """Returns (a label identifying the source for error messages, its raw text)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return str(path), path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PortfolioSchemaError(f"{path}: could not be read ({exc})") from exc

    label = getattr(source, "name", None) or "uploaded file"
    raw = source.read()
    try:
        return label, raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise PortfolioSchemaError(f"{label}: could not be read as UTF-8 text ({exc})") from exc


def load_portfolio(source: PortfolioSource) -> PortfolioLoadResult:
    label, text = _read_portfolio_text(source)

    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames
    if header is None:
        raise PortfolioSchemaError(f"{label}: file is empty, no header row found")

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing_columns:
        raise PortfolioSchemaError(f"{label}: missing required column(s): {', '.join(missing_columns)}")

    rows = list(reader)

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

    return Trade(
        trade_id=trade_id,
        trade_date=trade_date,
        currency_pair=currency_pair,
        side=side,  # type: ignore[arg-type]
        notional_base_ccy=notional,
        entry_price=entry_price
    )


def _parse_date(value: str, field_name: str) -> date:
    """Tries each of schema.DATE_FORMATS in order (ISO 8601 first) and returns the
    first match. A value matching none of them is rejected naming every format
    tried, rather than guessing at an ambiguous one (see DATE_FORMATS' docstring)."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    expected = ", ".join(DATE_FORMATS)
    raise ValueError(f"invalid {field_name}: {value!r} (expected one of: {expected})")


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
    return df
