"""Portfolio CSV schema: required/optional columns, the Trade record, and
validation-result containers used by src/data/portfolio_loader.py.

Column definitions and the sign convention below follow the
'Portfolio data schema (CSV)' and 'Sign convention' sections of
FX_Portfolio_Dashboard_Spec.md exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

REQUIRED_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "trade_date",
    "currency_pair",
    "side",
    "notional_base_ccy",
    "entry_price",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "settlement_date",
    "counterparty",
    "book",
    "trader",
    "notes",
)

ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

Side = Literal["BUY", "SELL"]
VALID_SIDES: tuple[str, ...] = ("BUY", "SELL")

DATE_FORMAT = "%Y-%m-%d"

# BASE/TERM, e.g. EUR/USD, USD/JPY — exactly two 3-letter uppercase codes.
CURRENCY_PAIR_RE = re.compile(r"^[A-Z]{3}/[A-Z]{3}$")


@dataclass(frozen=True)
class Trade:
    """One validated row of the portfolio CSV."""

    trade_id: str
    trade_date: date
    currency_pair: str
    side: Side
    notional_base_ccy: float
    entry_price: float
    settlement_date: Optional[date] = None
    counterparty: Optional[str] = None
    book: Optional[str] = None
    trader: Optional[str] = None
    notes: Optional[str] = None

    @property
    def base_currency(self) -> str:
        return self.currency_pair.split("/")[0]

    @property
    def term_currency(self) -> str:
        return self.currency_pair.split("/")[1]

    @property
    def side_sign(self) -> int:
        """s_i: +1 for BUY (long base), -1 for SELL (short base)."""
        return 1 if self.side == "BUY" else -1

    @property
    def is_cross(self) -> bool:
        """True when neither leg is USD (e.g. EUR/GBP)."""
        return "USD" not in (self.base_currency, self.term_currency)


@dataclass(frozen=True)
class RejectedRow:
    """A CSV row that failed validation and was excluded from the portfolio."""

    row_number: int  # 1-indexed position among data rows (header excluded)
    trade_id: Optional[str]
    reason: str


@dataclass
class ValidationReport:
    """Summary of a portfolio load: how many rows were accepted/rejected and why."""

    total_rows: int
    rejected_rows: list[RejectedRow]

    @property
    def accepted_count(self) -> int:
        return self.total_rows - len(self.rejected_rows)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_rows)

    @property
    def has_errors(self) -> bool:
        return self.rejected_count > 0


class PortfolioSchemaError(Exception):
    """Raised for file-level schema problems that make the whole file unloadable
    (e.g. a required column missing from the header, or the file is unreadable).
    Per-row problems are collected as RejectedRow entries instead, so one bad
    row never prevents the rest of the portfolio from loading.
    """
