from pathlib import Path

import pytest

from src.data.portfolio_loader import load_portfolio
from src.models.schema import PortfolioSchemaError

SAMPLE_PORTFOLIO = Path(__file__).resolve().parent.parent / "data" / "sample_portfolio.csv"

HEADER = (
    "trade_id,trade_date,currency_pair,side,notional_base_ccy,entry_price,"
    "settlement_date,counterparty,book,trader,notes"
)


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "portfolio.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


# --- sample_portfolio.csv -------------------------------------------------


def test_sample_portfolio_loads():
    result = load_portfolio(SAMPLE_PORTFOLIO)

    assert result.report.total_rows == 18
    assert result.report.rejected_count == 1
    assert result.report.accepted_count == 17
    assert len(result.trades) == 17


def test_sample_portfolio_rejects_the_deliberately_invalid_row():
    result = load_portfolio(SAMPLE_PORTFOLIO)

    rejected = result.report.rejected_rows[0]
    assert rejected.trade_id == "T0018"
    assert "notional_base_ccy" in rejected.reason


def test_sample_portfolio_has_no_duplicate_trade_ids():
    result = load_portfolio(SAMPLE_PORTFOLIO)

    assert result.trades["trade_id"].is_unique


def test_sample_portfolio_includes_cross_pairs():
    result = load_portfolio(SAMPLE_PORTFOLIO)
    df = result.trades

    cross_rows = df[df["currency_pair"].isin(["EUR/GBP", "AUD/JPY"])]
    assert len(cross_rows) == 2
    assert cross_rows["is_cross"].all()

    major_rows = df[df["currency_pair"] == "EUR/USD"]
    assert not major_rows["is_cross"].any()

    eurgbp = df[df["currency_pair"] == "EUR/GBP"].iloc[0]
    assert eurgbp["base_currency"] == "EUR"
    assert eurgbp["term_currency"] == "GBP"


def test_sample_portfolio_side_sign():
    result = load_portfolio(SAMPLE_PORTFOLIO)
    df = result.trades

    assert (df.loc[df["side"] == "BUY", "side_sign"] == 1).all()
    assert (df.loc[df["side"] == "SELL", "side_sign"] == -1).all()


# --- edge cases from the spec's 'Edge cases & data-quality risks' table --


def test_missing_required_column_raises_schema_error(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "trade_id,trade_date,currency_pair,side,notional_base_ccy\n"
        "T0001,2026-07-14,EUR/USD,BUY,5000000\n",
        encoding="utf-8",
    )

    with pytest.raises(PortfolioSchemaError):
        load_portfolio(path)


def test_empty_file_raises_schema_error(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(PortfolioSchemaError):
        load_portfolio(path)


def test_zero_or_negative_notional_rejected_but_rest_still_loads(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "T0001,2026-07-14,EUR/USD,BUY,0,1.0865,,,,,",
            "T0002,2026-08-01,GBP/USD,SELL,2000000,1.2740,,,,,",
        ],
    )

    result = load_portfolio(path)

    assert result.report.accepted_count == 1
    assert result.report.rejected_count == 1
    assert result.trades.iloc[0]["trade_id"] == "T0002"
    assert "notional_base_ccy" in result.report.rejected_rows[0].reason


def test_negative_entry_price_rejected(tmp_path):
    path = _write_csv(tmp_path, ["T0001,2026-07-14,EUR/USD,BUY,5000000,-1.0865,,,,,"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "entry_price" in result.report.rejected_rows[0].reason


def test_bad_date_format_rejected(tmp_path):
    path = _write_csv(tmp_path, ["T0001,14-07-2026,EUR/USD,BUY,5000000,1.0865,,,,,"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "trade_date" in result.report.rejected_rows[0].reason


def test_invalid_side_rejected(tmp_path):
    path = _write_csv(tmp_path, ["T0001,2026-07-14,EUR/USD,HOLD,5000000,1.0865,,,,,"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "side" in result.report.rejected_rows[0].reason


def test_malformed_currency_pair_rejected(tmp_path):
    path = _write_csv(tmp_path, ["T0001,2026-07-14,EURUSD,BUY,5000000,1.0865,,,,,"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "currency_pair" in result.report.rejected_rows[0].reason


def test_duplicate_trade_id_rejected(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "T0001,2026-07-14,EUR/USD,BUY,5000000,1.0865,,,,,",
            "T0001,2026-08-01,GBP/USD,SELL,2000000,1.2740,,,,,",
        ],
    )

    result = load_portfolio(path)

    assert result.report.accepted_count == 1
    assert result.report.rejected_count == 1
    assert "duplicate trade_id" in result.report.rejected_rows[0].reason
    # First occurrence is kept; the later duplicate is the one rejected.
    assert result.trades.iloc[0]["currency_pair"] == "EUR/USD"


def test_optional_columns_may_be_blank(tmp_path):
    path = _write_csv(tmp_path, ["T0001,2026-07-14,EUR/USD,BUY,5000000,1.0865,,,,,"])

    result = load_portfolio(path)

    assert result.report.accepted_count == 1
    row = result.trades.iloc[0]
    assert row["counterparty"] is None
    import pandas as pd

    assert pd.isna(row["settlement_date"])
