from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data.portfolio_loader import load_portfolio
from src.models.schema import PortfolioSchemaError

SAMPLE_PORTFOLIO = Path(__file__).resolve().parent.parent / "data" / "sample_portfolio.csv"

HEADER = "trade_id,trade_date,currency_pair,side,notional_base_ccy,entry_price"


def _write_csv(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "portfolio.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


# --- sample_portfolio.csv -------------------------------------------------


def test_sample_portfolio_loads():
    result = load_portfolio(SAMPLE_PORTFOLIO)

    assert result.report.total_rows == 17
    assert result.report.rejected_count == 0
    assert result.report.accepted_count == 17
    assert len(result.trades) == 17


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
            "T0001,2026-07-14,EUR/USD,BUY,0,1.0865",
            "T0002,2026-08-01,GBP/USD,SELL,2000000,1.2740",
        ],
    )

    result = load_portfolio(path)

    assert result.report.accepted_count == 1
    assert result.report.rejected_count == 1
    assert result.trades.iloc[0]["trade_id"] == "T0002"
    assert "notional_base_ccy" in result.report.rejected_rows[0].reason


def test_negative_entry_price_rejected(tmp_path):
    path = _write_csv(tmp_path, ["T0001,2026-07-14,EUR/USD,BUY,5000000,-1.0865"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "entry_price" in result.report.rejected_rows[0].reason


def test_bad_date_format_rejected(tmp_path):
    # Matches none of schema.DATE_FORMATS (day 32 isn't valid in any of them).
    path = _write_csv(tmp_path, ["T0001,32-13-2026,EUR/USD,BUY,5000000,1.0865"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "trade_date" in result.report.rejected_rows[0].reason
    assert "expected one of" in result.report.rejected_rows[0].reason


@pytest.mark.parametrize(
    "raw_date, expected",
    [
        ("2026-07-14", date(2026, 7, 14)),  # ISO 8601, canonical
        ("14-07-2026", date(2026, 7, 14)),  # day-first, 4-digit year
        ("14-07-26", date(2026, 7, 14)),  # day-first, 2-digit year -- the reported case
        ("14/07/2026", date(2026, 7, 14)),  # day-first, slash, 4-digit year
        ("14/07/26", date(2026, 7, 14)),  # day-first, slash, 2-digit year
    ],
)
def test_accepts_every_documented_date_format(tmp_path, raw_date, expected):
    path = _write_csv(tmp_path, [f"T0001,{raw_date},EUR/USD,BUY,5000000,1.0865"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 0
    assert result.trades.iloc[0]["trade_date"] == pd.Timestamp(expected)


def test_month_first_dates_are_not_guessed_and_either_reject_or_reinterpret_day_first(tmp_path):
    # "07-14-2026" (US MM-DD-YYYY) has no day-first reading either (14 can't be a
    # month), so it's correctly rejected rather than silently misparsed.
    path = _write_csv(tmp_path, ["T0001,07-14-2026,EUR/USD,BUY,5000000,1.0865"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "trade_date" in result.report.rejected_rows[0].reason


def test_ambiguous_day_month_is_read_day_first_not_month_first(tmp_path):
    # "03-04-2026": genuinely ambiguous between US (Apr 3) and day-first (Mar 4).
    # DATE_FORMATS deliberately has no month-first entry, so this resolves to the
    # single day-first reading rather than guessing -- documented, not silent.
    path = _write_csv(tmp_path, ["T0001,03-04-2026,EUR/USD,BUY,5000000,1.0865"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 0
    assert result.trades.iloc[0]["trade_date"] == pd.Timestamp(date(2026, 4, 3))


def test_invalid_side_rejected(tmp_path):
    path = _write_csv(tmp_path, ["T0001,2026-07-14,EUR/USD,HOLD,5000000,1.0865"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "side" in result.report.rejected_rows[0].reason


def test_malformed_currency_pair_rejected(tmp_path):
    path = _write_csv(tmp_path, ["T0001,2026-07-14,EURUSD,BUY,5000000,1.0865"])

    result = load_portfolio(path)

    assert result.report.rejected_count == 1
    assert "currency_pair" in result.report.rejected_rows[0].reason


def test_duplicate_trade_id_rejected(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "T0001,2026-07-14,EUR/USD,BUY,5000000,1.0865",
            "T0001,2026-08-01,GBP/USD,SELL,2000000,1.2740",
        ],
    )

    result = load_portfolio(path)

    assert result.report.accepted_count == 1
    assert result.report.rejected_count == 1
    assert "duplicate trade_id" in result.report.rejected_rows[0].reason
    # First occurrence is kept; the later duplicate is the one rejected.
    assert result.trades.iloc[0]["currency_pair"] == "EUR/USD"


def test_extra_metadata_columns_are_ignored_if_present(tmp_path):
    # The spec's optional metadata columns (settlement_date, counterparty, ...)
    # are accepted if a CSV still has them, since only the 6 required columns
    # are parsed into Trade.
    path = tmp_path / "portfolio.csv"
    path.write_text(
        HEADER + ",settlement_date,counterparty\n"
        "T0001,2026-07-14,EUR/USD,BUY,5000000,1.0865,2026-07-16,Bank A\n",
        encoding="utf-8",
    )

    result = load_portfolio(path)

    assert result.report.accepted_count == 1
    assert "settlement_date" not in result.trades.columns


# --- file-like sources (e.g. st.file_uploader) ------------------------------


def test_load_portfolio_accepts_a_text_file_like_object():
    import io

    buffer = io.StringIO("\n".join([HEADER, "T0001,2026-07-14,EUR/USD,BUY,5000000,1.0865"]) + "\n")
    buffer.name = "upload.csv"

    result = load_portfolio(buffer)

    assert result.report.accepted_count == 1
    assert result.trades.iloc[0]["trade_id"] == "T0001"


def test_load_portfolio_accepts_a_bytes_file_like_object_like_an_upload():
    import io

    content = ("\n".join([HEADER, "T0001,2026-07-14,EUR/USD,BUY,5000000,1.0865"]) + "\n").encode("utf-8")
    buffer = io.BytesIO(content)
    buffer.name = "upload.csv"  # st.file_uploader's UploadedFile exposes .name the same way

    result = load_portfolio(buffer)

    assert result.report.accepted_count == 1
    assert result.trades.iloc[0]["trade_id"] == "T0001"


def test_load_portfolio_file_like_source_still_rejects_bad_rows():
    import io

    buffer = io.StringIO(
        "\n".join(
            [
                HEADER,
                "T0001,2026-07-14,EUR/USD,BUY,5000000,1.0865",
                "T0002,2026-08-01,GBP/USD,SELL,-2000000,1.2740",
            ]
        )
        + "\n"
    )

    result = load_portfolio(buffer)

    assert result.report.accepted_count == 1
    assert result.report.rejected_count == 1
    assert "notional_base_ccy" in result.report.rejected_rows[0].reason


def test_load_portfolio_file_like_source_without_name_gets_a_generic_label():
    import io

    buffer = io.StringIO("")  # no .name attribute set, like a raw buffer

    with pytest.raises(PortfolioSchemaError, match="uploaded file"):
        load_portfolio(buffer)
