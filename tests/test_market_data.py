import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data import market_data as md

REAL_YF_HISTORY = md._yf_history

TODAY = date(2026, 10, 9)  # Friday
NOW = datetime(2026, 10, 9, 18, 0)


def _bdays(n, end):
    return pd.bdate_range(end=end, periods=n)


def _series(n=300, end=TODAY - timedelta(days=1), start=1.10, seed=0, name="X"):
    rng = np.random.default_rng(seed)
    values = (start * np.exp(np.cumsum(rng.normal(0, 0.005, n)))).round(5)
    return pd.Series(values, index=pd.DatetimeIndex(_bdays(n, end), name="date"), name=name)


def _closes(values, end=TODAY):
    return pd.Series(values, index=pd.DatetimeIndex(_bdays(len(values), end), name="date"), dtype=float)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(symbol, **kwargs):
        raise AssertionError(f"unexpected yfinance call for {symbol}")

    monkeypatch.setattr(md, "_yf_history", blocked)
    monkeypatch.setattr(md, "RETRY_DELAY_S", 0)


def _serve(monkeypatch, history=None, current=None):
    """Fake yfinance: `history` answers start/end requests, `current` answers period requests.
    Each maps symbol -> Series or Exception; unmapped symbols fail."""
    calls = []

    def fake(symbol, **kwargs):
        kind = "current" if "period" in kwargs else "history"
        calls.append((kind, symbol, kwargs))
        result = ((current if kind == "current" else history) or {}).get(
            symbol, md.MarketDataError(f"yfinance {symbol}: down")
        )
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(md, "_yf_history", fake)
    return calls


# --- ticker mapping & client -------------------------------------------------


@pytest.mark.parametrize(
    "currency, symbol, inverted",
    [
        ("EUR", "EURUSD=X", False), ("GBP", "GBPUSD=X", False), ("AUD", "AUDUSD=X", False),
        ("NZD", "NZDUSD=X", False), ("JPY", "JPY=X", True), ("CHF", "CHF=X", True),
        ("CAD", "CAD=X", True), ("CNY", "CNY=X", True), ("MXN", "MXN=X", True),
        ("SGD", "SGD=X", True), ("HKD", "HKD=X", True), ("SEK", "SEK=X", True),
        ("NOK", "NOK=X", True), ("INR", "INR=X", True), ("ZAR", "ZAR=X", True),
        ("BRL", "BRL=X", True), ("THB", "THB=X", True), ("IDR", "IDR=X", True),
        ("MYR", "MYR=X", True), ("PHP", "PHP=X", True), ("HUF", "HUF=X", True),
        ("ILS", "ILS=X", True),
    ],
)
def test_ticker_mapping(currency, symbol, inverted):
    assert md.YF_TICKERS[currency] == md.YfTicker(symbol, inverted)


def test_ticker_mapping_has_no_untested_entries():
    tested = {
        "EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "CNY", "MXN", "SGD", "HKD",
        "SEK", "NOK", "INR", "ZAR", "BRL", "THB", "IDR", "MYR", "PHP", "HUF", "ILS",
    }
    assert set(md.YF_TICKERS) == tested


def test_fetch_history_requests_800_days_and_drops_todays_session(monkeypatch):
    series = _series(10, end=TODAY)  # last row is today's still-forming bar
    calls = _serve(monkeypatch, history={"EURUSD=X": series})

    raw = md.fetch_history(md.YF_TICKERS["EUR"], TODAY)

    _, symbol, kwargs = calls[0]
    assert symbol == "EURUSD=X"
    assert kwargs["start"] == (TODAY - timedelta(days=800)).isoformat()
    assert kwargs["end"] == TODAY.isoformat()
    assert raw.index[-1] < pd.Timestamp(TODAY)
    assert len(raw) == 9


def test_fetch_retries_once_then_raises(monkeypatch):
    calls = _serve(monkeypatch)

    with pytest.raises(md.MarketDataError):
        md.fetch_history(md.YF_TICKERS["EUR"], TODAY)

    assert len(calls) == 2


def test_fetch_succeeds_on_retry(monkeypatch):
    attempts = []

    def flaky(symbol, **kwargs):
        attempts.append(symbol)
        if len(attempts) == 1:
            raise md.MarketDataError("rate limited")
        return _closes([1.17, 1.18])

    monkeypatch.setattr(md, "_yf_history", flaky)

    quote = md.fetch_current_quote(md.YF_TICKERS["EUR"], TODAY)

    assert len(attempts) == 2 and quote.value == 1.18


def _fake_ticker(monkeypatch, frame=None, exc=None):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            if exc:
                raise exc
            return frame

    monkeypatch.setattr(md.yf, "Ticker", FakeTicker)


def test_yf_history_dedupes_repeated_latest_session_and_strips_tz(monkeypatch):
    index = pd.DatetimeIndex(
        ["2026-10-07 00:00", "2026-10-08 00:00", "2026-10-09 00:00", "2026-10-09 14:30"], tz="Europe/London"
    )
    _fake_ticker(monkeypatch, pd.DataFrame({"Close": [1.16, 1.17, 1.175, 1.18]}, index=index))

    closes = REAL_YF_HISTORY("EURUSD=X", period="5d")

    assert [d.date() for d in closes.index] == [date(2026, 10, 7), date(2026, 10, 8), date(2026, 10, 9)]
    assert closes.index.tz is None
    assert closes.iloc[-1] == 1.18


def test_yf_history_folds_weekend_snapshot_row_into_friday_session(monkeypatch):
    # Seen live: GBPUSD=X on a Saturday returns a row dated Sat 2026-10-10 holding Friday's final price
    index = pd.DatetimeIndex(["2026-10-08", "2026-10-09", "2026-10-10"], tz="Europe/London")
    _fake_ticker(monkeypatch, pd.DataFrame({"Close": [1.321493, 1.322909, 1.323382]}, index=index))

    closes = REAL_YF_HISTORY("GBPUSD=X", period="5d")

    assert [d.date() for d in closes.index] == [date(2026, 10, 8), date(2026, 10, 9)]
    assert closes.iloc[-1] == 1.323382  # Friday's session carries the latest price


def test_yf_history_folds_fx_closure_row_into_prior_session(monkeypatch):
    index = pd.DatetimeIndex(["2026-12-24", "2026-12-25"], tz="Europe/London")
    _fake_ticker(monkeypatch, pd.DataFrame({"Close": [1.10, 1.11]}, index=index))

    closes = REAL_YF_HISTORY("EURUSD=X", period="5d")

    assert [d.date() for d in closes.index] == [date(2026, 12, 24)]
    assert closes.iloc[-1] == 1.11


@pytest.mark.parametrize(
    "frame, exc",
    [(pd.DataFrame(), None), (pd.DataFrame({"Close": []}), None), (None, RuntimeError("Yahoo broke"))],
    ids=["no-columns", "empty", "exception"],
)
def test_yf_history_failures_become_market_data_errors(monkeypatch, frame, exc):
    _fake_ticker(monkeypatch, frame, exc)

    with pytest.raises(md.MarketDataError):
        REAL_YF_HISTORY("EURUSD=X", period="5d")


# --- cleaning, canonical series, flags --------------------------------------


def test_clean_quotes_ffills_up_to_3_bdays_and_drops_the_rest():
    dates = _bdays(12, TODAY)
    values = [1.0, np.nan, np.nan, 1.1, np.nan, np.nan, np.nan, np.nan, np.nan, 1.2, 1.3, 1.4]
    # yfinance omits missing days rather than sending NaN rows
    raw = pd.Series(values, index=dates).dropna()

    cleaned = md.clean_quotes(raw)

    assert cleaned.loc[dates[1]] == 1.0 and cleaned.loc[dates[2]] == 1.0
    assert all(cleaned.loc[dates[i]] == 1.1 for i in (4, 5, 6))
    assert dates[7] not in cleaned.index and dates[8] not in cleaned.index
    assert len(cleaned) == 10


def test_clean_quotes_does_not_insert_fx_closures():
    # Dec 24 (Thu), Dec 28 (Mon) 2026; Dec 25 is an FX closure, not a gap to fill
    raw = pd.Series([1.10, 1.12], index=pd.DatetimeIndex(["2026-12-24", "2026-12-28"]))

    cleaned = md.clean_quotes(raw)

    assert list(cleaned.index) == [pd.Timestamp("2026-12-24"), pd.Timestamp("2026-12-28")]


def test_to_canonical_direct_and_inverted():
    quotes = pd.Series([1.10, 150.0])
    assert md.to_canonical(quotes, inverted=False).tolist() == [1.10, 150.0]
    assert md.to_canonical(quotes, inverted=True).tolist() == pytest.approx([1 / 1.10, 1 / 150.0])


def test_repeated_rate_runs_flags_more_than_two_bdays():
    dates = _bdays(7, TODAY)
    quotes = pd.Series([1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.3], index=dates)

    assert md.repeated_rate_runs(quotes) == [(dates[0].date(), dates[2].date(), 3)]


def test_outlier_flag_beyond_5_sigma():
    rng = np.random.default_rng(1)
    dates = _bdays(250, TODAY)
    returns = pd.Series(rng.normal(0, 0.005, 250), index=dates)
    returns.iloc[100] = 0.10

    assert md.outlier_dates(returns) == [dates[100].date()]


def test_log_returns_uses_inner_join_of_dates():
    dates = _bdays(5, TODAY)
    levels = pd.DataFrame(
        {"EUR": [1.10, 1.11, np.nan, 1.12, 1.13], "JPY": [0.0068, 0.0069, 0.0070, np.nan, 0.0071]},
        index=dates,
    )

    r = md.log_returns(levels, ["EUR", "JPY"])

    assert list(r.index) == [dates[1], dates[4]]
    assert r.loc[dates[4], "EUR"] == pytest.approx(np.log(1.13 / 1.11))


# --- FX trading calendar & staleness ----------------------------------------


@pytest.mark.parametrize(
    "today, expected",
    [
        (date(2026, 10, 10), date(2026, 10, 9)),  # Saturday -> Friday
        (date(2026, 10, 12), date(2026, 10, 9)),  # Monday -> Friday
        (date(2026, 10, 7), date(2026, 10, 6)),  # Wednesday -> Tuesday
        (date(2026, 10, 13), date(2026, 10, 12)),  # Columbus Day is an FX trading day
        (date(2026, 11, 27), date(2026, 11, 26)),  # Thanksgiving is an FX trading day
        (date(2026, 12, 28), date(2026, 12, 24)),  # Christmas (Fri) is an FX closure
        (date(2027, 1, 4), date(2026, 12, 31)),  # New Year's Day (Fri) is an FX closure
    ],
)
def test_last_business_day_uses_fx_calendar(today, expected):
    assert md.last_business_day(today) == expected


@pytest.mark.parametrize(
    "last_close, today",
    [
        (date(2026, 12, 24), date(2026, 12, 26)),  # Thu before / Sat after Christmas (Fri)
        (date(2026, 12, 31), date(2027, 1, 2)),  # Thu before / Sat after New Year's Day (Fri)
        (date(2026, 12, 24), date(2026, 12, 28)),  # Mon morning, before Monday's bar exists
    ],
    ids=["christmas", "new-year", "monday-after-christmas"],
)
def test_no_false_stale_after_an_fx_market_closure(last_close, today):
    """A naive Mon-Fri check would expect a quote on Dec 25 / Jan 1 and flag this stale."""
    quote = md._quote_from_series(_closes([1.10, 1.11], end=last_close), md.SOURCE_YFINANCE, today=today)

    assert not quote.stale
    assert quote.label == "indicative, not executable"


def test_us_bank_holiday_is_an_fx_trading_day_for_staleness():
    """FX trades through Columbus Day, so missing that day's quote is genuinely stale
    and having it is current (a bank-holiday calendar would get both wrong)."""
    missed_holiday = md._quote_from_series(
        _closes([1.10, 1.11], end=date(2026, 10, 9)), md.SOURCE_YFINANCE, today=date(2026, 10, 13)
    )
    has_holiday = md._quote_from_series(
        _closes([1.10, 1.11], end=date(2026, 10, 12)), md.SOURCE_YFINANCE, today=date(2026, 10, 13)
    )

    assert missed_holiday.stale
    assert not has_holiday.stale


# --- history: yfinance with parquet cache -------------------------------------


def test_history_success_writes_cache(tmp_path, monkeypatch):
    series = _series(name="EURUSD=X")
    _serve(monkeypatch, history={"EURUSD=X": series})

    history = md.load_history(["EUR", "USD"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert (tmp_path / "EUR.parquet").exists()
    status = history.status["EUR"]
    assert status.history_source == md.SOURCE_YFINANCE and not status.low_confidence
    assert status.as_of == series.index[-1].date()
    assert history.var_currencies == ["EUR"]


def test_inverted_currency_canonical_levels(tmp_path, monkeypatch):
    series = _series(start=150.0, name="JPY=X")
    _serve(monkeypatch, history={"JPY=X": series})

    history = md.load_history(["JPY"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert history.levels["JPY"].iloc[-1] == pytest.approx(1 / series.iloc[-1])


def test_usd_levels_are_unity(tmp_path, monkeypatch):
    _serve(monkeypatch, history={"EURUSD=X": _series()})

    history = md.load_history(["EUR", "USD"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert (history.levels["USD"] == 1.0).all()
    assert "USD" not in history.status


def test_fresh_cache_skips_network(tmp_path):
    md._write_cache(tmp_path, "EUR", _series())

    # autouse fixture fails the test on any yfinance call
    history = md.load_history(["EUR"], cache_dir=tmp_path, today=TODAY, now=datetime.now())

    assert history.status["EUR"].history_source == md.SOURCE_CACHED
    assert not history.status["EUR"].is_stale


def test_outage_uses_stale_cache_for_history(tmp_path, monkeypatch):
    md._write_cache(tmp_path, "EUR", _series())
    three_days_ago = (datetime.now() - timedelta(days=3)).timestamp()
    os.utime(tmp_path / "EUR.parquet", (three_days_ago, three_days_ago))
    calls = _serve(monkeypatch)

    history = md.load_history(["EUR"], cache_dir=tmp_path, today=TODAY, now=datetime.now())

    assert len(calls) == 2  # retried once
    status = history.status["EUR"]
    assert status.history_source == md.SOURCE_STALE_CACHE
    assert status.is_stale and status.stale_cache_age > md.CACHE_TTL
    assert history.var_currencies == ["EUR"]  # cached history still feeds VaR
    assert any("stale data" in w and "3.0 days old" in w for w in history.warnings())


def test_outage_and_no_cache_means_no_market_data(tmp_path, monkeypatch):
    _serve(monkeypatch)

    history = md.load_history(["EUR"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert history.status["EUR"].no_market_data
    assert history.var_currencies == []
    assert any("no market data" in w for w in history.warnings())


def test_unmapped_currency_is_no_market_data_without_a_fetch(tmp_path):
    history = md.load_history(["DKK"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert history.status["DKK"].no_market_data
    assert "no yfinance ticker mapped" in history.status["DKK"].error


def test_thin_history_flagged_low_confidence(tmp_path, monkeypatch):
    _serve(monkeypatch, history={"EURUSD=X": _series(30)})

    history = md.load_history(["EUR"], cache_dir=tmp_path, today=TODAY, now=NOW)

    status = history.status["EUR"]
    assert status.has_history and status.low_confidence
    assert status.n_return_obs == 29
    assert history.var_currencies == ["EUR"]  # still gets a VaR figure


def test_lagging_currency_and_common_as_of(tmp_path, monkeypatch):
    eur = _series(end=TODAY - timedelta(days=1))
    gbp = _series(end=TODAY - timedelta(days=4), start=1.27, seed=1)
    _serve(monkeypatch, history={"EURUSD=X": eur, "GBPUSD=X": gbp})

    history = md.load_history(["EUR", "GBP"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert history.lagging == ["GBP"]
    assert history.common_as_of == gbp.index[-1].date()
    assert history.returns().index[-1].date() == history.common_as_of
    assert any("GBP: last close" in w for w in history.warnings())


# --- current quotes ---------------------------------------------------------


def test_current_quote_is_live_yfinance_with_previous_close(tmp_path, monkeypatch):
    calls = _serve(monkeypatch, current={"EURUSD=X": _closes([1.1700, 1.1725])})

    quotes, errors = md.load_current_quotes(["EUR", "USD"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert calls[0][2]["period"] == "5d"
    eur = quotes["EUR"]
    assert eur.source == md.SOURCE_YFINANCE and eur.indicative
    assert eur.label == "indicative, not executable"
    assert eur.value == 1.1725 and eur.as_of == TODAY
    assert eur.previous_value == 1.1700 and eur.previous_as_of == date(2026, 10, 8)
    assert not eur.stale and errors == {}


def test_current_quote_inverted_ticker(tmp_path, monkeypatch):
    _serve(monkeypatch, current={"JPY=X": _closes([149.0, 150.0])})

    quotes, _ = md.load_current_quotes(["JPY"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert quotes["JPY"].value == pytest.approx(1 / 150.0)
    assert quotes["JPY"].previous_value == pytest.approx(1 / 149.0)


def test_current_outage_falls_back_to_cached_close_labelled_stale_with_age(tmp_path, monkeypatch):
    cached = _series(end=date(2026, 10, 8))
    md._write_cache(tmp_path, "EUR", cached)
    two_hours_ago = (NOW - timedelta(hours=2)).timestamp()
    os.utime(tmp_path / "EUR.parquet", (two_hours_ago, two_hours_ago))
    calls = _serve(monkeypatch)

    quotes, errors = md.load_current_quotes(["EUR"], cache_dir=tmp_path, today=TODAY, now=NOW)
    data = md.MarketData(
        history=md.HistoryData(pd.DataFrame(), {}, None, []), current=quotes, current_errors=errors
    )

    assert len(calls) == 2  # retried once
    eur = quotes["EUR"]
    assert eur.source == md.SOURCE_STALE_CACHE
    assert eur.value == pytest.approx(cached.iloc[-1]) and eur.previous_value == pytest.approx(cached.iloc[-2])
    assert eur.stale and eur.label == "indicative, not executable, stale"
    assert eur.cache_age == timedelta(hours=2)
    assert any("cached close" in w and "cache 2h old" in w for w in data.warnings())


def test_current_outage_and_no_cache_means_no_current_price(tmp_path, monkeypatch):
    _serve(monkeypatch)

    quotes, errors = md.load_current_quotes(["EUR"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert "EUR" not in quotes and "EUR" in errors


def test_stale_live_quote_is_flagged(tmp_path, monkeypatch):
    _serve(monkeypatch, current={"EURUSD=X": _closes([1.17, 1.18], end=date(2026, 10, 1))})

    data = md.load_market_data(["EUR"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert data.current["EUR"].stale
    assert any("yfinance quote as of 2026-10-01 is stale" in w for w in data.warnings())


def test_usd_current_quote_is_unity(tmp_path):
    quotes, errors = md.load_current_quotes(["USD"], cache_dir=tmp_path, today=TODAY, now=NOW)

    usd = quotes["USD"]
    assert usd.value == 1.0 and usd.previous_value == 1.0 and not usd.stale and errors == {}


def test_unmapped_currency_has_no_current_price(tmp_path):
    quotes, errors = md.load_current_quotes(["DKK"], cache_dir=tmp_path, today=TODAY, now=NOW)

    assert "DKK" not in quotes
    assert "no yfinance ticker mapped" in errors["DKK"]


def test_history_and_current_are_independent_fetches(tmp_path, monkeypatch):
    md._write_cache(tmp_path, "EUR", _series())  # fresh history cache
    calls = _serve(monkeypatch, current={"EURUSD=X": _closes([1.17, 1.18])})

    data = md.load_market_data(["EUR"], cache_dir=tmp_path, today=TODAY, now=datetime.now())

    assert [kind for kind, _, _ in calls] == ["current"]  # history served from cache, quote still live
    assert data.history.status["EUR"].history_source == md.SOURCE_CACHED
    assert data.current["EUR"].source == md.SOURCE_YFINANCE


def test_default_cache_dir_resolves_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "DEFAULT_CACHE_DIR", tmp_path)
    _serve(monkeypatch, history={"EURUSD=X": _series()})

    md.load_history(["EUR"], today=TODAY, now=NOW)

    assert (tmp_path / "EUR.parquet").exists()
