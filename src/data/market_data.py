"""Market data layer, per the spec's 'Market data strategy' section.

yfinance (Ticker.history) is the single source for both the current quote and
the 800-day history behind VaR. Resilience comes from a local parquet cache:
a failed fetch falls back to the last cached series, labelled stale with its age.

Every currency C is converted once, here, into its canonical value
U_C = USD value of one unit of C (U_USD = 1); nothing downstream consumes a
raw pair quote.

load_history and load_current_quotes are separate so app.py can cache them with
different TTLs (24h and 60s). The parquet cache also enforces the 24h history
TTL on disk, so a restarted app doesn't refetch.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay


@dataclass(frozen=True)
class YfTicker:
    symbol: str
    inverted: bool  # quoted USD/XXX, so U_C = 1 / quote


YF_TICKERS: dict[str, YfTicker] = {
    # Raw quote is USD per unit of the foreign currency
    "EUR": YfTicker("EURUSD=X", inverted=False),
    "GBP": YfTicker("GBPUSD=X", inverted=False),
    "AUD": YfTicker("AUDUSD=X", inverted=False),
    "NZD": YfTicker("NZDUSD=X", inverted=False),

    # Raw quote is foreign-currency units per USD;
    # invert to obtain USD per unit of foreign currency
    "JPY": YfTicker("JPY=X", inverted=True),
    "CHF": YfTicker("CHF=X", inverted=True),
    "CAD": YfTicker("CAD=X", inverted=True),
    "CNY": YfTicker("CNY=X", inverted=True),
    "MXN": YfTicker("MXN=X", inverted=True),
    "SGD": YfTicker("SGD=X", inverted=True),
    "HKD": YfTicker("HKD=X", inverted=True),
    "SEK": YfTicker("SEK=X", inverted=True),
    "NOK": YfTicker("NOK=X", inverted=True),
    "INR": YfTicker("INR=X", inverted=True),
    "ZAR": YfTicker("ZAR=X", inverted=True),
    "BRL": YfTicker("BRL=X", inverted=True),
    "THB": YfTicker("THB=X", inverted=True),
    "IDR": YfTicker("IDR=X", inverted=True),
    "MYR": YfTicker("MYR=X", inverted=True),
    "PHP": YfTicker("PHP=X", inverted=True),
    "HUF": YfTicker("HUF=X", inverted=True),
    "ILS": YfTicker("ILS=X", inverted=True),
}


class FXMarketClosures(AbstractHolidayCalendar):
    """Full FX-market closures only. US bank holidays (Columbus Day etc.) are
    normal FX trading days, so they are deliberately absent."""

    rules = [
        Holiday("Christmas Day", month=12, day=25),
        Holiday("New Year's Day", month=1, day=1),
    ]


FX_BUSINESS_DAY = CustomBusinessDay(calendar=FXMarketClosures())

HISTORY_CALENDAR_DAYS = 800
CURRENT_PERIOD = "5d"
FFILL_LIMIT_BDAYS = 3
MAX_REPEAT_BDAYS = 2  # identical rate on more than this many consecutive business days is flagged
OUTLIER_SIGMA = 5.0
MIN_RETURN_OBS = 60
CACHE_TTL = timedelta(hours=24)
MAX_ATTEMPTS = 2  # initial call + one retry
RETRY_DELAY_S = 1.0
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"

SOURCE_YFINANCE = "yfinance"
SOURCE_CACHED = "yfinance (cached)"
SOURCE_STALE_CACHE = "yfinance (stale cache)"
SOURCE_USD_UNIT = "USD (unit)"
SOURCE_NONE = "none"

INDICATIVE_LABEL = "indicative, not executable"


class MarketDataError(Exception):
    """A market-data fetch failed; the message is safe to display."""


@dataclass(frozen=True)
class Quote:
    """A U_C value with its as-of date, plus the prior close for daily P&L."""

    value: float
    as_of: date
    previous_value: Optional[float]
    previous_as_of: Optional[date]
    source: str
    indicative: bool
    stale: bool
    cache_age: Optional[timedelta] = None  # set when served from the parquet cache after a failed fetch

    @property
    def label(self) -> str:
        parts = ([INDICATIVE_LABEL] if self.indicative else []) + (["stale"] if self.stale else [])
        return ", ".join(parts)


@dataclass
class CurrencyStatus:
    """Data-quality state of one currency's history series."""

    currency: str
    history_source: str
    as_of: Optional[date] = None
    n_return_obs: int = 0
    stale_cache_age: Optional[timedelta] = None
    repeated_rate_runs: list[tuple[date, date, int]] = field(default_factory=list)
    outlier_dates: list[date] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_history(self) -> bool:
        return self.as_of is not None

    @property
    def no_market_data(self) -> bool:
        """No usable history, so the currency is excluded from VaR."""
        return not self.has_history

    @property
    def low_confidence(self) -> bool:
        return self.has_history and self.n_return_obs < MIN_RETURN_OBS

    @property
    def is_stale(self) -> bool:
        return self.stale_cache_age is not None


@dataclass
class HistoryData:
    levels: pd.DataFrame  # U_C by business date, one column per currency, USD = 1.0
    status: dict[str, CurrencyStatus]  # non-USD currencies only
    common_as_of: Optional[date]
    lagging: list[str]

    @property
    def var_currencies(self) -> list[str]:
        return sorted(c for c, s in self.status.items() if s.has_history)

    def returns(self, currencies: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return log_returns(self.levels, self.var_currencies if currencies is None else currencies)

    def warnings(self) -> list[str]:
        out: list[str] = []
        for ccy, s in self.status.items():
            if s.no_market_data:
                detail = f" ({s.error})" if s.error else ""
                out.append(f"{ccy}: no market data, excluded from VaR{detail}")
                continue
            if s.is_stale:
                out.append(
                    f"{ccy}: stale data, yfinance unreachable; using cached history "
                    f"{_format_age(s.stale_cache_age)} old"
                )
            if s.low_confidence:
                out.append(
                    f"{ccy}: low confidence, only {s.n_return_obs} return observations "
                    f"(minimum {MIN_RETURN_OBS})"
                )
            if s.repeated_rate_runs:
                _, end, length = s.repeated_rate_runs[-1]
                out.append(
                    f"{ccy}: identical rate repeated for more than {MAX_REPEAT_BDAYS} business days "
                    f"on {len(s.repeated_rate_runs)} occasion(s); most recent {length} days ending {end}"
                )
            if s.outlier_dates:
                out.append(
                    f"{ccy}: {len(s.outlier_dates)} daily return(s) beyond {OUTLIER_SIGMA:g} standard "
                    f"deviations, most recent {s.outlier_dates[-1]} (shown, not removed)"
                )
        for ccy in self.lagging:
            out.append(
                f"{ccy}: last close {self.status[ccy].as_of} lags other currencies; "
                f"portfolio VaR uses the common date {self.common_as_of}"
            )
        return out


@dataclass
class MarketData:
    history: HistoryData
    current: dict[str, Quote]
    current_errors: dict[str, str]  # why a live current quote could not be fetched

    def warnings(self) -> list[str]:
        out = self.history.warnings()
        for ccy, error in self.current_errors.items():
            quote = self.current.get(ccy)
            if quote is None:
                out.append(f"{ccy}: no current price available ({error})")
            else:
                out.append(
                    f"{ccy}: yfinance unavailable for the current price; showing the cached close "
                    f"as of {quote.as_of}, stale (cache {_format_age(quote.cache_age)} old)"
                )
        for ccy, quote in self.current.items():
            if quote.source == SOURCE_YFINANCE and quote.stale:
                out.append(f"{ccy}: yfinance quote as of {quote.as_of} is stale")
        return out


# --- yfinance client -------------------------------------------------------


def _yf_history(symbol: str, **kwargs) -> pd.Series:
    """Daily closes for one ticker, ascending, indexed by naive FX session date. Single attempt.

    yfinance can return a snapshot row dated on a weekend (e.g. Saturday) holding the
    final price of the preceding session, or repeat the latest session as a second
    row. Each row is mapped to its FX session (non-trading dates roll back to the
    previous FX trading day) and the latest value per session is kept."""
    try:
        hist = yf.Ticker(symbol).history(**kwargs)
    except Exception as exc:  # yfinance surfaces many unrelated exception types
        raise MarketDataError(f"yfinance {symbol}: {type(exc).__name__}: {exc}") from exc
    if "Close" not in hist:
        raise MarketDataError(f"yfinance {symbol}: no data returned")
    closes = hist["Close"].dropna().sort_index()
    closes.index = pd.DatetimeIndex(
        [FX_BUSINESS_DAY.rollback(pd.Timestamp(ts.date())) for ts in closes.index], name="date"
    )
    closes = closes[~closes.index.duplicated(keep="last")].rename(symbol)
    if closes.empty:
        raise MarketDataError(f"yfinance {symbol}: no data returned")
    return closes


def _fetch_with_retry(symbol: str, **kwargs) -> pd.Series:
    last_error: Optional[MarketDataError] = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            time.sleep(RETRY_DELAY_S)
        try:
            return _yf_history(symbol, **kwargs)
        except MarketDataError as exc:
            last_error = exc
    raise last_error


def fetch_history(ticker: YfTicker, today: date) -> pd.Series:
    """800 calendar days of completed daily closes in the ticker's own units.
    Today's still-forming session is excluded so it never enters the VaR series."""
    start = today - timedelta(days=HISTORY_CALENDAR_DAYS)
    closes = _fetch_with_retry(ticker.symbol, start=start.isoformat(), end=today.isoformat(), interval="1d")
    closes = closes[closes.index < pd.Timestamp(today)]
    if closes.empty:
        raise MarketDataError(f"yfinance {ticker.symbol}: no completed sessions returned")
    return closes


def fetch_current_quote(ticker: YfTicker, today: date) -> Quote:
    """Latest quote (today's live session when trading) plus the prior close."""
    closes = _fetch_with_retry(ticker.symbol, period=CURRENT_PERIOD, interval="1d")
    return _quote_from_series(to_canonical(closes, ticker.inverted), SOURCE_YFINANCE, today=today)


# --- cleaning & canonical series -------------------------------------------


def clean_quotes(raw: pd.Series) -> pd.Series:
    """Reindex to FX trading days (input is already keyed by FX session date),
    forward-fill gaps up to FFILL_LIMIT_BDAYS and drop anything still missing.
    Nothing is filled past the last observation, so the last index date is
    always a real close."""
    valid = raw.dropna()
    if valid.empty:
        return valid.astype(float)
    calendar = pd.date_range(valid.index[0], valid.index[-1], freq=FX_BUSINESS_DAY, name="date")
    return raw.reindex(calendar).ffill(limit=FFILL_LIMIT_BDAYS).dropna()


def to_canonical(quotes: pd.Series, inverted: bool) -> pd.Series:
    return 1.0 / quotes if inverted else quotes.copy()


def last_business_day(today: date) -> date:
    """The most recent FX trading day strictly before today: weekends and
    Dec 25 / Jan 1 skipped, US bank holidays not."""
    return (pd.Timestamp(today) - FX_BUSINESS_DAY).date()


def repeated_rate_runs(quotes: pd.Series, max_repeat: int = MAX_REPEAT_BDAYS) -> list[tuple[date, date, int]]:
    if quotes.empty:
        return []
    run_id = (quotes != quotes.shift()).cumsum()
    runs = []
    for _, run in quotes.groupby(run_id):
        if len(run) > max_repeat:
            runs.append((run.index[0].date(), run.index[-1].date(), len(run)))
    return runs


def outlier_dates(returns: pd.Series, n_sigma: float = OUTLIER_SIGMA) -> list[date]:
    if len(returns) < 2:
        return []
    sigma = returns.std()
    if not sigma > 0:
        return []
    return [d.date() for d in returns.index[returns.abs() > n_sigma * sigma]]


def log_returns(levels: pd.DataFrame, currencies: Iterable[str]) -> pd.DataFrame:
    """Daily log returns of U_C on dates common to every requested currency (inner join)."""
    aligned = levels[list(currencies)].dropna(how="any")
    return np.log(aligned).diff().dropna(how="any")


def _quote_from_series(
    u: pd.Series,
    source: str,
    *,
    today: date,
    force_stale: bool = False,
    cache_age: Optional[timedelta] = None,
) -> Quote:
    as_of = u.index[-1].date()
    has_previous = len(u) >= 2
    return Quote(
        value=float(u.iloc[-1]),
        as_of=as_of,
        previous_value=float(u.iloc[-2]) if has_previous else None,
        previous_as_of=u.index[-2].date() if has_previous else None,
        source=source,
        indicative=True,
        stale=force_stale or as_of < last_business_day(today),
        cache_age=cache_age,
    )


# --- parquet cache ---------------------------------------------------------


@dataclass(frozen=True)
class _CachedSeries:
    series: pd.Series
    age: timedelta


def _cache_path(cache_dir: Path, currency: str) -> Path:
    return cache_dir / f"{currency}.parquet"


def _write_cache(cache_dir: Path, currency: str, raw: pd.Series) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, currency)
    tmp = path.with_suffix(".parquet.tmp")
    raw.rename("value").to_frame().to_parquet(tmp)
    os.replace(tmp, path)


def _read_cache(cache_dir: Path, currency: str, now: datetime) -> Optional[_CachedSeries]:
    path = _cache_path(cache_dir, currency)
    if not path.exists():
        return None
    series = pd.read_parquet(path)["value"].rename(currency)
    age = now - datetime.fromtimestamp(path.stat().st_mtime)
    return _CachedSeries(series=series, age=age)


def _resolve_cache_dir(cache_dir: Optional[Path | str]) -> Path:
    return Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR


# --- history ---------------------------------------------------------------


def _load_raw_history(
    currency: str, ticker: YfTicker, cache_dir: Path, today: date, now: datetime
) -> tuple[Optional[pd.Series], str, Optional[timedelta], Optional[str]]:
    cached = _read_cache(cache_dir, currency, now)
    if cached is not None and cached.age < CACHE_TTL:
        return cached.series, SOURCE_CACHED, None, None

    try:
        raw = fetch_history(ticker, today)
    except MarketDataError as exc:
        error = str(exc)
    else:
        _write_cache(cache_dir, currency, raw)
        return raw, SOURCE_YFINANCE, None, None

    if cached is not None:
        return cached.series, SOURCE_STALE_CACHE, cached.age, error
    return None, SOURCE_NONE, None, error


def _load_currency_history(
    currency: str, cache_dir: Path, today: date, now: datetime
) -> tuple[Optional[pd.Series], CurrencyStatus]:
    ticker = YF_TICKERS.get(currency)
    if ticker is None:
        return None, CurrencyStatus(
            currency, history_source=SOURCE_NONE, error=f"no yfinance ticker mapped for {currency}"
        )

    raw, source, stale_age, error = _load_raw_history(currency, ticker, cache_dir, today, now)
    if raw is None:
        return None, CurrencyStatus(currency, history_source=source, error=error)

    quotes = clean_quotes(raw)
    u = to_canonical(quotes, ticker.inverted).rename(currency)
    returns = np.log(u).diff().dropna()
    status = CurrencyStatus(
        currency=currency,
        history_source=source,
        as_of=u.index[-1].date(),
        n_return_obs=len(returns),
        stale_cache_age=stale_age,
        repeated_rate_runs=repeated_rate_runs(quotes),
        outlier_dates=outlier_dates(returns),
        error=error,
    )
    return u, status


def load_history(
    currencies: Iterable[str],
    *,
    cache_dir: Optional[Path | str] = None,
    today: Optional[date] = None,
    now: Optional[datetime] = None,
) -> HistoryData:
    now = now or datetime.now()
    today = today or now.date()
    cache_dir = _resolve_cache_dir(cache_dir)

    levels_by_ccy: dict[str, pd.Series] = {}
    status: dict[str, CurrencyStatus] = {}
    for ccy in sorted(set(currencies) - {"USD"}):
        u, ccy_status = _load_currency_history(ccy, cache_dir, today, now)
        status[ccy] = ccy_status
        if u is not None:
            levels_by_ccy[ccy] = u

    if levels_by_ccy:
        levels = pd.concat(levels_by_ccy, axis=1, sort=True)
    else:
        levels = pd.DataFrame(index=pd.DatetimeIndex([]))
    levels["USD"] = 1.0
    levels.index.name = "date"

    as_ofs = {c: s.as_of for c, s in status.items() if s.as_of is not None}
    common_as_of = min(as_ofs.values()) if as_ofs else None
    latest = max(as_ofs.values()) if as_ofs else None
    lagging = sorted(c for c, d in as_ofs.items() if d < latest)

    return HistoryData(levels=levels, status=status, common_as_of=common_as_of, lagging=lagging)


# --- current quotes --------------------------------------------------------


def _usd_quote(today: date) -> Quote:
    return Quote(
        value=1.0,
        as_of=today,
        previous_value=1.0,
        previous_as_of=last_business_day(today),
        source=SOURCE_USD_UNIT,
        indicative=False,
        stale=False,
    )


def load_current_quotes(
    currencies: Iterable[str],
    *,
    cache_dir: Optional[Path | str] = None,
    today: Optional[date] = None,
    now: Optional[datetime] = None,
) -> tuple[dict[str, Quote], dict[str, str]]:
    """Current U_C per currency from yfinance. On a failed fetch, falls back to the
    last close in the parquet cache, labelled stale with the cache's age. Returns
    (quotes, errors); errors explains every currency without a live quote."""
    now = now or datetime.now()
    today = today or now.date()
    cache_dir = _resolve_cache_dir(cache_dir)

    quotes: dict[str, Quote] = {}
    errors: dict[str, str] = {}
    for ccy in sorted(set(currencies)):
        if ccy == "USD":
            quotes[ccy] = _usd_quote(today)
            continue
        ticker = YF_TICKERS.get(ccy)
        if ticker is None:
            errors[ccy] = f"no yfinance ticker mapped for {ccy}"
            continue
        try:
            quotes[ccy] = fetch_current_quote(ticker, today)
        except MarketDataError as exc:
            errors[ccy] = str(exc)
            cached = _read_cache(cache_dir, ccy, now)
            if cached is not None and not cached.series.dropna().empty:
                u = to_canonical(cached.series.dropna(), ticker.inverted)
                quotes[ccy] = _quote_from_series(
                    u, SOURCE_STALE_CACHE, today=today, force_stale=True, cache_age=cached.age
                )
    return quotes, errors


def load_market_data(
    currencies: Iterable[str],
    *,
    cache_dir: Optional[Path | str] = None,
    today: Optional[date] = None,
    now: Optional[datetime] = None,
) -> MarketData:
    currencies = set(currencies)
    now = now or datetime.now()
    today = today or now.date()
    history = load_history(currencies, cache_dir=cache_dir, today=today, now=now)
    current, errors = load_current_quotes(currencies, cache_dir=cache_dir, today=today, now=now)
    return MarketData(history=history, current=current, current_errors=errors)


def _format_age(age: Optional[timedelta]) -> str:
    if age is None:
        return "unknown age"
    hours = age.total_seconds() / 3600
    return f"{hours:.0f}h" if hours < 48 else f"{hours / 24:.1f} days"
