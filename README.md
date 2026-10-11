# FX Portfolio Dashboard

A Streamlit dashboard for an FX portfolio manager to monitor spot FX P&L and risk:
inception and daily USD P&L, parametric portfolio/standalone/component VaR, currency
exposure, and the data-quality state behind all of it, computed off one uploaded or
bundled CSV and live yfinance prices.

**Live link:** _add the deployed Streamlit Community Cloud URL here before submitting._

## Contents

- [Quick start](#quick-start)
- [Methodology](#methodology)
- [Major assumptions](#major-assumptions)
- [Limitations & known issues](#limitations--known-issues)
- [What we'd improve with more time](#what-wed-improve-with-more-time)
- [Testing](#testing)
- [Repository structure](#repository-structure)
- [Deployment](#deployment)

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # opens the dashboard on the bundled sample portfolio
pytest                        # 233 tests, no network access required
```

No API key or secret is required. The app reads `data/sample_portfolio.csv` by
default; the sidebar's "Upload a portfolio CSV" control swaps in any file with the
same six required columns without restarting the app.

## Methodology

### P&L

One formula for every pair, major or cross:

```
P_i,t = s_i * N_i * (U_A,t - S_i,0 * U_B,t)
```

where `U_C,t` is currency `C`'s canonical USD value (`U_USD ≡ 1`), `N_i` is the
trade's notional in **base-currency units**, `s_i` is +1 for BUY / −1 for SELL, and
`S_i,0` is the entry rate. Every currency is converted to `U_C` once, in
`market_data.py`; the P&L and VaR formulas never branch on whether a pair is a
major, a USD-base pair, or a cross. Inception P&L is `P_i,t` itself (zero at entry,
by construction); daily P&L is `P_i,t − P_i,t−1`.

Two things that aren't 0 or a crash by design:
- **A trade with no prior close** (same-day, or first mark after execution) has its
  previous mark treated as exactly 0 — the entry price stands in for the missing
  prior close — so its daily P&L equals its inception P&L on that first mark.
- **An existing position whose current quote is missing its previous close** (a
  genuine data gap, not a new trade) gets a **blank** daily P&L, not zero. Blank
  positions are excluded from the portfolio daily-P&L sum (not propagated as NaN,
  which would blank the whole total, and not silently zeroed, which would hide the
  gap) — the total carries a footnote naming which trades were excluded if any, and why.

### Historical P&L Curve

The "Historical P&L" section replays the portfolio's *current* holdings against
*past* prices, using the exact same `position_value` formula as the point-in-time
P&L above — it is not a second model. For each trade, `U_A,t` and `U_B,t` come from
`market_data.HistoryData.levels` (the same 800-day history the VaR engine uses)
instead of just today's quote, giving `P_i,t` at every past settled date; the
portfolio curve is the sum across trades, with each trade contributing exactly 0
before its own `trade_date`. The daily bars are just that curve's day-over-day
diff. A trade missing market data entirely is excluded from the curve, not zeroed.

This is a replay, not a record: there's no stored history of what the portfolio's
holdings actually were on a past date, so a trade is assumed to have been held,
unchanged, from its `trade_date` to today (see **Major assumptions** below). The
curve also stops at the last fully-settled session — it does not include today's
still-live quote, which is why it can sit one point behind the headline Portfolio
summary metrics above it.

### VaR methodology

Parametric (delta-normal), 1-day horizon, 95% confidence, 252-trading-day lookback
by default — all three adjustable from the sidebar, recomputed on every change
(nothing VaR-related is cached):

```
sigma_p^2 = w' * Sigma * w            VaR_p = z * sigma_p
Standalone_i = z * sqrt(w_i' * Sigma * w_i)      (position i's own exposure, zero elsewhere)
Component_i  = z * (w_i' * Sigma * w) / sigma_p   (i's exact share of VaR_p)
Marginal_i   = Component_i / x_i,  x_i = signed notional
```

- **z** comes from `scipy.stats.norm.ppf(confidence)` directly, the sidebar
  only offers 90/95/99%, with the value itself computed live.
- **Sigma** is the sample covariance of each currency's daily log return (vs USD)
  over the lookback window, on dates common to every currency in the portfolio
  (inner join, enforced inside `covariance_matrix` itself, not assumed of the
  caller). It's regularised with a small ridge **only if** it fails a
  positive-semidefinite (PSD) check — a true sample covariance is always PSD, so this
  is a rarely-triggered safety net, not an unconditional perturbation of real data.
- **Component VaR sums to portfolio VaR exactly** (Euler's theorem, since VaR is
  homogeneous of degree one in exposure) — tested to 1e-6 at every lookback window
  from 60 to 500 days, and both reconciliation captions on the dashboard state the
  sum next to the total so this is checkable on screen, not just asserted.
- **Confidence and horizon rescale every $ VaR figure by the same factor**
  (`z × √horizon`), so a position's **%** share of portfolio VaR is mathematically
  unchanged by them — that ratio only depends on the correlation structure, which
  only the **lookback window** changes. (Verified empirically: toggling 95%→99%
  moved portfolio VaR by exactly the ratio of the two z-scores while every
  percentage stayed byte-for-byte identical; changing the lookback window moved
  both.) This is noted in the sidebar so it reads as intentional, not broken.

**Two granularities, both exact, used for different things:**

| | Trade level | Pair level |
|---|---|---|
| Standalone VaR | Each trade's own gross exposure | Each pair's **net** exposure (all its trades' exposure vectors summed first, quadratic form applied once) |
| Component VaR | `var_concentration()` | `pair_summary()`, an exact sum of the pair's trades' component VaR |
| Used for | Trade-level detail table | Positions-by-pair table, the component-VaR chart, diversification benefit |

Standalone VaR is **never** obtained by summing a pair's trades' own standalone
VaR — doing so double-counts the netting between two offsetting trades on the same
pair as if it were real diversification. The headline **Diversification Benefit**
metric is `sum(pair-level standalone VaR) − portfolio VaR`. The trade-level
version still exists in the code — it backs the trade-level subadditivity
invariant test — it's just not what the dashboard displays as the headline figure.

### Currency exposure

Net USD exposure per currency, USD included (unlike the VaR engine, which drops USD
since it's the zero-variance numeraire): each trade's base leg is `s_i·N_i·U_A,t`
(current price on the trade's notional) and its term leg is
`−s_i·N_i·S_i,0·U_B,t` (today's price applied to the fixed quantity of term
currency the trade transacted at entry — the *quantity* is fixed since execution;
the *USD value* is marked to market today). Both current USD prices and the
portfolio's actual current holdings, in other words — MTM exposure.

### Market data

yfinance (`Ticker.history`) is the sole source, for both the live current quote and
the 800-calendar-day history behind VaR. Every raw quote is converted once into a
canonical per-currency value `U_C` (direct for EUR/GBP/AUD/NZD, inverted for
JPY/CHF/CAD and the rest — see the table below); nothing downstream ever consumes a
raw pair quote. Resilience is a local parquet cache, not a second provider: history
refreshes every 24h, current price every 60s, and a failed live fetch falls back to
the last cached value with a "stale" flag and the cache's age. Staleness itself is
checked against an **FX trading calendar** (weekends plus Dec 25/Jan 1 only) — not a
US bank-holiday calendar.

**Supported currencies** (anything else is rejected at the loader or flagged "no
market data" downstream due to yfinance ticker mapping, never silently mispriced):

| Direct quote (USD per unit) | Inverted quote (units per USD) |
|---|---|
| EUR, GBP, AUD, NZD | JPY, CHF, CAD, CNY, MXN, SGD, HKD, SEK, NOK, INR, ZAR, BRL, THB, IDR, MYR, PHP, HUF, ILS |

Any pair formed between USD and one of these, or between any two of them
(triangulated through USD — see P&L above), is supported. 22 currencies plus USD.

### Portfolio loading & validation

A CSV is read from an upload or the bundled sample through the exact same path:
`load_portfolio()` accepts a filesystem path or any readable file-like object
(so `src/` has no dependency on Streamlit). Validation is per the spec's edge-case
table — a bad row (negative notional, invalid side, malformed pair, duplicate
trade_id, unparseable date) is rejected individually with a reason, and the rest of
the file still loads; a structurally broken file (missing a required column, empty)
fails the whole load with one clear error.

## Major assumptions

- **Spot only.** No forwards, swaps, or other instruments — every position is a
  single spot trade with no maturity.
- **USD is the only reporting currency.** Every figure on the dashboard is in USD;
  there's no option to report in EUR or any other base.
- **`notional_base_ccy` is in the pair's base-currency units**, always a positive
  magnitude; sign comes from `side` (BUY = +1, SELL = −1), never from the number
  itself.
- **One set of VaR parameters for the whole portfolio.** Confidence, horizon and
  lookback are global sidebar controls, not configurable per position or per pair.
- **Equal-weighted covariance.** Every day in the lookback window counts the same.
- **A static book.** Positions don't amortize, roll, or rebalance — the quantity of
  each currency a trade holds is fixed from execution; only its USD value is marked
  to market.
- **Every row is a still-open position; nothing is ever closed.** The schema has
  no exit price or exit date, so every trade counts toward every calculation —
  current P&L, VaR, exposure, and the historical curve, which replays it across
  its *entire* life from `trade_date` to today — as if it's still live now. A
  trade that was in reality closed out last month has no way to say so.
- **Today's still-forming yfinance session is excluded from the VaR history** (so a
  live-vs-settled-bar inconsistency can't leak into the covariance matrix) but *is*
  used for the current price and daily P&L.

## Limitations & known issues

### Data-quality warnings you should expect to see

Running the bundled sample portfolio against live data currently surfaces warnings
like:

```
AUD: identical rate repeated for more than 2 business days on 1 occasion(s);
     most recent 3 days ending 2025-04-21
EUR: 1 daily return(s) beyond 5 standard deviations, most recent 2025-04-11
     (shown, not removed)
```

These are **genuine, correctly-firing signals, not bugs**: the repeated-rate flags
trace to a yfinance data gap around the April 2025 Good Friday/Easter weekend, and
the 5-sigma flags are real volatility events in that window. They fire on G10
majors just as much as anything more exotic, so there's no way to remove a
calendar artifact that hits every currency in the window equally. All such
warnings are collapsed in
a "Data-quality warnings" expander at the bottom of the page — visible, not buried,
but not competing with the headline numbers either.

### Market data

- **Single provider, no fallback beyond the local cache.** yfinance has no official
  API or SLA (Yahoo retired its own in 2017); if it's down *and* the local parquet
  cache is empty or stale, that currency is excluded from VaR and flagged "no
  market data" rather than silently mispriced — but there's no second live source
  to fail over to the way the original FRED+yfinance design had.
- **The local cache is ephemeral on most cloud hosts.** Streamlit Community Cloud
  doesn't guarantee local disk persists across restarts/redeploys, so the "runs
  even if the API is down" promise only holds once the cache has been warmed by at
  least one successful fetch after the most recent cold start.
- **CNY vs CNH**: the ticker mapped for the Chinese yuan is onshore CNY (`CNY=X`);
  yfinance's offshore CNH ticker had almost no usable history when checked. They
  trade at a small, real basis to each other — not a bug, just worth knowing if
  you're expecting offshore pricing.

### VaR methodology

- **Parametric (delta-normal) VaR assumes normally-distributed returns** and
  understates tail risk relative to historical simulation or Monte Carlo,
  especially around FX regime shifts (e.g. a sudden peg break).
- **Equal-weighted 252-day lookback reacts slowly** to a sudden rise in volatility,
  versus an EWMA-weighted alternative.
- **P&L is exact; VaR is linear.** A USD-base pair's P&L (e.g. USD/JPY) is mildly
  non-linear in the rate because the offsetting leg is fixed in the foreign
  currency, not USD; parametric VaR, by construction, is a linear (delta) model.
  The two are consistent, not identical, for a large move. Reconciling them exactly
  needs simulation-based VaR, which can potentially be an improvement given more time.
- **No VaR backtesting** in this version.
- **Cross-pair entry price and the triangulated rate don't exactly agree**, by
  design; a real cross trades its own bid/offer, so inception P&L on a cross is a
  small non-zero basis, not exactly 0. Expected, not suppressed.

### Scope not implemented

- Expected shortfall / CVaR, or any tail-risk measure beyond VaR.
- Closing or reducing a position — see "every row is a still-open position" above.

## What we'd improve with more time

### Additional analytics
- **Position lifecycle (closing/reducing a trade)** — the historical P&L curve and
  every other calculation currently assume a trade is still fully open today (see
  **Major assumptions**). An exit price/date on the schema would let the curve
  reflect a position actually being closed, rather than replaying it to the present
  regardless.
- **Per-pair or per-currency historical curves**, not just the portfolio total —
  the data (`market_data.HistoryData.levels`) already supports it; it's a
  presentation addition on top of `pnl.historical_portfolio_pnl`.
- **Expected shortfall (CVaR)** — the average loss *beyond* the VaR threshold, not
  just the threshold itself. It's a more informative tail-risk number than VaR alone.
  Natural to add alongside the existing parametric engine as a second column next
  to each standalone/component VaR figure.
- **Historical-simulation and Monte Carlo VaR** as togglable alternatives to the
  parametric model, directly addressing the normality assumption above.
- **EWMA/GARCH volatility weighting** as a lookback alternative to the current
  equal-weighted window.
- **VaR backtesting** to validate the 95%/99% coverage claims against realized P&L over time.
- **A correlation heatmap** across the portfolio's currencies — the covariance
  matrix is already computed; this is purely a presentation addition.
- **A VaR term-structure chart** (1-day through 10-day, say) now that horizon is
  already a first-class parameter.

### Front-end
Streamlit was the right call for this brief: it got a working, data-dense,
interactive dashboard to a deployed link fast, with no separate frontend build
step. For a tool meant for daily production use rather than a single review link,
two things would be worth revisiting:
- **A richer data grid** than `st.dataframe` — sortable/filterable columns, frozen
  headers, and inline cell formatting (so e.g. the pair table wouldn't need to
  pre-format every number to a string) are native to something like AG Grid, which
  means either a custom Streamlit component or stepping outside Streamlit to a
  decoupled API (FastAPI) + React frontend.
- **Shareable, bookmarkable views.** Sidebar selections (confidence, lookback, an
  uploaded portfolio) don't currently survive a page reload or show up in the URL;
  Streamlit's query-params API could carry the VaR parameters, though an uploaded
  file inherently can't round-trip through a URL.

A decoupled frontend would also remove Streamlit's full-script-rerun-per-interaction
model, which currently means every sidebar tweak re-walks the whole page.

### Engineering
- **CI** (GitHub Actions) running the 233-test suite on every push, rather than
  relying on a local `pytest` run before each push.
- **A persistent cache** (a small hosted KV store or database) instead of local
  parquet files, so the "survives an API outage" guarantee holds across Streamlit
  Cloud cold starts and redeploys, not just within one running instance.
- **Reintroducing a second data source** for the VaR history specifically (FRED or
  ECB), now that the single-source v1 has been live-tested — with the two-source
  edge cases (mixed-leg staleness, two calendars) under full test coverage from the
  start this time, rather than discovered mid-build.

## Testing

233 tests, `pytest`, no network access required (yfinance is mocked at the client
boundary in every test):

| File | Tests | Covers |
|---|---|---|
| `test_portfolio_loader.py` | 24 | Schema validation, rejection behavior, date-format fallbacks, file-like (upload) sources |
| `test_market_data.py` | 67 | Ticker mapping, FX trading calendar, caching/staleness, quote cleaning, outage fallback |
| `test_pnl.py` | 51 | The unified P&L formula, new-position/missing-previous-close handling, triangulation consistency, the historical P&L curve |
| `test_risk.py` | 47 | z-score, covariance regularisation, standalone/component/marginal VaR, Euler reconciliation, horizon scaling |
| `test_exposure.py` | 29 | Currency exposure, pair-level aggregation, diversification benefit, reconciliation totals |
| `test_app.py` | 15 | End-to-end page rendering, caching TTLs, file upload, section ordering, the historical P&L section |

Invariants checked directly (not just spot-checked): component VaR sums to
portfolio VaR to 1e-6 at every lookback from 60–500 days; portfolio VaR never
exceeds the sum of standalone VaRs; a single-position portfolio has identical
portfolio/standalone/component VaR; a flat net position contributes exactly zero;
10-day VaR equals 1-day VaR × √10; a pair's net standalone VaR is strictly less
than the sum of its trades' own standalone VaR whenever they partly offset; a new
trade's entry-day daily P&L (point-in-time or historical) equals its own inception
mark.

## Repository structure

```
app.py                          Streamlit entry point — presentation only
src/
├── data/
│   ├── portfolio_loader.py     CSV/upload parsing & validation
│   ├── market_data.py          yfinance client, caching, canonical U_C series
│   └── cache/                  local parquet cache (gitignored)
├── analytics/
│   ├── pnl.py                  the unified P&L formula + the historical curve
│   ├── risk.py                 parametric VaR (portfolio/standalone/component/marginal)
│   └── exposure.py             currency exposure, VaR concentration, pair-level views
└── models/
    └── schema.py                portfolio schema, Trade record, date formats
data/
└── sample_portfolio.csv        17 trades, 9 major pairs (incl. EUR/GBP, AUD/JPY crosses)
tests/                          233 tests, one file per src/ module plus test_app.py
conftest.py                     puts the repo root on sys.path for `from src...` imports
.streamlit/
├── config.toml                 (empty — no custom theme set)
└── secrets.toml                (empty — no secret is needed; yfinance requires no API key)
```

## Deployment

1. Push to a GitHub repository.
2. On [Streamlit Community Cloud](https://share.streamlit.io), create a new app
   pointing at this repo's `app.py` on the branch to deploy.
3. First load after a cold start fetches 800 days of history per currency live
   from yfinance (a few seconds); subsequent loads within 24h are served from the
   parquet cache written during that first load.
