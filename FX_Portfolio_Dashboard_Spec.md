# FX Portfolio Dashboard — Implementation Specification

Oct 10, 2026 · @Jiawei Lee

Python + Streamlit, yfinance as the single market-data source, and a 1-day 95% parametric covariance VaR are confirmed as the right choices for this brief; every ambiguity below is resolved with an explicit default before any code is written.

## Executive assessment & key recommendations

The brief's own suggested stack and methodology both hold up; the table below is the full set of decisions needed before coding starts.

| Decision | Choice | Rationale |
| --- | --- | --- |
| Language & framework | Python 3.11 + Streamlit | Fastest path to an interactive dashboard inside 24 hours; no reason to deviate |
| Portfolio storage | CSV (primary), optional SQLite mirror | Brief asks for a supplied sample file; CSV is easiest to inspect, diff and version |
| Market data | yfinance, for both current price and the historical/VaR series | One source removes the whole class of bugs a two-source design creates (mixed-leg staleness, cross-calendar mismatches); resilience comes from the local parquet cache, not a second live provider. Trade-off disclosed in Limitations: this drops the "official government source" story FRED offered for the VaR history |
| VaR methodology | 1-day, 95% parametric (delta-normal) covariance VaR, 252-day lookback | Matches the brief; explainable, fast to build and test in the time available |
| Deployment target | Streamlit Community Cloud | Free, public URL, deploys straight from GitHub, fits the 24-hour window |

One inconsistency is worth flagging now rather than discovering it mid-build: exact USD P&L on a USD/XXX pair (e.g. USD/JPY) is mildly non-linear in the exchange rate, because the offsetting foreign-currency leg is fixed in that foreign currency, not in USD. Parametric VaR, by construction, is a linear (delta) model. The resolution used throughout is to keep P&L exact and let VaR stay linear — this is standard industry practice, not an error, and is disclosed as a limitation rather than papered over.

## Data flow & repository structure

&#91;embedded content: data flow · portfolio load + market data (primary/fallback) → P&L/VaR engine → dashboard\]

yfinance is the single source for both current price and the historical/VaR series; it falls back to the local parquet cache on an outage, with a stale-data banner showing the cache's age.

### Repository structure

```
fx-portfolio-dashboard/
├── app.py
├── src/
│   ├── data/
│   │   ├── portfolio_loader.py
│   │   ├── market_data.py
│   │   └── cache/
│   ├── analytics/
│   │   ├── pnl.py
│   │   ├── risk.py
│   │   ├── stress.py
│   │   └── exposure.py
│   ├── models/
│   │   └── schema.py
│   └── utils/
│       ├── config.py
│       └── validation.py
├── data/
│   └── sample_portfolio.csv
├── tests/
│   ├── test_pnl.py
│   ├── test_risk.py
│   └── test_market_data.py
├── README.md
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Requirements checklist

Every must-have maps directly to a line in the brief; should-haves are genuinely useful additions that fit the time budget; could-haves are named so they are not silently forgotten.

| Priority | Feature |
| --- | --- |
| Must-have | Load portfolio from CSV/SQLite with a supplied sample file |
| Must-have | Current and historical FX reference prices from a real data source |
| Must-have | Position- and portfolio-level USD P&L, daily and since inception |
| Must-have | Position (standalone) VaR, portfolio VaR, marginal VaR, component VaR |
| Must-have | README covering methodology, assumptions, limitations, next steps |
| Must-have | Working deployed link, complete source code |
| Must-have | Cross-pair pricing and risk, via triangulation off two USD legs (e.g. EUR/GBP from EUR/USD and GBP/USD) |
| Should-have | Currency exposure breakdown (net USD exposure by currency) |
| Should-have | Risk concentration view (% of portfolio VaR by position/pair) |
| Should-have | Stress scenarios (e.g. ±1%/±2%/±5% shocks per pair) |
| Should-have | Component VaR contribution chart |
| Should-have | Data-quality warnings (stale data, thin history, rejected rows) |
| Should-have | Explicit reference-rate vs executable-quote labelling throughout the UI |
| Should-have | VaR parameters (confidence level, horizon, lookback window) exposed as sidebar controls, defaulting to 95% / 1-day / 252-day |
| Could-have | EWMA/GARCH volatility, or a historical-simulation/Monte Carlo VaR toggle |
| Could-have | VaR backtesting (Kupiec exception count) |
| Could-have | Correlation heatmap across pairs |
| Could-have | Historical portfolio P&L / equity curve |
| Could-have | FX forwards or other instruments |
| Could-have | Excel/PDF export of the risk report |
| Could-have | PM-configurable VaR limits with breach alerts |

## Portfolio data schema (CSV)

The portfolio loads from one CSV with eleven columns; six are required, five are optional metadata. Follow the task guidelines for supplied file data's required fields at the very least, as this will be used when tested.

| Column | Type | Required | Description |
| --- | --- | --- | --- |
| trade_id | string | Yes | Unique trade identifier, e.g. T0001 |
| trade_date | date (YYYY-MM-DD) | Yes | Execution date of the spot trade |
| currency_pair | string, BASE/TERM | Yes | Market-convention pair, e.g. EUR/USD, USD/JPY |
| side | enum BUY / SELL | Yes | BUY = long base currency, SELL = short base currency |
| notional_base_ccy | float, > 0 | Yes | Trade size in base-currency units, always a positive magnitude |
| entry_price | float, > 0 | Yes | BASE/TERM rate at execution |
| settlement_date | date | No | Informational only; spot convention = trade_date + 2 business days |
| counterparty | string | No | For credit/concentration grouping |
| book | string | No | Strategy or book tag for grouping |
| trader | string | No | For attribution |
| notes | string | No | Free text |

Sign convention: signed exposure = +notional_base_ccy if side is BUY, −notional_base_ccy if side is SELL, always in base-currency units. Cross pairs (neither leg USD, e.g. EUR/GBP) are priced via triangulation off their two USD legs rather than treated as a third price type — see Definitions & Formulas.

```csv
trade_id,trade_date,currency_pair,side,notional_base_ccy,entry_price,settlement_date,counterparty,book,trader,notes
T0001,2026-07-14,EUR/USD,BUY,5000000,1.0865,2026-07-16,Bank A,G10 Spot,J.Lee,
T0002,2026-08-01,GBP/USD,SELL,2000000,1.2740,2026-08-05,Bank B,G10 Spot,J.Lee,
T0003,2026-08-20,USD/JPY,BUY,4000000,147.35,2026-08-24,Bank A,G10 Spot,M.Oni,
T0004,2026-09-02,USD/CHF,SELL,1500000,0.8012,2026-09-04,Bank C,G10 Spot,M.Oni,
T0005,2026-09-18,AUD/USD,BUY,3000000,0.6610,2026-09-22,Bank B,G10 Spot,J.Lee,
T0006,2026-10-01,USD/CAD,SELL,2500000,1.3885,2026-10-05,Bank A,G10 Spot,M.Oni,
```

## Definitions & formulas

Six formulas drive the dashboard: two for P&L and four for risk, all expressed in USD from each position's signed exposure and its pair's return series.

### Sign convention and the canonical currency value

Every currency C gets one canonical series, U\_{C,t}: the USD value of one unit of C on date t, with U\_{USD,t} ≡ 1. For a currency FRED quotes directly against USD (EUR, GBP, AUD, NZD), U\_C is that quote; for a currency FRED quotes as USD-per-unit-of-C's inverse (JPY, CHF, CAD, CNH, MXN — the "USD/XXX" convention), U\_C is 1 divided by that quote. This inversion happens once per currency, in the market-data layer; the P&L and risk formulas below never branch on it.

### P&L (one formula, every pair)

Position i is on pair A/B (A = base currency, B = quote currency), with base notional N\_i > 0, s\_i = +1 for a BUY (long base) and −1 for a SELL, and entry rate S\_{i,0}:

```latex
P_{i,t} = s_i N_i \left(U_{A,t} - S_{i,0} U_{B,t}\right)
```

```latex
PnL_{inception} = P_{i,t}, \qquad PnL_{daily} = P_{i,t} - P_{i,t-1}
```

This one formula covers EUR/USD, USD/JPY and a cross such as EUR/GBP alike — a major is just the special case where A or B is USD itself, so U\_A or U\_B collapses to 1. P\_{i,0} = 0 by construction, since S\_{i,0} = U\_{A,0}/U\_{B,0} at trade entry.

Critical trap: N\_i(S\_t - S\_{i,0}) is P&L in the quote currency B, not USD, unless B is USD. For EUR/GBP that amount is in GBP and still needs converting via U\_{GBP,t}; the formula above avoids the bug by working in USD-per-currency terms from the start, rather than computing a quote-currency P&L and converting it afterward. A newly opened trade's first daily mark compares against P\_{i,t} = 0 at entry — entry\_price is standing in for the missing P\_{i,t-1}, since nothing else exists yet to play that role — never an artificial pre-trade holding; this is why inception and daily P&L are identical on a trade's own trade date and diverge from day two on, once an actual prior close exists. weekend and holiday alignment is handled once in the Market Data Strategy section, not per position.

### USD exposure (the risk sensitivity)

Every position decomposes into at most two currency legs, used directly as the w vector feeding the VaR formulas below:

```latex
w^{(A)}_i = s_i N_i U_{A,t}, \qquad w^{(B)}_i = -s_i N_i S_{i,0} U_{B,t}
```

A major position's USD-side leg has zero variance by construction (U\_{USD} ≡ 1) and drops out of the risk calculation on its own — no special-casing needed. A cross position keeps both non-USD legs active, and only prices if both A and B already have their own canonical U\_C series; a currency with none at all still can't be triangulated.

### VaR

Portfolio variance and 1-day, 95% portfolio VaR, where w is the vector of net USD exposure per currency and Σ is the covariance matrix of each currency's daily log returns (vs USD) over the 252-day lookback:

```latex
\sigma_p^2 = w^T \Sigma w, \qquad VaR_p = z_{0.95} \times \sqrt{w^T \Sigma w} = 1.645 \times \sqrt{w^T \Sigma w}
```

Standalone position VaR (no diversification credit), using σ\_i, the standard deviation of pair i's own return series:

```latex
VaR_i^{standalone} = z_{0.95} \times |w_i| \times \sigma_i
```

A cross position (two currency legs) uses the same quadratic form restricted to its own exposure vector w\_i (zero elsewhere), which collapses to the single-currency formula above when a position has only one leg:

```latex
VaR_i^{standalone} = z_{0.95} \times \sqrt{w_i^T \Sigma w_i}
```

Marginal VaR (the portfolio VaR's sensitivity to a one-unit change in position i's signed exposure) and component VaR (marginal VaR scaled back to the position's actual size):

```latex
MVaR_i = z_{0.95} \times \frac{(\Sigma w)_i}{\sigma_p}, \qquad CVaR_i = w_i \times MVaR_i
```

Component VaRs sum exactly to portfolio VaR by Euler's theorem (VaR is homogeneous of degree one in exposure), so the position-level breakdown always reconciles to the headline number:

```latex
\sum_i CVaR_i = VaR_p, \qquad DB = \sum_i VaR_i^{standalone} - VaR_p
```

DB is the diversification benefit shown alongside the headline VaR. Standalone, marginal and component VaR are computed at position level (not just pair level) so a PM can see which individual trade, not just which currency, drives the risk; positions sharing a pair are perfectly correlated with each other by construction. A worked 2-asset numerical example is used directly as the unit-test fixture in the Test Plan section.

### Verification

Euler's theorem holds exactly here, not approximately: since VaR\_p = z×√(w'Σw) is homogeneous of degree one in w, ∑CVaR\_i = z×(w'Σw)/σ\_p = z×σ\_p = VaR\_p is an algebraic identity, confirmed by direct differentiation above. The unified P&L formula was checked numerically on a toy EUR/GBP cross (U\_EUR 1.0830→1.0850, U\_GBP 1.2710→1.2700, N=1,000,000): P\_{i,t} gave $2,852, matching the independent two-leg sum ($2,000 EUR leg + $852 GBP leg) to the cent. The z-scores in the Market Data Strategy table are the standard one-tailed normal quantiles (90/95/99%), cross-checked against scipy.stats.norm.ppf.

## Market data strategy

yfinance is the single source for the headline current price, daily P&L and the VaR history, after a live test showed FRED's own lag was far worse than first assumed.

| Aspect | Approach |
| --- | --- |
| Rate convention | BASE/TERM per market convention, matching each yfinance ticker's own units exactly (no re-inversion) |
| yfinance ticker mapping | EURUSD=X, GBPUSD=X, AUDUSD=X, NZDUSD=X (EUR/GBP/AUD/NZD, quoted USD-per-unit), JPY=X (USD/JPY), CHF=X (USD/CHF), CAD=X (USD/CAD), CNH=X (USD/CNH), MXN=X (USD/MXN) |
| "Current price" definition | The latest yfinance quote, with its as-of timestamp shown beside every price; labelled "indicative, not executable" everywhere it appears, which is how the brief's reference-rate-vs-live-quote distinction is satisfied with a single source |
| Historical coverage | 800 calendar days (\~3 years) requested per currency up front, so the sidebar's lookback-window control (Requirements Checklist) can range from 60 to 500 trading days without ever running short; the 252-day default still only consumes the most recent slice of it |
| Missing dates | Weekends and the rare full FX-market closure (Dec 25, Jan 1) are simply absent from yfinance's response rather than marked NaN; forward-filled up to 3 consecutive business days, otherwise that date is dropped from the pair's return series |
| Staleness-check calendar | "Is this quote stale?" compares against an FX trading calendar (weekends and Dec 25/Jan 1 only), not a US bank-holiday calendar — FX trades through Presidents' Day, Columbus Day, Veterans Day etc. even when US banks are closed, so a bank-holiday calendar would falsely flag live quotes as stale on those days |
| Cross-pair alignment | Returns are computed only on dates common to every pair in the portfolio (inner join), so the covariance matrix is always well-defined |
| Thin history | A pair with fewer than 60 valid return observations still gets a VaR figure, flagged "low confidence" rather than suppressed |
| Outages | A failed yfinance call falls back to the last successfully cached price or series (local parquet), labelled "stale" with the cache's age shown; there's no second live provider to fail over to, so resilience depends entirely on the cache staying fresh |
| Caching | Historical series: local parquet files, refreshed once a day (24h TTL via st.cache\_data), persisted to disk so the app still runs if the API is down during a demo. Current price: a separate, short TTL (60s via st.cache\_data) — long enough that Streamlit's rerun-on-every-interaction doesn't hammer yfinance on each sidebar click, short enough that "current" stays genuinely current. No intraday P&L series is stored; each page load recomputes a fresh point-in-time snapshot, it doesn't track P&L through the day. |
| Source timestamp | Every price and every VaR figure carries the as-of date of its underlying fixing, shown in the dashboard header |
| Secrets | None needed for market data — yfinance requires no API key, which removes a deployment dependency the FRED integration would otherwise have needed |
| Canonical currency value | Each currency gets one series U\_C,t = USD value of 1 unit of C (direct yfinance quote for EUR/GBP/AUD/NZD; 1/quote for JPY/CHF/CAD/CNH/MXN); every P&L and VaR formula consumes U\_C only, never a raw pair quote directly |
| Confidence level → z-score | 90% → 1.282, 95% → 1.645 (default), 99% → 2.326 (one-tailed normal quantile); the sidebar's confidence control looks this up rather than taking a raw z input |
| yfinance ingestion mechanism | The yfinance Python package (Ticker.history() for both the current quote and the 800-day lookback), wrapped in a thin client in market\_data.py so the caching and fallback logic stays in one place and swapping the provider later touches only that file |

## Edge cases & data-quality risks

Twelve edge cases across three categories are handled explicitly rather than left to crash or silently misreport.

| Category | Edge case | Handling |
| --- | --- | --- |
| Financial | Opposite-signed trades on the same pair | Net to the pair's true exposure; VaR reflects the netted risk, not the gross |
| Financial | Trade dated today (the position's own trade date) | Both inception and daily P&L show immediately off the live yfinance quote, session open or not; on the trade date they're the same number (P\_{i,t} − 0), which is standard desk convention — real systems show it, not blank. Included normally in the portfolio daily P&L sum. Flagged "new position" as an informational badge only, not a suppression trigger. |
| Financial | Zero or negative notional in the file | Row rejected at load with a validation error; the rest of the portfolio still loads |
| Financial | Pair not covered by yfinance | Position still shown if quotable, excluded from VaR, flagged "no market data" |
| Data quality | Duplicate trade\_id | Rejected at load and listed in a validation report |
| Data quality | Repeated/stale rate (holiday carried forward) | Flagged if the identical rate repeats for more than 2 consecutive business days |
| Data quality | Outlier / fat-finger rate spike | Shown, not auto-removed — a PM should see real volatility — but flagged if a daily return exceeds 5 standard deviations |
| Data quality | Mismatched as-of dates across pairs | Portfolio VaR is computed only on the latest common date; a banner names any pair lagging behind |
| Technical | Singular or near-singular covariance matrix | Diagonal regularised with a small epsilon (ridge) before inversion |
| Technical | API timeout or rate limit | Retried once, then falls back per the Market Data Strategy section |
| Technical | Large portfolio performance | Covariance is computed at pair level, not position level, keeping the matrix small regardless of trade count |
| Technical | Secrets in a public repo | No market-data secrets exist to leak — yfinance needs no API key; any Streamlit Cloud app-level secrets still follow st.secrets, never committed |
| Financial | Cross-pair entry price vs. triangulated rate don't exactly agree | Expected, not a bug: real cross rates trade with their own independent bid/offer, so inception P&L on a cross is a tiny non-zero basis rather than exactly 0 — disclosed in the README, not suppressed |
| Data quality | An existing position is missing a previous close, so its daily P&L is blank | Excluded from the portfolio daily P&L sum, not propagated as NaN and not silently treated as zero; the portfolio total carries a footnote naming how many positions were excluded |

## Test plan & acceptance criteria

Nine quantitative invariants and five acceptance checks gate the submission; all run as automated pytest tests, not manual spot-checks.

### Invariants

| Invariant | Check |
| --- | --- |
| Component VaR reconciliation | Sum of component VaRs equals portfolio VaR within 1e-6 |
| Subadditivity | Portfolio VaR ≤ sum of standalone VaRs (diversification benefit ≥ 0) |
| Single-position portfolio | Portfolio VaR = standalone VaR = component VaR for that one position |
| Flat net exposure | A pair netting to zero contributes 0 to portfolio VaR |
| Inception P&L at trade date | P&L = 0 when the current price equals the entry price |
| Type T vs Type B symmetry | An equivalent USD/XXX and XXX/USD position produce matching USD P&L signs |
| Covariance matrix validity | Σ is symmetric and positive semi-definite on random synthetic return data |
| VaR horizon scaling | 10-day VaR = 1-day VaR × √10, as a regression test on the horizon parameter |
| Schema validation | Malformed rows (missing column, bad date, negative notional) are rejected with a clear error, never a silent crash |
| Triangulation consistency | A cross position's P\_{i,t} formula matches its two-leg decomposition exactly (verified algebraically and with a numeric 2-currency example); confirms the unified formula collapses correctly rather than approximately |
| No false-stale after a holiday | Simulated case (last fixing = the day before a US Fed holiday, today = the day after it) does not raise the stale flag — catches a regression to naive Mon–Fri staleness logic |

### Acceptance criteria

| Criterion | Pass condition |
| --- | --- |
| Sample portfolio loads | App starts and renders the full dashboard with zero unhandled exceptions |
| VaR cross-check | Portfolio VaR matches an independent spreadsheet calculation within 0.5% |
| Deployed link | The public Streamlit Cloud URL loads the dashboard with no local-only dependencies |
| Data-quality banners | At least one deliberately broken row in the sample data visibly triggers a warning |
| README complete | Methodology, assumptions, limitations and next steps are all present |

## 24-hour delivery plan

Ten two-to-four-hour phases take the project from this spec to a deployed, tested, documented dashboard.

| Hours | Phase | Output |
| --- | --- | --- |
| 0–1 | Finalize this specification, confirm defaults, scaffold the repo | Agreed spec, repo skeleton, requirements.txt |
| 1–3 | Data layer: portfolio loader/validator, FRED + yfinance client, caching | Tested data-access modules, sample\_portfolio.csv |
| 3–6 | P&L engine (inception, daily, Type T/B formulas) | pnl.py plus unit tests |
| 6–10 | Risk engine (covariance, portfolio/standalone/marginal/component VaR) | risk.py plus invariant tests |
| 10–12 | Supplementary analytics: currency exposure, concentration, stress scenarios | exposure.py, stress.py |
| 12–16 | Streamlit UI build: tables, charts, banners, layout | app.py wired to the analytics layer |
| 16–18 | Integration and edge-case testing, performance check | Full pytest suite green |
| 18–20 | Deploy to Streamlit Community Cloud, smoke test the public link | Working deployed link |
| 20–22 | Write the README (methodology, limitations, next steps), finish docstrings | README.md |
| 22–24 | Buffer, final QA pass, package the submission | Source code, README and link submitted |

## Ambiguities, defaults & flagged inconsistencies

Seven open questions in the brief are closed here with an explicit default, so nothing is decided improvised mid-build.

| Ambiguity in the brief | Default chosen | Why |
| --- | --- | --- |
| "Appropriate data source" could mean one official source or a hybrid for reliability | yfinance alone, for both current price and the VaR history; FRED was tried and dropped once it was live-tested | A hybrid design was built and live-tested first; it works, but every edge case it adds (mixed-leg staleness, two calendars, two outage paths) is risk the PM's test link doesn't need. One source, backed by the local cache instead of a second live provider, is more reliable for exactly the goal that matters here — a link that works when opened. The official-source story is named as a disclosed trade-off, not hidden. |
| Cross pairs (neither leg USD) weren't addressed by name | Supported via triangulation: decomposed into two USD-vs-currency legs (e.g. EUR/GBP = long EUR leg + short GBP leg) | Adds no real cost: the risk engine already prices every currency against USD, so a cross is just two existing legs; only a currency with no USD-quoted series stays out of scope |
| "Approximately 252 observations" is imprecise | Exactly 252 trading days where available, minimum 60 with a warning | Gives a reproducible, testable default while degrading gracefully for new pairs |
| P&L exactness vs. VaR linearity | P&L is computed exactly, including Type B's non-linearity; VaR stays linear (delta-normal) | Flagged as a genuine inconsistency, not quietly resolved: reconciling the two exactly needs simulation-based VaR, out of scope here |
| VaR granularity isn't specified | Standalone, marginal and component VaR computed per trade, not just per pair | More useful to a PM who wants to see which specific trade drives risk, not just which currency |
| "A reasonable VaR methodology" is open-ended | Parametric (delta-normal), equal-weighted 252-day window (every day counts the same, unlike EWMA) | This is the brief's own suggested default; fastest to build, test and explain in 24 hours |
| "Additional analytics" is undefined | Currency exposure, VaR concentration, stress scenarios, data-quality banners | Directly useful to a PM monitoring risk and realistically buildable in the time available |

## README limitations to disclose

The README states eight limitations up front, so the Senior Portfolio Manager can judge the submission's scope correctly rather than discover gaps later.

- Parametric (delta-normal) VaR assumes normally-distributed returns and a static covariance matrix; it understates tail risk versus historical simulation or Monte Carlo, especially around FX regime shifts.
- "Current price" is the latest daily reference fixing, not an executable live quote; intraday moves since the last fixing aren't reflected.
- Cross-pair P&L and VaR rely on triangulation through two USD legs; a currency with no USD-quoted series at all (some thinly-traded EM currencies) still can't be priced.
- The covariance matrix uses an equal-weighted 252-day window — every day counts the same, unlike EWMA, which weights recent days more — so it reacts slowly to a sudden rise in volatility.
- Type B P&L (USD-base pairs) is computed exactly and is mildly non-linear in the rate, while VaR stays linear; the two are consistent models, not identical numbers, for large moves.
- No VaR backtesting (Kupiec-style exception counts) is included in v1.
- FX forwards, swaps and other instruments are out of scope; the brief's spot-only assumption is kept throughout.
- Single-source trade-off, made deliberately: yfinance has no official API (Yahoo retired theirs in 2017) and no SLA, which is why it was initially paired with FRED for the "official source" story. A live test found FRED's H.10 lagging by over a week in practice, and the two-source design was adding more risk (mixed-leg staleness, two calendars, two outage paths) than it removed. v1 uses yfinance alone, backed by local caching rather than a second live provider; reintroducing an official source for the VaR history is listed below as a future improvement, not hidden as an oversight.

What we'd improve with more time: EWMA/GARCH volatility, historical-simulation and Monte Carlo VaR as alternatives, VaR backtesting, an official reference-rate source (FRED or ECB) reintroduced alongside yfinance for the VaR history once the two-source edge cases are under full test coverage, FX forwards and swaps, and a configurable reporting currency beyond USD.

## Appendix: original task brief (verbatim)

The Senior Portfolio Manager's brief, unedited, as supplied at the start of this assessment — kept here so the derived spec above can always be checked back against the actual ask:

> FX Portfolio Management Dashboard
>
> Build a portfolio-management dashboard that an FX portfolio manager could use to monitor current P&L and risk.
>
> Assume initially that the portfolio contains FX spot positions. You may add other products if you wish, but this is not required.
>
> Your application should:
>
> - Read an existing portfolio from the supplied file/database (create your own sample file). The data will include fields such as trade date, currency pair, notional and entry price.
> - Obtain current and historical market prices from an appropriate data source.
> - Display the current portfolio and relevant position information, including USD P&L, both daily and since trade inception.
> - Calculate and display position-level and portfolio-level risk. At minimum, include position VaR, portfolio VaR and marginal/component VaR. You may choose a reasonable VaR methodology and parameters.
> - Include any additional analytics or functionality that you believe would genuinely be useful to a portfolio manager.
>
> You may use AI tools and external libraries. You should nevertheless understand and be able to explain all material parts of the submitted code and calculations.
>
> Please also provide a short README describing:
>
> - your methodology and major assumptions;
> - any limitations or known issues;
> - what you would improve if given more time.
>
> Submit:
>
> - a working link to the dashboard;
> - the complete source code;
> - the README.
>
> You have 24 hours from your chosen start time to complete the assignment.
