"""Portfolio, standalone, marginal and component VaR per the spec's 'VaR' section.

    sigma_p^2 = w'Sigma w,          VaR_p = z * sigma_p
    Standalone_i = z * sqrt(w_i'Sigma w_i)     (position i's own exposure vector, zero elsewhere;
                                                 collapses to z*|w_i|*sigma_i for a one-leg position)
    Component_i  = z * (w_i'Sigma w) / sigma_p  (position i's exact share of VaR_p)
    Marginal_i   = Component_i / x_i,  x_i = s_i * N_i  (signed base-currency notional)

Component_i generalises the spec's currency-level CVaR_i = w_i * MVaR_i to position
granularity: at currency level, "w_i" there is the currency's own dollar exposure and
MVaR_i = z*(Sigma w)_i/sigma_p is its per-dollar sensitivity. At position level the
natural per-unit variable is the position's own signed notional x_i, not a dollar
amount (its exposure vector w_i spans up to two currencies), so Marginal_i is defined
as the per-unit-of-x_i sensitivity and Component_i = x_i * Marginal_i is "marginal VaR
scaled back to the position's actual size" exactly as the spec's prose describes.
Euler's theorem (sum_i Component_i = VaR_p, shown algebraically in the spec's
Verification note) holds for this generalisation: Component_i = z*(w_i'Sigma w)/sigma_p
sums over positions to z*(w'Sigma w)/sigma_p = z*sigma_p = VaR_p, since w = sum_i w_i.

w (net USD exposure per currency) and Sigma (covariance of daily log returns vs USD,
over the lookback window) are both restricted to the currencies in `returns`' columns.
USD is never a dimension of either (U_USD = 1 has zero variance, so its leg always
drops out); a position with a leg in a currency outside that set has no market data
for that leg and is excluded from VaR entirely (flagged, not silently partial).

Nothing here ever inverts Sigma -- every formula above uses only w'Sigma w or Sigma w,
both well-defined for any positive *semi*-definite matrix. So the spec's "ridge before
inversion" edge case is, strictly, insurance against a problem this engine doesn't
directly have; it is kept anyway, applied only when the raw sample covariance fails
the positive-semidefinite check (see validate_covariance_matrix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data.market_data import Quote

# The sidebar still only offers these three confidence levels (never a raw z input),
# but the z-value itself is computed live via scipy.stats.norm.ppf, not a hardcoded table.
CONFIDENCE_LEVELS: tuple[float, ...] = (0.90, 0.95, 0.99)
DEFAULT_CONFIDENCE = 0.95
DEFAULT_HORIZON_DAYS = 1
DEFAULT_LOOKBACK_DAYS = 252
MIN_LOOKBACK_DAYS = 60
MAX_LOOKBACK_DAYS = 500

# Eigenvalues at or above this (a small negative number, allowing for floating-point
# noise around zero) count as non-negative for the positive-semidefinite check.
PSD_TOLERANCE = -1e-10

# Added to Sigma's diagonal, scaled to each currency's own variance -- applied only as
# a fallback when the raw sample covariance fails the PSD check (see
# validate_covariance_matrix), not unconditionally. See the module-level note below on
# why this is still worth keeping even though nothing here ever inverts Sigma.
RIDGE_EPSILON = 1e-10


def z_score(confidence: float) -> float:
    """One-tailed normal quantile for a confidence level, via scipy.stats.norm.ppf.
    Restricted to CONFIDENCE_LEVELS so the sidebar's confidence control always looks
    this up from a fixed set rather than accepting an arbitrary raw z."""
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(
            f"unsupported confidence level {confidence!r}; choose one of {CONFIDENCE_LEVELS}"
        )
    return float(norm.ppf(confidence))


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """The three properties a valid covariance matrix must have."""

    symmetric: bool
    finite: bool
    positive_semidefinite: bool
    min_eigenvalue: float


def validate_covariance_matrix(sigma: pd.DataFrame, tol: float = PSD_TOLERANCE) -> CovarianceDiagnostics:
    """Checks symmetry, finiteness, and positive semi-definiteness (every eigenvalue
    >= tol). Used both internally, to decide whether covariance_matrix() needs its
    ridge fallback, and directly in tests against both the raw and final matrix."""
    m = sigma.to_numpy()
    finite = bool(np.isfinite(m).all())
    symmetric = bool(np.allclose(m, m.T, atol=1e-12)) if finite else False
    min_eigenvalue = float(np.linalg.eigvalsh((m + m.T) / 2).min()) if finite else float("nan")
    positive_semidefinite = finite and min_eigenvalue >= tol
    return CovarianceDiagnostics(
        symmetric=symmetric, finite=finite, positive_semidefinite=positive_semidefinite, min_eigenvalue=min_eigenvalue
    )


def _symmetrize(sigma: pd.DataFrame) -> pd.DataFrame:
    """Forces exact symmetry. A covariance matrix is symmetric by construction, but
    cov(A, B) and cov(B, A) can land on different floating-point values depending on
    computation order, which an exact np.allclose(m, m.T) check would otherwise fail."""
    m = sigma.to_numpy()
    return pd.DataFrame((m + m.T) / 2, index=sigma.index, columns=sigma.columns)


def _apply_ridge(sigma: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    diag = np.diag(sigma.to_numpy())
    ridge = epsilon * np.where(diag > 0, diag, 1.0)
    return sigma + np.diag(ridge)


def regularize_if_needed(sigma: pd.DataFrame, epsilon: float = RIDGE_EPSILON) -> pd.DataFrame:
    """Returns sigma unchanged whenever it already passes validate_covariance_matrix's
    PSD check -- which a true sample covariance always does, a zero eigenvalue
    (merely singular, e.g. two currencies moving in exact lockstep) included, since
    positive *semi*-definite explicitly allows zero. Only a matrix that is genuinely
    indefinite (a negative eigenvalue, beyond floating-point noise) gets the ridge
    fallback. Exposed separately from covariance_matrix so this conditional behaviour
    is directly testable against a hand-built matrix, since pandas' own .cov() is
    essentially never indefinite in practice."""
    sigma = _symmetrize(sigma)
    if validate_covariance_matrix(sigma).positive_semidefinite:
        return sigma
    return _symmetrize(_apply_ridge(sigma, epsilon))


def covariance_matrix(
    returns: pd.DataFrame, lookback_days: int = DEFAULT_LOOKBACK_DAYS, epsilon: float = RIDGE_EPSILON
) -> pd.DataFrame:
    """Sample covariance of daily log returns, restricted to dates common to every
    currency in `returns` (inner join -- this is enforced here, not just assumed of
    the caller) and then to the most recent `lookback_days` of those common dates.
    Regularised via regularize_if_needed, which leaves realistic, properly-aligned
    data unchanged and only intervenes on genuine indefiniteness."""
    aligned = returns.dropna(how="any")
    window = aligned.tail(lookback_days)
    if len(window) < 2:
        raise ValueError(f"need at least 2 return observations common to every currency, got {len(window)}")
    return regularize_if_needed(window.cov(), epsilon)


def position_exposures(
    trades: pd.DataFrame, quotes: Mapping[str, Quote], currencies: Iterable[str]
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """w_i per position: USD exposure per currency (trade_id index x currency columns),
    restricted to `currencies`. USD legs are skipped (zero variance, so always zero
    exposure in this restricted space). A position with a non-USD leg outside
    `currencies`, or missing a quote for one, has no row and its trade_id is returned
    separately as excluded (no market data, not partially priced)."""
    currencies = list(currencies)
    rows: dict[str, dict[str, float]] = {}
    excluded: list[str] = []

    for _, t in trades.iterrows():
        trade_id = t["trade_id"]
        legs: dict[str, float] = {}
        priceable = True
        for ccy, is_base in ((t["base_currency"], True), (t["term_currency"], False)):
            if ccy == "USD":
                continue
            if ccy not in currencies or ccy not in quotes:
                priceable = False
                break
            u = quotes[ccy].value
            if is_base:
                legs[ccy] = legs.get(ccy, 0.0) + t["side_sign"] * t["notional_base_ccy"] * u
            else:
                legs[ccy] = legs.get(ccy, 0.0) - (
                    t["side_sign"] * t["notional_base_ccy"] * t["entry_price"] * u
                )
        if priceable:
            rows[trade_id] = legs
        else:
            excluded.append(trade_id)

    exposures = pd.DataFrame.from_dict(rows, orient="index", columns=currencies).fillna(0.0)
    exposures.index.name = "trade_id"
    return exposures, tuple(excluded)


def standalone_var(exposures: pd.DataFrame, sigma: pd.DataFrame, z: float, scale: float = 1.0) -> pd.Series:
    """z*sqrt(w'Sigma w) for each row of `exposures` (the full quadratic form, so a
    two-leg row has both legs and their correlation accounted for; collapses to
    z*|w|*sigma for a one-leg row), against the SAME Sigma either way.

    `exposures` can be indexed by anything: trade_id for per-position standalone
    VaR, or a currency_pair after summing the trade-level exposure vectors of every
    trade sharing that pair, for a pair's net ("position") standalone VaR. Either
    way this must run on each row's own exposure vector directly, never be
    approximated by summing standalone VaRs computed at a finer grain: summing
    throws away any netting between offsetting rows (e.g. two trades on the same
    pair with opposite signs), which is exactly what standalone VaR, computed on
    the net exposure, is supposed to capture."""
    currencies = list(sigma.columns)
    exposures = exposures.reindex(columns=currencies, fill_value=0.0)
    if exposures.empty:
        return pd.Series(dtype=float, index=exposures.index)
    sigma_np = sigma.to_numpy()
    exposures_np = exposures.to_numpy()
    values = z * np.sqrt(np.clip(np.einsum("ij,jk,ik->i", exposures_np, sigma_np, exposures_np), 0, None)) * scale
    return pd.Series(values, index=exposures.index)


@dataclass(frozen=True)
class VaRMetrics:
    net_exposure: pd.Series  # w
    portfolio_var: float  # VaR_p
    standalone_sum: float  # sum_i Standalone_i
    positions: pd.DataFrame  # index trade_id: standalone_var, marginal_var, component_var


def var_metrics(
    exposures: pd.DataFrame, sigma: pd.DataFrame, signed_notional: pd.Series, z: float, scale: float = 1.0
) -> VaRMetrics:
    """The four VaR formulas, given exposures (trade_id x currency, from
    position_exposures), Sigma (currency x currency) and each position's signed
    notional x_i = s_i*N_i. Separated from compute_var so the formulas themselves are
    testable against hand-built matrices, independent of return-series plumbing."""
    currencies = list(sigma.columns)
    sigma_np = sigma.to_numpy()
    exposures = exposures.reindex(columns=currencies, fill_value=0.0)

    w = exposures.sum(axis=0)
    w_np = w.to_numpy()
    portfolio_variance = max(float(w_np @ sigma_np @ w_np), 0.0)
    sigma_p = float(np.sqrt(portfolio_variance))
    portfolio_var = z * sigma_p * scale

    if exposures.empty:
        empty = pd.DataFrame(columns=["standalone_var", "marginal_var", "component_var"])
        return VaRMetrics(net_exposure=w, portfolio_var=portfolio_var, standalone_sum=0.0, positions=empty)

    standalone = standalone_var(exposures, sigma, z, scale)
    exposures_np = exposures.to_numpy()

    sigma_w = sigma_np @ w_np
    if sigma_p > 0:
        component_var = z * (exposures_np @ sigma_w) / sigma_p * scale
    else:
        # Net exposure is exactly flat: the portfolio has zero VaR, so no position
        # -- however large its own standalone risk -- has anything left to contribute to.
        component_var = np.zeros(len(exposures))

    sn = signed_notional.reindex(exposures.index).to_numpy()
    marginal_var = np.divide(component_var, sn, out=np.zeros_like(component_var), where=sn != 0)

    positions = pd.DataFrame(
        {"standalone_var": standalone, "marginal_var": marginal_var, "component_var": component_var},
        index=exposures.index,
    )
    return VaRMetrics(
        net_exposure=w, portfolio_var=portfolio_var, standalone_sum=float(standalone.sum()), positions=positions
    )


@dataclass(frozen=True)
class VaRResult:
    confidence: float
    z: float
    horizon_days: int
    lookback_days: int
    currencies: tuple[str, ...]
    net_exposure: pd.Series  # w: net USD exposure per currency
    sigma: pd.DataFrame  # Sigma, ridge-regularised, over `currencies`
    portfolio_var: float  # VaR_p
    standalone_sum: float  # sum_i Standalone_i (undiversified total)
    diversification_benefit: float  # DB = standalone_sum - portfolio_var
    positions: pd.DataFrame  # index trade_id: standalone_var, marginal_var, component_var
    excluded_trade_ids: tuple[str, ...]  # no market data, excluded from VaR

    @property
    def scale(self) -> float:
        """sqrt(horizon_days): the horizon-scaling factor every VaR figure here was
        multiplied by. Exposed so any VaR recomputed at a different grain from this
        same result (e.g. exposure.py's pair-level standalone VaR) scales consistently."""
        return float(np.sqrt(self.horizon_days))


def compute_var(
    trades: pd.DataFrame,
    quotes: Mapping[str, Quote],
    returns: pd.DataFrame,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    epsilon: float = RIDGE_EPSILON,
) -> VaRResult:
    z = z_score(confidence)
    sigma = covariance_matrix(returns, lookback_days, epsilon)
    currencies = tuple(sigma.columns)
    scale = float(np.sqrt(horizon_days))

    exposures, excluded = position_exposures(trades, quotes, currencies)
    signed_notional = trades.set_index("trade_id")["side_sign"] * trades.set_index("trade_id")["notional_base_ccy"]
    metrics = var_metrics(exposures, sigma, signed_notional, z, scale)

    return VaRResult(
        confidence=confidence,
        z=z,
        horizon_days=horizon_days,
        lookback_days=lookback_days,
        currencies=currencies,
        net_exposure=metrics.net_exposure,
        sigma=sigma,
        portfolio_var=metrics.portfolio_var,
        standalone_sum=metrics.standalone_sum,
        diversification_benefit=metrics.standalone_sum - metrics.portfolio_var,
        positions=metrics.positions,
        excluded_trade_ids=excluded,
    )
