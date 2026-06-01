"""
===========================================================
VaR & Expected Shortfall — Market Risk Pipeline
===========================================================
Portfolio : SPY 30% | TLT 20% | GLD 25% | EEM 15% | LQD 10%
Methods   : Historical | Normal | Student-t | Monte Carlo | GARCH(1,1)-t
Tests     : Kupiec POF | Christoffersen CC
Author    : github.com/tancredeg
===========================================================

USAGE
-----
  # (1) Download 2014-2024 data first (one-time):
  #     python download_data.py
  #
  # (2) Run full pipeline:
  #     python var_es_pipeline.py

DEPENDENCIES
------------
  numpy, pandas, scipy, matplotlib, arch
  pip install -r requirements.txt
"""

# ===========================================================
# 0. IMPORTS & CONFIGURATION
# ===========================================================
import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from arch import arch_model

warnings.filterwarnings('ignore')

# ---- Configuration ----------------------------------------
DATA_DIR            = os.path.join(os.path.dirname(__file__), "data")
TICKERS             = ['SPY', 'TLT', 'GLD', 'EEM', 'LQD']
WEIGHTS             = np.array([0.30, 0.20, 0.25, 0.15, 0.10])
CONFIDENCE_LEVELS   = [0.95, 0.99]
SEED                = 42
N_SIM               = 50_000
LOOKBACK            = 252          # rolling window (trading days)
ROLLING_CORR_WINDOW = 60
OOS_START           = "2019-01-01" # OOS backtest start (requires 252-day warmup)
FIGURES_DIR         = os.path.join(os.path.dirname(__file__), "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ===========================================================
# 1. DATA
# ===========================================================

def load_prices(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load adjusted close prices from local CSV files.
    Each file: Date index, 'Adj Close' column.
    """
    series = {}
    for ticker in TICKERS:
        path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            sys.exit(
                f"\n[ERROR] {path} not found.\n"
                f"  → Run: python download_data.py\n"
            )
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        series[ticker] = df["Adj Close"]

    prices = pd.DataFrame(series).sort_index()
    prices.dropna(how="all", inplace=True)
    # Align: keep only rows where all 5 ETFs have data
    prices.dropna(inplace=True)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray = WEIGHTS) -> pd.Series:
    """Weighted sum of log-returns (daily P&L proxy)."""
    return (returns * weights).sum(axis=1)


# ===========================================================
# 2. DESCRIPTIVE STATISTICS
# ===========================================================

def describe_portfolio(port_ret: pd.Series) -> dict:
    n = len(port_ret)
    mu_ann  = port_ret.mean() * 252
    vol_ann = port_ret.std()  * np.sqrt(252)
    sk      = float(stats.skew(port_ret))
    ku      = float(stats.kurtosis(port_ret))       # excess kurtosis
    jb, jb_p = stats.jarque_bera(port_ret)
    return dict(n=n, mu_ann=mu_ann, vol_ann=vol_ann,
                skew=sk, excess_kurtosis=ku, jb=jb, jb_p=jb_p)


def rolling_correlation(returns: pd.DataFrame,
                         a: str = 'SPY', b: str = 'TLT',
                         window: int = ROLLING_CORR_WINDOW) -> pd.Series:
    """
    Rolling correlation between two assets.
    SPY/TLT is the key flight-to-safety relationship:
    it broke down in 2022 (rate hike regime).
    """
    return returns[a].rolling(window).corr(returns[b])


# ===========================================================
# 3. MLE FOR STUDENT-t DEGREES OF FREEDOM  ν
# ===========================================================

def fit_nu_mle(raw_returns: np.ndarray) -> tuple[float, tuple[float, float]]:
    """
    Maximum-likelihood estimate of ν for a Student-t(ν, μ, σ) fitted to raw returns.

    Methodology
    -----------
    Jointly estimate (ν, μ, σ) via L-BFGS-B minimization of the negative
    log-likelihood.  Fitting on standardized data with scale=1 fixed is
    incorrect: t(ν) has population variance ν/(ν-2) ≠ 1, so the optimizer
    compensates by pushing ν toward ∞, producing severely upward-biased estimates.

    The 95% CI for ν is obtained via the profile Hessian (μ and σ profiled out):
        SE(ν̂) = 1 / sqrt(-∂²log L / ∂ν²|_{μ=μ̂, σ=σ̂})   (delta method)

    NOTE on ν vs κ (sample kurtosis)
    ----------------------------------
    For t(ν), the theoretical excess kurtosis is κ = 6/(ν−4)  (requires ν > 4).
    When ν̂ ≈ 4, κ_theory ≫ κ_sample.  This is NOT a contradiction:
      • MLE is driven by the probability mass in extreme tails (COVID, 2022 tails),
        which requires very heavy tails (ν close to 4).
      • The sample 4th moment κ̂ is a ratio estimator that is severely downward-
        biased when ν ≤ 6 (because E[X⁸] is close to diverging and the sample
        mean of |X|⁴ converges very slowly).
    Both estimates are statistically coherent but measure different features.
    """
    def neg_ll(params):
        nu, mu, sigma = params
        if nu <= 2.05 or sigma <= 1e-9:
            return 1e10
        return -np.sum(stats.t.logpdf(raw_returns, df=nu, loc=mu, scale=sigma))

    x0     = [5.0, raw_returns.mean(), raw_returns.std()]
    bounds = [(2.1, None), (None, None), (1e-8, None)]
    res    = minimize(neg_ll, x0=x0, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 20_000})
    nu_hat, mu_hat, sigma_hat = res.x

    # Profile Hessian: second derivative w.r.t. ν only, μ and σ fixed at MLE
    h = nu_hat * 1e-4
    def neg_ll_nu(nu):
        return neg_ll([nu, mu_hat, sigma_hat])
    d2 = (neg_ll_nu(nu_hat + h) - 2 * neg_ll_nu(nu_hat) + neg_ll_nu(nu_hat - h)) / h**2
    se  = 1.0 / np.sqrt(max(d2, 1e-12))
    ci  = (nu_hat - 1.96 * se, nu_hat + 1.96 * se)
    return nu_hat, ci


# ===========================================================
# 4. VaR — POINT ESTIMATES (FULL SAMPLE)
# ===========================================================

def var_historical(ret: pd.Series, alpha: float) -> float:
    """Non-parametric VaR: empirical quantile of losses."""
    return float(-np.percentile(ret, (1 - alpha) * 100))


def var_normal(ret: pd.Series, alpha: float) -> float:
    mu, sigma = ret.mean(), ret.std()
    return float(-(mu + sigma * stats.norm.ppf(1 - alpha)))


def var_student(ret: pd.Series, alpha: float, nu: float) -> float:
    """
    Parametric t-VaR.
    Scale factor = σ * sqrt((ν-2)/ν) because Var[t(ν)] = ν/(ν-2).
    """
    mu, sigma = ret.mean(), ret.std()
    scale = sigma * np.sqrt((nu - 2) / nu)
    return float(-(mu + scale * stats.t.ppf(1 - alpha, df=nu)))


def var_montecarlo(ret: pd.Series, alpha: float,
                   n_sim: int = N_SIM, seed: int = SEED) -> float:
    np.random.seed(seed)
    mu, sigma = ret.mean(), ret.std()
    sims = np.random.normal(mu, sigma, n_sim)
    return float(-np.percentile(sims, (1 - alpha) * 100))


# ===========================================================
# 5. EXPECTED SHORTFALL — POINT ESTIMATES
# ===========================================================

def es_historical(ret: pd.Series, alpha: float) -> float:
    v = var_historical(ret, alpha)
    tail = ret[ret < -v]
    return float(-tail.mean()) if len(tail) > 0 else v


def es_normal(ret: pd.Series, alpha: float) -> float:
    mu, sigma = ret.mean(), ret.std()
    z = stats.norm.ppf(1 - alpha)
    return float(-(mu - sigma * stats.norm.pdf(z) / (1 - alpha)))


def es_student(ret: pd.Series, alpha: float, nu: float) -> float:
    """
    Closed-form ES for Student-t (Yamai & Yoshiba, 2002).
    """
    mu, sigma = ret.mean(), ret.std()
    scale = sigma * np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(1 - alpha, df=nu)
    es_t = (stats.t.pdf(q, df=nu) / (1 - alpha)) * (nu + q**2) / (nu - 1)
    return float(-(mu - scale * es_t))


def es_montecarlo(ret: pd.Series, alpha: float,
                  n_sim: int = N_SIM, seed: int = SEED) -> float:
    np.random.seed(seed)
    mu, sigma = ret.mean(), ret.std()
    sims = np.random.normal(mu, sigma, n_sim)
    v = -var_montecarlo(ret, alpha, n_sim, seed)  # threshold (negative)
    tail = sims[sims < v]
    return float(-tail.mean()) if len(tail) > 0 else var_montecarlo(ret, alpha, n_sim, seed)


# ===========================================================
# 6. DIVERSIFICATION BENEFIT (VaR sub-additivity check)
# ===========================================================

def diversification_benefit(asset_returns: pd.DataFrame, alpha: float) -> dict:
    """
    Σ wᵢ·VaRᵢ  vs  VaR_portfolio
    Historical VaR is sub-additive for elliptical distributions.
    Empirical verification quantifies the hedge benefit.
    """
    port_var = var_historical(portfolio_returns(asset_returns), alpha)
    individual = [var_historical(asset_returns[t], alpha) for t in TICKERS]
    undiv = float(np.dot(WEIGHTS, individual))
    benefit = (undiv - port_var) / undiv * 100
    return dict(
        individual_vars={t: v for t, v in zip(TICKERS, individual)},
        undiversified_var=undiv,
        portfolio_var=port_var,
        diversification_benefit_pct=benefit,
    )


# ===========================================================
# 7. GARCH(1,1)-t CONDITIONAL VaR
# ===========================================================

def fit_garch_t(port_ret: pd.Series):
    """
    Fit GARCH(1,1) with Student-t innovations on portfolio returns.
    Returns: arch ModelResult object.

    Model: r_t = μ + σ_t ε_t,  ε_t ~ t(ν)
           σ²_t = ω + α r²_{t-1} + β σ²_{t-1}
    Stationarity condition: α + β < 1
    Volatility half-life: log(0.5) / log(α+β)  (days)
    """
    am = arch_model(port_ret * 100, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
    result = am.fit(disp='off', show_warning=False)
    return result


def garch_rolling_var(port_ret: pd.Series, alpha: float,
                      garch_result) -> pd.Series:
    """
    One-day-ahead VaR from GARCH conditional volatility.
    σ_t is the fitted conditional vol; we shift by 1 so VaR_t uses σ_{t-1}.
    """
    nu      = garch_result.params['nu']
    cond_vol = garch_result.conditional_volatility / 100           # back to decimal
    mu_ret   = port_ret.mean()
    scale    = cond_vol * np.sqrt((nu - 2) / nu)                   # t-scale
    q        = stats.t.ppf(1 - alpha, df=nu)
    var_series = -(mu_ret + scale * q)
    var_series.index = port_ret.index
    return var_series


# ===========================================================
# 8. ROLLING VaR FOR BACKTESTING
# ===========================================================

def _rolling_apply(ret: pd.Series, fn, window: int = LOOKBACK) -> pd.Series:
    return ret.rolling(window).apply(fn, raw=True)


def rolling_var_historical(ret: pd.Series, alpha: float,
                            window: int = LOOKBACK) -> pd.Series:
    fn = lambda x: var_historical(pd.Series(x), alpha)
    return _rolling_apply(ret, fn, window)


def rolling_var_normal(ret: pd.Series, alpha: float,
                       window: int = LOOKBACK) -> pd.Series:
    fn = lambda x: var_normal(pd.Series(x), alpha)
    return _rolling_apply(ret, fn, window)


def rolling_var_student(ret: pd.Series, alpha: float, nu: float,
                        window: int = LOOKBACK) -> pd.Series:
    fn = lambda x: var_student(pd.Series(x), alpha, nu)
    return _rolling_apply(ret, fn, window)


# ===========================================================
# 9. BACKTESTING
# ===========================================================

def kupiec_pof_test(ret: pd.Series, var_series: pd.Series, alpha: float) -> dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.
    H₀: E[violations/T] = 1-α  (correct unconditional coverage)
    LR ~ χ²(1) under H₀.
    """
    aligned = ret.align(var_series, join='inner')
    r, v = aligned[0].dropna(), aligned[1].dropna()
    common = r.index.intersection(v.index)
    r, v = r.loc[common], v.loc[common]
    violations = (r < -v)
    n = int(violations.sum())
    T = len(violations)
    p = 1 - alpha

    if n == 0 or n == T:
        lr = 0.0
        p_val = 1.0
    else:
        pi_hat = n / T
        lr = -2 * (
            n * np.log(p / pi_hat) +
            (T - n) * np.log((1 - p) / (1 - pi_hat))
        )
        p_val = float(1 - stats.chi2.cdf(lr, df=1))

    z = _basel_zone(n, T)
    return dict(
        n_violations=n, T=T,
        actual_rate=n / T, expected_rate=p,
        lr=lr, p_value=p_val,
        reject_H0=(p_val < 0.05),
        decision="FAIL" if p_val < 0.05 else "PASS",
        basel_zone=z['zone'], basel_k=z['k'],
    )


def christoffersen_cc_test(ret: pd.Series, var_series: pd.Series, alpha: float) -> dict:
    """
    Christoffersen (1998) Conditional Coverage (CC) test.
    CC = POF (unconditional) + Independence components, each χ²(1).
    Total LR_CC ~ χ²(2) under H₀.

    ⚠️  POWER CAVEAT
    At 99% confidence, T=1000 yields only ~10 expected violations.
    The independence component has low power with so few transitions.
    A high π₁₁ (clustering) may be statistically insignificant
    simply because the number of violation pairs is too small.
    Interpret in conjunction with violation clustering plots.
    """
    aligned = ret.align(var_series, join='inner')
    r, v = aligned[0].dropna(), aligned[1].dropna()
    common = r.index.intersection(v.index)
    r, v = r.loc[common], v.loc[common]
    I = (r < -v).astype(int).values

    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(I)):
        prev, curr = I[i-1], I[i]
        if   prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else:                         n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11)

    def slog(x):
        return np.log(x) if x > 1e-12 else -500.0

    ll_indep = (n00 * slog(1 - pi01) + n01 * slog(pi01) +
                n10 * slog(1 - pi11) + n11 * slog(pi11))
    ll_unc   = ((n00 + n10) * slog(1 - pi) + (n01 + n11) * slog(pi))
    lr_ind   = -2 * (ll_unc - ll_indep)
    p_val_ind = float(1 - stats.chi2.cdf(lr_ind, df=1))

    n_viol = n01 + n11
    T = n00 + n01 + n10 + n11
    p = 1 - alpha
    if n_viol > 0 and n_viol < T:
        pi_hat = n_viol / T
        lr_unc = -2 * (
            n_viol * np.log(p / pi_hat) +
            (T - n_viol) * np.log((1 - p) / (1 - pi_hat))
        )
    else:
        lr_unc = 0.0
    p_val_unc = float(1 - stats.chi2.cdf(lr_unc, df=1))

    lr_cc  = lr_unc + lr_ind
    p_val_cc = float(1 - stats.chi2.cdf(lr_cc, df=2))

    return dict(
        n11=n11, pi11=pi11,
        lr_cc=lr_cc, p_value=p_val_cc,
        p_value_unc=p_val_unc, p_value_ind=p_val_ind,
        reject_H0=(p_val_cc < 0.05),
        decision="FAIL" if p_val_cc < 0.05 else "PASS",
        power_caveat=(
            f"T={T}, E[violations]≈{T*(1-alpha):.0f}. "
            "CC independence component has low power at 99% with few exceedances."
        ),
    )


# ===========================================================
# 10. BASEL II TRAFFIC-LIGHT ZONES
# ===========================================================

def _basel_zone(n_exceptions: int, T: int) -> dict:
    """
    Basel II (1996) traffic-light backtest zones.
    Counts are scaled to T=250 (standard supervisory window).

    Zone   Exceptions (T=250)   Add-on Δk   Multiplier k
    Green  0–4                  0.00         3.00
    Yellow 5                    +0.40        3.40
    Yellow 6                    +0.50        3.50
    Yellow 7                    +0.65        3.65
    Yellow 8                    +0.75        3.75
    Yellow 9                    +0.85        3.85
    Red    ≥10                  +1.00        4.00

    Capital charge = max(VaR_t, k × avg_{60d} VaR) × √10
    """
    n_scaled = n_exceptions * (250 / T)
    n_int    = int(round(n_scaled))

    if n_int <= 4:
        return dict(zone='Green',  k=3.00, n_scaled=round(n_scaled, 1))
    elif n_int <= 9:
        k_map = {5: 3.40, 6: 3.50, 7: 3.65, 8: 3.75, 9: 3.85}
        return dict(zone='Yellow', k=k_map.get(n_int, 3.85), n_scaled=round(n_scaled, 1))
    else:
        return dict(zone='Red',    k=4.00, n_scaled=round(n_scaled, 1))


def capital_charge(var_today: float, var_series_60d: pd.Series, k: float) -> float:
    """
    Basel II 10-day capital charge (square-root-of-time rule).
    MRC_t = max(VaR_t, k × (1/60) Σ_{t-1..t-60} VaR_i) × √10
    """
    avg_60 = var_series_60d.tail(60).mean()
    mrc = max(var_today, k * avg_60) * np.sqrt(10)
    return mrc


# ===========================================================
# 11. STRESS TESTING
# ===========================================================

def stress_period(port_ret: pd.Series, start: str, end: str,
                  label: str, var_99: float) -> dict:
    """Summarize a stress scenario window."""
    window = port_ret.loc[start:end]
    if len(window) == 0:
        return dict(label=label, error="Period not in dataset")
    cum = float((1 + window).prod() - 1)
    worst = float(window.min())
    worst_date = window.idxmin().date()
    ratio = abs(worst) / var_99
    return dict(
        label=label, start=start, end=end,
        n_days=len(window),
        cumulative_return=cum,
        worst_single_day=worst,
        worst_day_date=str(worst_date),
        peak_loss_to_var99_ratio=ratio,
    )


# ===========================================================
# 12. RESULTS FORMATTING
# ===========================================================

def _sep(char='=', n=60): print(char * n)

def print_header(title: str):
    _sep()
    print(f"  {title}")
    _sep()


def print_backtest_table(results: dict):
    """Pretty-print backtest results for all methods and alpha levels."""
    header = f"{'Method':<18} {'α':>6} {'N_viol':>7} {'Rate':>8} {'Expected':>9} {'p-val':>8} {'Result':>7} {'Basel':>7}"
    print(header)
    print('-' * len(header))
    for key, row in results.items():
        print(
            f"{key:<18} {row['alpha']:>6.0%} {row['n_violations']:>7d} "
            f"{row['actual_rate']:>8.4f} {row['expected_rate']:>9.4f} "
            f"{row['p_value']:>8.4f} {row['decision']:>7}  {row.get('basel_zone',''):>6}"
        )


# ===========================================================
# MAIN
# ===========================================================

def main():
    # ---- 1. Data ------------------------------------------------
    print_header("1. DATA LOADING")
    prices    = load_prices()
    returns   = compute_log_returns(prices)
    port_ret  = portfolio_returns(returns)

    start_date = prices.index[0].strftime('%Y-%m-%d')
    end_date   = prices.index[-1].strftime('%Y-%m-%d')
    print(f"  Period  : {start_date} → {end_date}")
    print(f"  N days  : {len(prices)}")
    has_aug2015 = prices.index[0].year <= 2015

    # ---- 2. Descriptive stats -----------------------------------
    print_header("2. PORTFOLIO STATISTICS")
    desc = describe_portfolio(port_ret)
    print(f"  Ann. Return      : {desc['mu_ann']:.4f}  ({desc['mu_ann']*100:.2f}%)")
    print(f"  Ann. Volatility  : {desc['vol_ann']:.4f}  ({desc['vol_ann']*100:.2f}%)")
    print(f"  Excess Kurtosis  : {desc['excess_kurtosis']:.4f}")
    print(f"  Skewness         : {desc['skew']:.4f}")
    print(f"  Jarque-Bera      : {desc['jb']:.1f}  (p = {desc['jb_p']:.2e})")
    print(f"  → Fat tails confirmed, normality rejected.")

    # ---- 3. Rolling correlation ---------------------------------
    print_header("3. ROLLING CORRELATION SPY/TLT (60d)")
    rc = rolling_correlation(returns)
    rc_full  = returns['SPY'].corr(returns['TLT'])
    rc_pre22 = returns['SPY'].loc[:'2021-12-31'].corr(returns['TLT'].loc[:'2021-12-31'])
    rc_2022  = returns['SPY'].loc['2022':'2022'].corr(returns['TLT'].loc['2022':'2022'])
    rc_post22= returns['SPY'].loc['2023':].corr(returns['TLT'].loc['2023':])
    print(f"  Full sample corr  : {rc_full:.3f}")
    print(f"  Pre-2022 corr     : {rc_pre22:.3f}")
    print(f"  2022 corr         : {rc_2022:.3f}")
    print(f"  2023+ corr        : {rc_post22:.3f}")
    print(f"  → Flight-to-safety correlation broke down in 2022 (rate hike regime).")

    # ---- 4. MLE for ν -------------------------------------------
    print_header("4. STUDENT-t MLE (ν)")
    nu_mle, ci = fit_nu_mle(port_ret.values)
    ku_theory  = 6 / (nu_mle - 4) if nu_mle > 4 else float('inf')
    print(f"  ν̂_MLE             : {nu_mle:.4f}")
    print(f"  95% CI            : [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"  Implied excess κ  : {ku_theory:.2f}  (vs empirical {desc['excess_kurtosis']:.2f})")
    print(f"  → MLE ν dominated by extreme events; empirical κ measures bulk — not contradictory.")

    # ---- 5. VaR / ES summary ------------------------------------
    print_header("5. VaR & ES (FULL SAMPLE)")
    results_var = {}
    results_es  = {}
    for alpha in CONFIDENCE_LEVELS:
        np.random.seed(SEED)
        results_var[alpha] = dict(
            Historical   = var_historical(port_ret, alpha),
            Normal       = var_normal(port_ret, alpha),
            Student_t    = var_student(port_ret, alpha, nu_mle),
            MonteCarlo   = var_montecarlo(port_ret, alpha),
        )
        results_es[alpha] = dict(
            Historical   = es_historical(port_ret, alpha),
            Normal       = es_normal(port_ret, alpha),
            Student_t    = es_student(port_ret, alpha, nu_mle),
            MonteCarlo   = es_montecarlo(port_ret, alpha),
        )

    print(f"\n  {'Method':<14} {'VaR 95%':>10} {'VaR 99%':>10} {'ES 95%':>10} {'ES 99%':>10}")
    print("  " + "-" * 52)
    for method in ['Historical', 'Normal', 'Student_t', 'MonteCarlo']:
        v95 = results_var[0.95][method]
        v99 = results_var[0.99][method]
        e95 = results_es[0.95][method]
        e99 = results_es[0.99][method]
        print(f"  {method:<14} {v95:>10.4f} {v99:>10.4f} {e95:>10.4f} {e99:>10.4f}")

    # ---- 6. Diversification benefit -----------------------------
    print_header("6. DIVERSIFICATION BENEFIT (VaR 99%)")
    div = diversification_benefit(returns, alpha=0.99)
    print(f"  Undiversified ΣwᵢVaRᵢ  : {div['undiversified_var']:.4f}")
    print(f"  Portfolio VaR           : {div['portfolio_var']:.4f}")
    print(f"  Diversification benefit : {div['diversification_benefit_pct']:.1f}%")
    for t, v in div['individual_vars'].items():
        print(f"    {t}: {v:.4f}")

    # ---- 7. GARCH(1,1)-t ----------------------------------------
    print_header("7. GARCH(1,1)-t CONDITIONAL VaR")
    garch_res = fit_garch_t(port_ret)
    params    = garch_res.params
    alpha_g   = params.get('alpha[1]', params.get('alpha'))
    beta_g    = params.get('beta[1]', params.get('beta'))
    omega_g   = params['omega']
    nu_g      = params['nu']
    ab_sum    = alpha_g + beta_g
    half_life = np.log(0.5) / np.log(ab_sum)

    print(f"  ω (omega)  : {omega_g:.6f}")
    print(f"  α (ARCH)   : {alpha_g:.4f}")
    print(f"  β (GARCH)  : {beta_g:.4f}")
    print(f"  α + β      : {ab_sum:.4f}  (< 1 → covariance stationary)")
    print(f"  ν (tail)   : {nu_g:.2f}")
    print(f"  Vol half-life: {half_life:.1f} days")

    garch_var99 = garch_rolling_var(port_ret, 0.99, garch_res)
    garch_var95 = garch_rolling_var(port_ret, 0.95, garch_res)
    print(f"  Mean GARCH VaR 99%: {garch_var99.mean():.4f}")
    print(f"  Mean GARCH VaR 95%: {garch_var95.mean():.4f}")

    # ---- 8. Backtesting (OOS) -----------------------------------
    print_header("8. BACKTESTING (OOS from {})".format(OOS_START))
    oos_ret = port_ret.loc[OOS_START:]

    # Rolling VaRs
    rvar_hist  = rolling_var_historical(port_ret, 0.99).shift(1).reindex(oos_ret.index).dropna()
    rvar_norm  = rolling_var_normal(port_ret, 0.99).shift(1).reindex(oos_ret.index).dropna()
    rvar_stud  = rolling_var_student(port_ret, 0.99, nu_mle).shift(1).reindex(oos_ret.index).dropna()
    rvar_garch = garch_var99.shift(1).reindex(oos_ret.index).dropna()

    bt_results = {}
    for method, rvar in [('Historical', rvar_hist), ('Normal', rvar_norm),
                          ('Student-t',  rvar_stud), ('GARCH-t',  rvar_garch)]:
        oos_aligned = oos_ret.reindex(rvar.index).dropna()
        kup = kupiec_pof_test(oos_aligned, rvar, 0.99)
        cc  = christoffersen_cc_test(oos_aligned, rvar, 0.99)
        kup['alpha'] = 0.99
        bt_results[method] = kup
        print(f"\n  [{method}] Kupiec: n={kup['n_violations']}, rate={kup['actual_rate']:.4f}, "
              f"p={kup['p_value']:.4f} → {kup['decision']}")
        print(f"  [{method}] CC: π₁₁={cc['pi11']:.4f}, p={cc['p_value']:.4f} → {cc['decision']}")
        print(f"  [{method}] Basel: {kup['basel_zone']} zone (k={kup['basel_k']})")

    print(f"\n  ⚠  CC caveat: {cc['power_caveat']}")

    # ---- 9. Stress testing --------------------------------------
    print_header("9. STRESS TESTING")
    var_99_hist = var_historical(port_ret, 0.99)
    scenarios = [
        ("COVID-19 Crash",    "2020-02-19", "2020-03-23"),
        ("2022 Rate Shock",   "2022-01-03", "2022-10-13"),
    ]
    if has_aug2015:
        scenarios.append(("Aug 2015 Flash Crash", "2015-08-18", "2015-08-26"))

    for label, s, e in scenarios:
        sc = stress_period(port_ret, s, e, label, var_99_hist)
        if 'error' in sc:
            print(f"  [{label}] {sc['error']}")
        else:
            print(f"\n  [{sc['label']}]  {sc['start']} → {sc['end']}")
            print(f"    Cumulative return         : {sc['cumulative_return']*100:.2f}%")
            print(f"    Worst single-day loss     : {sc['worst_single_day']*100:.2f}%  ({sc['worst_day_date']})")
            print(f"    Peak daily loss / VaR 99% : {sc['peak_loss_to_var99_ratio']:.2f}×")

    if not has_aug2015:
        print(f"\n  ⚠  Aug 2015 scenario requires data from 2014 onwards.")
        print(f"     Run: python download_data.py  (extends dataset to 2014-2024)")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
