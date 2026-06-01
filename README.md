# Market Risk: VaR & Expected Shortfall Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A complete, production-style market risk pipeline implemented from scratch in Python.  
Built as a quantitative finance portfolio project targeting buy-side and sell-side risk roles.

---

## Portfolio

| ETF | Exposure | Asset Class |
|-----|----------|-------------|
| SPY | 30% | US Large-Cap Equity |
| TLT | 20% | US Long-Term Treasury |
| GLD | 25% | Gold (Inflation hedge) |
| EEM | 15% | Emerging Market Equity |
| LQD | 10% | US Investment Grade Credit |

---

## Methods

| Method | Type | Notes |
|--------|------|-------|
| Historical Simulation | Non-parametric | Rolling 252-day window |
| Normal (Parametric) | Parametric | Gaussian tails |
| Student-t | Parametric | MLE ν via 3-param joint optimization |
| Monte Carlo | Simulation | N = 50,000, Normal innovations |
| GARCH(1,1)-t | Conditional | Volatility-adaptive, Student-t innovations |

---

## Key Results (2018–2024 sample)

| Metric | Value |
|--------|-------|
| Annualised Return | 5.90% |
| Annualised Volatility | 10.80% |
| Excess Kurtosis | 9.48 (Gaussian rejected, JB p ≈ 0) |
| MLE ν̂ | 4.11  CI 95%: [3.50, 4.71] |
| VaR 99% Historical | 1.60% |
| VaR 99% Student-t | 1.78% |
| Diversification benefit (VaR 99%) | ~43% |
| GARCH: α + β | 0.9808 (vol half-life ≈ 36 days) |

---

## Backtesting (OOS 2019–2024, α = 99%)

| Model | Violations | Kupiec | Basel Zone |
|-------|-----------|--------|------------|
| Historical | 23 | PASS | Green (k = 3.0) |
| Normal | 27 | FAIL | Green (k = 3.0) |
| Student-t | 19 | PASS | Green (k = 3.0) |
| GARCH(1,1)-t | 23 | PASS | Green (k = 3.0) |

All models fail Christoffersen CC (violation clustering during COVID-19 / 2022).  
This is a model-class limitation, not a single-model failure: no static VaR handles regime shifts.

---

## Stress Scenarios

| Event | Period | Cumulative Loss | Peak / VaR 99% |
|-------|--------|----------------|----------------|
| COVID-19 crash | Feb–Mar 2020 | −16.2% | 3.7× |
| 2022 Rate Shock | Jan–Oct 2022 | −22.7% | 2.0× |
| Aug 2015 Flash Crash | Aug 18–26, 2015 | *(requires 2014+ data)* | — |

---

## Repository Structure

```
market-risk-var-es/
├── VaR_ES_Market_Risk.ipynb   ← Narrative notebook (render document)
├── var_es_pipeline.py         ← Pure Python pipeline (executable)
├── download_data.py           ← Data fetch from Yahoo Finance
├── generate_notebook.py       ← Regenerates the .ipynb from scratch
├── requirements.txt
├── .gitignore
└── data/                      ← CSV files (not tracked in git)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download historical data (2014–2024)
python download_data.py

# 3. Run the pipeline
python var_es_pipeline.py

# 4. Or open the narrative notebook
jupyter notebook VaR_ES_Market_Risk.ipynb
```

---

## Technical Highlights

- **MLE for ν**: 3-parameter joint optimization (ν, μ, σ) with profile Hessian CI.  
  Fixing σ=1 on standardized returns is a common pitfall — it biases ν̂ upward by 2–3×.
- **No look-ahead bias**: rolling VaR uses `.shift(1)` before aligning with OOS returns.
- **GARCH-t**: `arch` library, GARCH(1,1) with Student-t innovations.  
  α + β < 1 ensures covariance stationarity.
- **Basel II zones**: exception counts scaled to T=250, multiplier k ∈ {3.0, 3.40, …, 4.0}.
- **Christoffersen CC**: includes explicit power caveat for T≈1500, α=99%.

---

## Extensions (Planned)

- [ ] GJR-GARCH (leverage effect: σ↑ more after negative returns)
- [ ] Filtered Historical Simulation (FHS) — Basel FRTB IMA standard
- [ ] DCC-GARCH (dynamic correlations — resolves 2022 SPY/TLT breakdown)
- [ ] EVT/GPD (Extreme Value Theory beyond 99% quantile)
- [ ] t-Copula Monte Carlo (non-linear tail dependence)

---

## Author

**Tancrede Gaignard** · [github.com/tancredeg](https://github.com/tancredeg)
