# Methodology

Every choice below is implemented in `src/var_es/` and covered by `tests/`. The numbers
quoted in the README come from `scripts/run_results.py`.

## 1. Data and portfolio

* **Instruments.** SPY (US equities), TLT (long Treasuries), GLD (gold), EEM (emerging-market
  equities), LQD (investment-grade credit). Daily adjusted closes from Yahoo Finance, from
  2004-11-18 (first GLD trading day) to 2025-12-31: 5,313 dates, no gaps after the inner join.
* **Frozen snapshot.** Yahoo revises adjusted closes after each distribution, so two downloads
  of the same window differ. The snapshot is committed with a SHA-256 manifest
  (`data/MANIFEST.json`) and `load_prices` refuses a file whose digest differs. `.gitattributes`
  stops git from converting line endings, which would change the digest on Windows.
* **Returns.** Simple returns $R_{i,t} = P_{i,t}/P_{i,t-1} - 1$. With weights restored at
  every close, the portfolio return is exactly $r_{p,t} = \sum_i w_i R_{i,t}$ and the loss is
  $L_t = -r_{p,t}$, as a fraction of portfolio value. Summing weighted *log* returns is not the
  portfolio log return and biases the tail; this is tested.

## 2. Forecasting protocol

* One-day horizon, levels $\alpha \in \{99\,\%, 97.5\,\%\}$.
* **No look-ahead.** The forecast for date $t$ receives the asset returns of dates
  $\le t-1$ as a read-only array. A property test perturbs every return from a given date on
  and checks that no earlier forecast changes, for each of the five models.
* **Estimation window** $W = 500$ trading days (about two years) for every window-based
  estimate; first forecast on 2006-11-14, last on 2025-12-31 (4,812 forecasts). $W = 250$
  is run as a sensitivity.
* **Re-estimation calendar.** Slow parameters (Student-t degrees of freedom, GARCH
  coefficients) are re-estimated every 21 trading days on a calendar fixed by dates only;
  fast parameters (location, scale, correlations, GARCH filter) are updated daily.

## 3. Models

| Model | Predictive loss distribution for day $t$ |
|---|---|
| Historical simulation | Empirical distribution of the last $W$ portfolio losses |
| Normal | $N(\hat\mu, \hat\sigma^2)$, sample moments of the last $W$ returns |
| Student-t | $\mu + s\,t_\nu$; $\nu$ by profile maximum likelihood (every 21 days), $(\mu, s)$ by EM at fixed $\nu$ (daily) |
| FHS-GARCH | $-(\mu + \sigma_{t} z_s)$, $\sigma_t$ from a GARCH(1,1) filter, $z_s$ the last $W$ standardised residuals |
| Monte Carlo | 20,000 scenarios: Student-t marginal per asset + Gaussian copula (normal-scores correlation), common random numbers |

**FHS-GARCH.** $r_t = \mu + \varepsilon_t$, $\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 +
\beta\sigma_{t-1}^2$, estimated by Gaussian quasi-maximum likelihood (`arch`) on the expanding
history (500 points are too few for a stable GARCH fit). The filter is the same recursion as
`arch` (tested to $10^{-8}$ once the start-up value has decayed). Barone-Adesi et al. (1999);
Hull and White (1998).

**Monte Carlo.** Each asset keeps its own tail thickness $\nu_i$; dependence is a Gaussian
copula with correlation estimated from van der Waerden normal scores, which is rank-based and
invariant to the marginals (tested). The same standard-normal draws are reused every day, so
the VaR path reflects the data, not simulation noise. Seed-to-seed dispersion is reported in
`results/mc_noise.csv`.

## 4. Risk-measure estimators

* **Empirical VaR**: Hyndman-Fan type-6 quantile, $h = (n+1)\alpha$. For i.i.d. data a new
  draw exceeds it with probability close to $1-\alpha$. The plain inverse empirical CDF
  would use the 6th largest of 500 losses at 99 % and raise the expected exception rate from
  1 % to 6/501 = 1.20 % (tested by simulation).
* **Empirical ES**: Acerbi-Tasche tail mean with a fractional last observation,
  $\widehat{ES}_\alpha = \frac{1}{m}\big(\sum_{i \le \lfloor m \rfloor} L_{[i]} +
  (m - \lfloor m \rfloor) L_{[\lfloor m \rfloor + 1]}\big)$, $m = n(1-\alpha)$. Exactly the ES
  of the empirical distribution; sub-additive (tested, with the textbook counter-example
  where VaR is not).
* **Closed forms**: normal and Student-t VaR / ES, checked against numerical integration of
  the quantile function.

## 5. Backtests

* **Kupiec (1995) POF**: $LR_{uc}$, $\chi^2_1$. Zero exceptions is a rejection, not a pass
  ($LR_{uc} = -2T\ln(1-p)$). Non-rejection regions reproduce Jorion's table
  (T = 510: 2-10; T = 1,000: 5-16 exceptions at 99 %).
* **Christoffersen (1998)**: independence against a first-order Markov chain,
  $LR_{ind} \sim \chi^2_1$, and conditional coverage $LR_{cc} = LR_{uc}^{(T-1)} + LR_{ind}
  \sim \chi^2_2$ (the identity is exact on the $T-1$ transitions and is tested).
  `ind_share` $= LR_{ind}/LR_{cc}$ says whether a rejection comes from the *number* or the
  *timing* of exceptions.
* **Clustering within 10 days** (descriptive): share of exceptions that follow the previous
  one within 10 trading days, against $1-\alpha^{10}$ for i.i.d. hits. Christoffersen's test
  only sees exceptions on consecutive days; crisis exceptions typically bunch over weeks.
* **Basel traffic light (BCBS, 1996)**: rolling count of 99 % exceptions over the last 250
  days; green 0-4, yellow 5-9, red 10+ (cumulative binomial probability below 95 %, below
  99.99 %, above). At 97.5 % the same two probabilities give green 0-10, yellow 11-16, red 17+;
  this is a convention of the study, not a regulatory rule.
* **Acerbi-Székely (2014) $Z_2$** at 97.5 %:
  $Z_2 = 1 - \frac{1}{T(1-\alpha)}\sum_t L_t\mathbf 1\{L_t > VaR_t\}/ES_t$, zero in
  expectation for a correct model, negative when tail risk is under-estimated. The null
  distribution is simulated under *each model's own* predictive distribution (5,000 paths,
  drawn day by day inside the engine), so the p-value is exact for that model, including the
  discreteness of the empirical models. The simulation reproduces the published 5 % threshold
  ($\approx -0.70$ for $T = 250$).

## 6. Limitations (what the study does not do)

* One-day horizon, linear portfolio of five ETFs, no intraday or liquidity effects; ETF closes
  can deviate from NAV in stress (LQD fell 9.1 % and rebounded 9.8 % on 29-30 September 2008).
* One data vendor; the pre-2014 part of the snapshot is not cross-checked against a second
  source (2014-2024 returns match an independent earlier download to $2\cdot10^{-6}$).
* Asymptotic $\chi^2$ p-values for Kupiec and Christoffersen; with about 48 expected 99 %
  exceptions in 4,812 days the approximation is adequate on the full sample, not on short
  crisis windows (crisis tables report counts, not p-values, except for $Z_2$ whose p-values
  are simulated).
* Backtests penalise too many exceptions, not excess conservatism: a model that over-reacts
  after a shock (FHS in late 2008) is not rejected for the capital it would have tied up.
* Gaussian copula: no tail dependence by construction. GARCH(1,1) is symmetric (no leverage
  term). Both are deliberate simplifications.

## References

* Acerbi, C. and Székely, B. (2014). Backtesting Expected Shortfall. *Risk*, December.
* Acerbi, C. and Tasche, D. (2002). On the coherence of Expected Shortfall. *Journal of Banking & Finance* 26(7).
* Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets* 19(5).
* Basel Committee on Banking Supervision (1996). Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements.
* Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics* 31(3).
* Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review* 39(4).
* Hull, J. and White, A. (1998). Incorporating volatility updating into the historical simulation method for value-at-risk. *Journal of Risk* 1(1).
* Hyndman, R. J. and Fan, Y. (1996). Sample quantiles in statistical packages. *The American Statistician* 50(4).
* Jorion, P. (2007). *Value at Risk*, 3rd ed. McGraw-Hill.
* Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives* 3(2).
* Liu, C. and Rubin, D. B. (1995). ML estimation of the t distribution using EM and its extensions, ECM and ECME. *Statistica Sinica* 5.
* McNeil, A., Frey, R. and Embrechts, P. (2015). *Quantitative Risk Management*, revised ed. Princeton University Press.
