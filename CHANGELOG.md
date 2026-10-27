# Changelog

## v1.0.0 — 2026-10-27

First public version of the rewrite. It replaces the June 2026 pipeline, whose Student-t and
GARCH backtests used full-sample parameters (see `AUDIT.md`, section 9).

- Five one-day VaR / ES models: historical simulation, normal, Student-t (maximum likelihood),
  GARCH(1,1)-filtered historical simulation, Monte Carlo with Student-t marginals and a
  Gaussian copula.
- Rolling out-of-sample engine with no look-ahead (property-tested), fixed re-estimation
  calendar, simulated Acerbi-Székely null distribution.
- Backtests: Kupiec POF, Christoffersen independence and conditional coverage, rolling Basel
  traffic light (250 days), Acerbi-Székely Z2 at 97.5 %.
- Frozen 2004-2025 price snapshot with SHA-256 manifest; out-of-sample period 2006-11-14 to
  2025-12-31 (4,812 days); sensitivity with a 250-day window.
- Basel 1996 capital charge per model (level, crisis level, procyclicality).
- Data audit script: snapshot closes against a fresh download and daily ranges, dividend
  adjustments, SPY against the S&P 500 index.
- 131 tests (known laws, published reference values, regression sample with tolerances
  derived from a measured cross-platform noise floor), 99 % coverage, ruff, strict mypy,
  GitHub Actions CI on Linux (Python 3.12, 3.13) and Windows.
