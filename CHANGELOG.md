# Changelog

## v1.0.0 — 2026-10-14

First public version, rewritten from scratch.

- Five one-day VaR / ES models: historical simulation, normal, Student-t (maximum likelihood),
  GARCH(1,1)-filtered historical simulation, Monte Carlo with Student-t marginals and a
  Gaussian copula.
- Rolling out-of-sample engine with no look-ahead (property-tested), fixed re-estimation
  calendar, simulated Acerbi-Székely null distribution.
- Backtests: Kupiec POF, Christoffersen independence and conditional coverage, rolling Basel
  traffic light (250 days), Acerbi-Székely Z2 at 97.5 %.
- Frozen 2004-2025 price snapshot with SHA-256 manifest; out-of-sample period 2006-11-14 to
  2025-12-31 (4,812 days); sensitivity with a 250-day window.
- 126 tests (known laws, published reference values, regression sample), 99 % coverage,
  ruff, strict mypy, GitHub Actions CI.
