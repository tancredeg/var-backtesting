# PROGRESS — var-backtesting

## 2026-10-27 — Intégration dans `tancredeg/var-backtesting`, audits, coût en capital

Fait
- Ancien code (`dd4ff65`, juin 2026) enfin lu : 11 constats avec numéros de ligne (`AUDIT.md` §9).
  Le README public affichait encore le build biaisé (Student-t « 19 / PASS », zones toutes vertes).
- v1 rejouée par-dessus `origin/main` sans réécrire son historique : commit de retrait de l'ancien
  pipeline, puis les commits v1, auteur `282603140+tancredeg@users.noreply.github.com`.
- Test de régression : tolérances tirées d'un plancher de bruit mesuré (bruit 1e-15 sur les prix →
  6e-5 relatif sur le Monte Carlo, < 2e-8 ailleurs) ; comptages exacts.
- CI : job Windows ajouté ; checkout CRLF simulé en local, 126/126 tests verts.
- Audit des données (`scripts/audit_data.py`) : clôtures = téléchargement frais à 5e-10, toutes dans
  [plus bas, plus haut] ; ajustements de dividendes plausibles ; SPY vs S&P 500 : écart médian
  10,6 pb (2004–2009) puis 2,7 pb, autocorrélation −0,37 (effet d'heure de clôture, pas d'erreur).
- Coût en capital Bâle 1996 (`results/capital_charge.csv`) : FHS 17,2 % en moyenne contre 19,2 %
  pour l'historique, mais procyclique (43,5 % de sept. 2008 à mars 2009, pic = 4,5 × la médiane).
- 131 tests, couverture 99 %, ruff, mypy strict. Résultats existants régénérés à l'identique.

- Références alignées sur la vérification documentaire du 2026-10-27 (seuils Z2 valables pour
  ν ≥ 5 ; multiplicateur 3 = minimum réglementaire ; heures de clôture des ETF sourcées ; mention
  « décote de LQD » retirée).
- Commits signés (committer Claude), auteur Tancrède Geli.

Poussé le 2026-10-28 : `main` (avance rapide depuis `dd4ff65`, sans force push) et tag `v1.0.0`.

Suite
1. Vérifier que la CI passe (Linux 3.12 / 3.13 et Windows).
2. Release v1.0.0 (texte : `CHANGELOG.md`), épinglage du repo, post LinkedIn.


## 2026-10-14 — Phases 1 à 6 (session unique)

Fait
- Snapshot 2004-11-18 → 2025-12-31 (5 313 dates) + manifeste SHA-256 ; extrait 2019–2020 dans `data/sample/`.
- Package `src/var_es/` : 5 modèles (historique, normale, Student-t, FHS-GARCH, MC copule),
  moteur glissant sans look-ahead, Kupiec, Christoffersen IND/CC, zones de Bâle glissantes, Z2.
- 126 tests (lois connues, cas de référence Jorion / Bâle 1996 / Acerbi–Székely, non-look-ahead,
  régression), couverture 99 % ; ruff, mypy strict ; workflow GitHub Actions.
- `scripts/run_results.py` (462 s) : `results/*.csv`, 3 figures, tableau du README réécrit.
- README (ordre imposé), `docs/methodology.md`, notebook de démo (< 5 s), CHANGELOG, tag v1.0.0.

Chiffres clés (W = 500, hors échantillon 2006-11-14 → 2025-12-31, 4 812 jours)
- VaR 99 % : historique 50 exceptions (48,1 attendues, Kupiec p = 0,79), FHS 46 (p = 0,76),
  normale 85, Student-t 69, MC 73. Indépendance rejetée pour les 5 (p < 0,001).
- Exceptions à moins de 10 jours de la précédente : 43–49 % (non conditionnels) vs 20 % (FHS) ; i.i.d. 9,6 %.
- Zones de Bâle : 4 modèles rouges dès oct. 2008 (pic 11–19) ; FHS jamais rouge (pic 9).
- Z2 97,5 % : rejet normale, t, MC (p ≤ 0,0012) ; historique p = 0,060 ; FHS p = 0,22 ;
  fév.–avr. 2020 : rejet des 5 (p ≤ 0,0004).

Décisions : D1–D10 de `AUDIT.md` appliquées telles que recommandées.

Suite (côté Tancrède, en local)
1. Créer / vider le repo GitHub `market-risk-var-es`, pousser `main` et le tag `v1.0.0`.
2. Vérifier que le badge CI passe au vert ; épingler le repo sur le profil.
3. Publier la release v1.0.0 avec le texte de `CHANGELOG.md` ; publier le post LinkedIn.
4. Option : retirer les lignes `Co-Authored-By` des commits si non souhaitées.
