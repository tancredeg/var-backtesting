# CLAUDE.md — market-risk-var-es

Instructions permanentes pour Claude Code dans ce repo. À lire en début de séance avec
`AUDIT.md` et `PROGRESS.md`.

## Objectif

Un recruteur quant / risk comprend en 90 s ce que fait le repo et ce qu'il prouve, et peut tout
relancer en 2 commandes. Repo d'appui pour les profils quant / risk : comparer 5 méthodes de
VaR / ES et montrer, chiffres à l'appui, laquelle échoue en crise et pourquoi.

## Spécification

Méthodes : historique ; paramétrique normale ; paramétrique Student-t ; historique filtrée par
GARCH(1,1) (FHS) ; Monte Carlo. Niveaux 99 % et 97,5 %. Horizon 1 jour.
Portefeuille : SPY 30 %, TLT 20 %, GLD 25 %, EEM 15 %, LQD 10 %, rebalancement quotidien.

Résultats (Phase 4) :
1. Kupiec (POF), Christoffersen (indépendance, couverture conditionnelle), par méthode, à 99 % et 97,5 %.
2. Zones « feux tricolores » de Bâle sur fenêtres glissantes de 250 jours ; zooms 2008, mars 2020, 2022.
3. Backtest de l'ES à 97,5 % (Acerbi–Székely, Z2 ; p-values par simulation).
4. Tableau final : méthode × taux d'exceptions × p-values × zone.

Le README dit quelle méthode échoue en crise et pourquoi, à partir de la décomposition
LR_uc / LR_ind — jamais d'un récit écrit à l'avance.

## Invariants (non négociables)

- **Aucun look-ahead** : la prévision pour la date t n'utilise que des données datées ≤ t−1.
  Tout paramètre (ν, GARCH, corrélations) est estimé dans la fenêtre. Test de propriété dédié.
- **Données** : uniquement le snapshot figé `data/` contrôlé contre `data/MANIFEST.json`.
  Aucun appel réseau hors `scripts/download_data.py`.
- **Chiffres** : chaque nombre du README et du notebook sort de `scripts/run_results.py`
  (`results/*.csv`). Jamais de chiffre recopié d'un build antérieur.
- **Verdicts** : publier les p-values ; « pass / fail » seulement à côté de la p-value.
- **Reproductibilité** : graines fixées (`config.py`), dépendances épinglées, aucun chemin absolu.
- **Données publiques uniquement** ; aucune méthode, donnée ou information liée à un stage en cours.
- Conventions : pertes positives ; VaR et ES positifs ; rendements simples du portefeuille ;
  docstrings avec formule et référence ; type hints ; fonctions pures dans `src/var_es/`.

## Standard « repo recruteur » — phases (une par séance)

0. Audit → `AUDIT.md` (fait, préliminaire : terminer l'audit du code).
1. Structure : `src/var_es/`, `tests/`, `notebooks/`, `data/sample/`, `scripts/`, `docs/` ;
   `pyproject.toml` épinglé ; `pip install -e .`.
2. Tests pytest : lois connues, cas de référence, régression sur `data/sample/` ; couverture ≥ 80 %.
3. ruff + mypy ; GitHub Actions (lint + tests à chaque push) ; badge.
4. `scripts/run_results.py` régénère toutes les figures et tables ; 2–4 résultats interprétés.
5. README en anglais : phrase-problème ; résultats (tableau + 1–2 figures) ; méthodologie et
   limites ; reproduire (2 commandes) ; structure ; références.
6. Notebook de démo < 2 min (lit `results/`) ; tag v1.0 ; épinglage ; brouillon LinkedIn 5 lignes.

## Règles de travail

- Plan validé avant toute refonte importante ; décisions consignées dans `PROGRESS.md`.
- Commits atomiques, messages explicites.
- Fin de séance : mettre à jour `PROGRESS.md` (fait, chiffres obtenus, décisions, suite).

## Commandes

```bash
pip install -e ".[dev]"
pytest -q --cov=var_es
python scripts/run_results.py          # régénère results/ et figures/
```
