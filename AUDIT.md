# AUDIT — Repo 1 : VaR / Expected Shortfall backtestées

Phase 0 du standard « repo recruteur ». Date : 2026-10-13. Mise à jour : 2026-10-14 (§8).
Statut : **CLOS** — l'ancien code n'a jamais pu être lu (voir §0) ; il est remplacé par une
réécriture complète (décision D1). La résolution de chaque constat est en §8.
Statut d'origine : **PRÉLIMINAIRE** — le code du repo n'a pas pu être lu (voir §0). Les constats ci-dessous
portent sur les données, les figures produites par le repo, l'historique des audits précédents et
une réplication indépendante. Tous les chiffres cités sont imprimés par
`scripts/phase0_replication.py` (section indiquée entre crochets, ex. [B]).

---

## 0. Périmètre et méthode

| Élément | Statut |
|---|---|
| Code (`var_es_pipeline.py`, `generate_notebook.py`, notebook) | **Non lu.** Repo `tancredeg/market-risk-var-es` inaccessible depuis la session (application GitHub non liée au compte). |
| Données (5 CSV, Adj Close) | Lues et contrôlées [A]. |
| Figures 01–07 | Lues ; chiffres affichés recoupés par réplication [B][C][D]. |
| Historique d'audit (mai–juillet 2026) | Repris des sessions précédentes ; non revérifié sur le code. |

Pour clore la Phase 0 : relancer l'audit du code dans Claude Code sur le repo cloné (section §2
« À vérifier dans le code »), puis valider les décisions §4.

---

## 1. Synthèse

1. Les figures du Project sont celles du **build antérieur à l'audit de juillet** : look-ahead du Student-t
   (N = 19 sur la fig. 06) et capitalisation arithmétique des log-rendements (courbes finissant à ≈ 1,89).
   Si le repo public affiche encore ces figures, il expose un défaut éliminatoire.
2. La spécification exige un zoom 2008 ; les données commencent le 2014-01-02. Prérequis bloquant :
   re-télécharger depuis le 2004-11-18 (première cotation de GLD).
3. Écart méthodologique avec la spec : pas de FHS, pas de niveau 97,5 %, pas de backtest d'ES,
   zones de Bâle calculées par mise à l'échelle au lieu de fenêtres de 250 jours.
4. Les verdicts actuels (« passe / échoue ») tiennent à une exception près : l'historique est à
   p = 0,057 avec N = 23, borne haute exacte de la région de non-rejet [9, 23] [C][D].
5. Recommandation : réécrire le cœur de calcul en package testé plutôt que refactorer le pipeline
   monolithique (estimation : plus de la moitié de la méthodologie change). À confirmer après lecture du code.

---

## 2. Défauts classés par gravité

### Critiques

**C1 — Figures issues d'un build biaisé.**
- Fig. 06 : Student-t N = 19 (Kupiec p = 0,331 [D]) = résultat avec ν̂ plein échantillon injecté dans la
  fenêtre OOS. Après correction : 24 (audit de juillet) ; 23 dans la réplication indépendante
  [C, `--with-student`] → p = 0,034 ou 0,057.
- Fig. 02 et 07 : valeur finale ≈ 1,89. Or `cumprod(1 + Σ wᵢ rᵢ)` donne **1,888** [B], contre
  exp(Σ r) = 1,985 avec la convention du repo et 2,169 pour le portefeuille réellement rebalancé.
  Signature du défaut « arithmetic compounding of log-returns » réputé corrigé.
- Action immédiate (10 min) : ouvrir le README et les figures du repo public ; si N = 19 / « Student-t
  PASS » y figure, corriger avant tout envoi de candidature.

**C2 — Fenêtre de données incompatible avec la spec.**
Données 2014-01-02 → 2024-12-30, 2 767 lignes [A]. Le zoom 2008 est impossible. Avec une fenêtre
d'estimation de 500 jours, le premier jour hors échantillon possible tombe vers novembre 2006 si l'on
part du 2004-11-18. Le téléchargement ne peut pas se faire depuis un environnement sandbox (Yahoo bloqué) :
exécution locale de `scripts/download_data.py`.

### Majeurs

**M1 — Agrégation du portefeuille.** Le repo somme des log-rendements pondérés (ν̂ = 3,9986 et
moyenne 6,25 % reproduits exactement [B]). Ce n'est pas le log-rendement du portefeuille :
écart maximal 0,27 point le 2020-03-16 (−4,65 % contre −4,38 %), soit 6 % d'erreur relative sur une
observation de queue ; biais de moyenne annuelle 0,8 point (6,25 % contre 7,05 %). Le rebalancement
quotidien implicite des poids fixes n'est documenté nulle part.
→ Rendement simple du portefeuille, rebalancement quotidien explicite, VaR en % de la valeur.

**M2 — Verdicts binaires fragiles.**
- Historique : N = 23, Kupiec p = 0,057 ; une exception de plus bascule en rejet [C].
- Student-t : 23 ou 24 selon l'optimiseur (bornes de ν, tolérance) → verdict opposé.
- Christoffersen indépendance (historique) : p = 0,048 avec n₁₁ = 2 [C] ; deux paires
  d'exceptions consécutives décident le test.
→ Publier des p-values, jamais PASS/FAIL seul ; figer l'optimiseur ; test de régression sur le
comptage.

**M3 — Fenêtre d'estimation de 250 jours.** La VaR historique 99 % interpole entre la 2ᵉ et la
3ᵉ pire observation ; l'ES historique 97,5 % moyenne ≈ 6 observations. Variance d'estimation
dominante. → W = 500 commune à toutes les méthodes ; W = 250 en sensibilité.

**M4 — Zones de Bâle mal appliquées.** Le repo met à l'échelle le comptage total vers T = 250
(historique de juillet). Les seuils binomiaux de Bâle (1996) valent pour exactement 250
observations. → Fenêtres glissantes de 250 jours (exigé par la spec).

**M5 — GARCH : diagnostic non démontré et méthode différente de la spec.** L'explication « échec
dû à la calibration sur 2014–2018 calme » est une hypothèse non testée ; si les paramètres sont
figés, c'est un choix de conception à corriger (ré-estimation glissante), pas un résultat. La spec
demande une simulation historique filtrée (FHS), pas un GARCH-t paramétrique. La fig. 05 (bas)
superpose une VaR GARCH in-sample aux pertes : à retirer ou étiqueter « in-sample ».

**M6 — Monte Carlo redondant.** Pour un portefeuille linéaire, un MC normal (ou t multivarié à ν
commun) reproduit la VaR paramétrique au bruit de simulation près : aucune information
nouvelle. → Marges Student-t par actif + copule (décision D6).

**M7 — Reproductibilité des données.** `data/*.csv` était ignoré par git ; l'Adj Close de Yahoo est
révisé rétroactivement. → Snapshot figé versionné + manifeste SHA-256 (`scripts/download_data.py`)
et extrait figé dans `data/sample/` pour les tests de régression.

### Mineurs

- **m1** — Dernière date 2024-12-30 : le 2024-12-31 manque (borne `end` exclusive de yfinance).
- **m2** — Fig. 04 : densités tracées sur [−0,037 ; 0,029] alors que les rendements vont de −0,059
  à 0,048 ; les queues que la figure doit montrer sont coupées. ν affiché « 4.00 » (3,9986) :
  ressemble à un paramètre fixé ; afficher 4,00 [3,54 ; 4,46].
- **m3** — Stress tests, corrélation SPY/TLT, diversification : hors des 2–4 résultats clés de la
  spec → notebook ou suppression, pas README.
- **m4** — Puissance du test de Christoffersen à 99 % : très faible avec ≈ 15 exceptions attendues
  sur 1 509 jours. À dire explicitement ; le niveau 97,5 % donne plus de puissance.
- **m5** — Temps de calcul : MLE Student-t glissante ≈ 4 min pour 1 509 jours ; ≈ 13 min attendues
  sur ≈ 4 800 jours OOS (2007–2025). Incompatible avec la CI et le notebook < 2 min.
  → ν ré-estimé tous les 21 jours, résultats mis en cache, notebook lisant `results/`.

### À vérifier dans le code (non lu)

- Alignement temporel : VaR_t estimée sur r_{t−W..t−1} uniquement (pas de `rolling().quantile()`
  aligné sur t inclus).
- Définition du quantile historique (`np.quantile` interpolé vs statistique d'ordre) et de l'ES
  (inégalité stricte, cas où aucune observation ne dépasse la VaR).
- Graine du Monte Carlo, nombre de tirages, erreur standard.
- Convention de signe (pertes positives) cohérente entre VaR, ES et tests.
- Dépendances non épinglées (`requirements.txt` en `>=`).

---

## 3. Écarts avec la spécification Repo 1

| Exigence | État actuel | Écart | Phase |
|---|---|---|---|
| Historique | Oui, W = 250, 99 % | W, 97,5 % | 1–4 |
| Paramétrique normale, Student-t | Oui, 99 % | 97,5 %, ES, cadence de ré-estimation | 1–4 |
| FHS GARCH(1,1) | Non (GARCH-t paramétrique) | Méthode à créer | 1 |
| Monte Carlo | Oui (redondant) | Refonte (D6) | 1 |
| Tests sur lois connues | Non | Tout | 2 |
| Tests statistiques sur cas de référence | Non | Tout | 2 |
| Kupiec, Christoffersen IND/CC, 99 % et 97,5 % | 99 % seulement | 97,5 % | 4 |
| Zones de Bâle, fenêtres glissantes 250 j, zooms 2008 / mars 2020 / 2022 | Mise à l'échelle, pas de 2008 | Données + calcul | 0bis, 4 |
| Backtest ES 97,5 % (Acerbi–Székely) | Non | Tout | 1, 2, 4 |
| Tableau méthode × taux × p-values × zone | Partiel | Refonte | 4 |
| README : quelle méthode échoue en crise et pourquoi | Narratif non fondé sur la décomposition | Décomposition LR_uc / LR_ind | 4–5 |

---

## 4. Décisions à valider avant la Phase 1

| # | Décision | Recommandation | Raison |
|---|---|---|---|
| D1 | Réécriture ou refactor | **Réécriture du cœur** dans `src/var_es/`, en portant les briques déjà corrigées (MLE de ν, Kupiec n = 0) ; ancien pipeline supprimé (l'historique git le conserve) | Plus de la moitié de la méthode change (estimation) ; le standard impose fonctions pures + tests |
| D2 | Fenêtre de données | **2004-11-18 → 2025-12-31** (alternative : → 2024-12-31) | 2008 couvert ; 2025 ajoute le choc d'avril 2025 et date le repo de façon crédible en 2026–27 |
| D3 | Rendement du portefeuille | **Rendement simple, rebalancement quotidien**, VaR et ES en % de la valeur | Corrige M1 ; convention P&L standard |
| D4 | Fenêtre d'estimation | **W = 500** pour toutes les méthodes ; OOS à partir du 2007-01-02 ; W = 250 en sensibilité | M3 ; compatible avec C2 |
| D5 | FHS | GARCH(1,1), moyenne constante, QMLE gaussienne ; **fenêtre expansive (min. 500), ré-estimation tous les 21 jours** ; quantiles empiriques des 500 derniers résidus standardisés | GARCH instable sur 500 points ; aucun look-ahead |
| D6 | Monte Carlo | **Marges Student-t par actif + copule gaussienne** (corrélation sur W), 20 000 tirages, graine fixe, erreur standard publiée ; copule t en option | Seule variante qui n'est pas un doublon du paramétrique |
| D7 | Cadence Student-t | ν ré-estimé tous les 21 jours, location/échelle quotidiennes à ν fixé | m5 |
| D8 | Zones | Bâle 1996 sur VaR 99 % (fenêtres 250 j) ; 97,5 % comparé au seuil FRTB desk (> 30 exceptions / 250 j), présenté comme indicatif | Les zones de Bâle ne sont définies qu'à 99 % — seuil FRTB à vérifier dans le texte MAR32 |
| D9 | Périmètre README | 4 résultats de la spec uniquement ; stress tests, corrélation, diversification hors README | m3 |
| D10 | GARCH-t paramétrique | **Remplacé** par FHS (5 méthodes, comme la spec) | Lisibilité ; une méthode conditionnelle suffit pour le récit |

---

## 5. Plan des phases 1–6 adapté

Estimation totale ≈ 19–21 h, soit ≈ 3 semaines à 6–8 h/semaine, hors Phase 0bis.

**Phase 0bis — Prérequis (≈ 45 min, en local).**
Vérifier le repo public (C1). `python scripts/download_data.py` ; commit de `data/` + `MANIFEST.json`.
Copier `AUDIT.md`, `CLAUDE.md`, `PROGRESS.md` dans le repo ; terminer l'audit du code dans Claude Code.

**Phase 1 — Structure (≈ 4 h).**
```
src/var_es/
  config.py          dataclass figée : poids, niveaux, W, cadences, graines
  data.py            lecture du snapshot, contrôle SHA-256 contre MANIFEST.json
  portfolio.py       rendements simples, rebalancement quotidien
  models/            historical.py, parametric.py (normale, t), fhs.py, montecarlo.py
                     → chaque modèle expose fit(window) et predict(level) -> (VaR, ES)
  forecast.py        moteur glissant : pour chaque t, information ≤ t−1 ; cadence de ré-estimation
  backtest/          coverage.py (Kupiec, Christoffersen IND/CC), traffic_light.py, es_test.py (Z1, Z2)
  plotting.py
scripts/  download_data.py, run_results.py
tests/  notebooks/demo.ipynb  data/{raw,sample}/  docs/
```
`pyproject.toml` épinglé (numpy, pandas, scipy, arch, matplotlib ; extra `[data]` = yfinance ;
extra `[dev]` = pytest, pytest-cov, ruff, mypy, nbconvert). Sortie : `pip install -e .[dev]` passe.

**Phase 2 — Tests (≈ 5 h).**
- Lois connues : i.i.d. N(0, σ) → VaR historique dans ±3 erreurs standard du quantile théorique ;
  t(4) mis à l'échelle → VaR et ES paramétriques = formules fermées, ν̂ dans son IC ; GARCH(1,1)-t
  simulé à paramètres connus → couverture FHS non rejetée et VaR conditionnelle corrélée à σ_t q.
- Garde-fou anti look-ahead : modifier r_s pour s ≥ t ne change pas VaR_t (test de propriété).
- Cas de référence : table de Bâle (§7) ; régions de Kupiec de Jorion (§7) ; n = 0 rejeté ;
  Christoffersen sur séquences construites à la main (dont n₁₁ = 0) ; Z2 de taille correcte sous
  H0 et puissant quand l'ES est sous-estimée de 20 %.
- Taille empirique de Kupiec sous H0 par simulation ≈ 5 % (discrétisation documentée).
- Régression : `data/sample/` (2019–2020) figé, séries de VaR et statistiques avec tolérances.
- Couverture ≥ 80 % sur `src/var_es` hors `plotting.py`.

**Phase 3 — Qualité et CI (≈ 1 h).** ruff (lint + format), mypy sur `src/`, GitHub Actions
(Python 3.11 et 3.12, lint + tests), badge.

**Phase 4 — Résultats (≈ 5 h, calcul compris).** `scripts/run_results.py` → `results/*.csv` + figures :
1. Kupiec, Christoffersen IND et CC par méthode × {99 %, 97,5 %}, avec décomposition LR_uc / LR_ind.
2. Zones de Bâle glissantes (250 j), bande de couleur par méthode, zooms 2008, mars 2020, 2022.
3. Z2 d'Acerbi–Székely à 97,5 %, p-values par simulation sous la loi prédictive de chaque modèle.
4. Tableau final méthode × taux d'exceptions × p-values × zone (pire zone, % de jours en rouge).
Chaque résultat interprété en une phrase dérivée des chiffres.

**Phase 5 — README (≈ 2 h).** Ordre imposé par le standard, en anglais. Section « What fails in a
crisis, and why » écrite à partir de la décomposition LR_uc / LR_ind, pas avant.

**Phase 6 — Livraison (≈ 1,5 h).** Notebook de démonstration lisant `results/` (< 2 min), tag v1.0,
notes de version, épinglage, brouillon LinkedIn 5 lignes.

---

## 6. Hypothèses de résultats (à confirmer ou réfuter en Phase 4, pas à écrire d'avance)

- H1 : les méthodes non conditionnelles (historique, normale, t, MC) rejettent l'indépendance
  autour de 2008 et de mars 2020 ; FHS ne la rejette pas.
- H2 : à 99 %, l'échec dominant reste la couverture (LR_uc), comme dans l'audit de juillet pour 3
  modèles sur 4 ; le regroupement n'apparaît nettement qu'à 97,5 %.
- H3 : Z2 rejette l'ES normale et pas l'ES Student-t.
- Si H1 est réfutée, le narratif « échec en crise par regroupement » demandé par la spec est faux
  et doit être remplacé par ce que montrent les chiffres.

---

## 7. Valeurs de référence pour les tests (vérifiées par calcul)

Bâle 1996, T = 250, p = 1 %, P(N ≤ k) : 0 → 8,11 % ; 1 → 28,58 % ; 2 → 54,32 % ; 3 → 75,81 % ;
4 → 89,22 % ; 5 → 95,88 % ; 6 → 98,63 % ; 7 → 99,60 % ; 8 → 99,89 % ; 9 → 99,97 % ;
10 → 99,99 %. Zones : vert 0–4, orange 5–9, rouge ≥ 10.

Kupiec, test à 5 %, région de non-rejet (p = 1 %) : T = 250 → [1, 6] ; T = 510 → [2, 10] ;
T = 1 000 → [5, 16] ; T = 1 509 → [9, 23]. Les deux cas intermédiaires correspondent à la table de
Jorion (*Value at Risk*, 3ᵉ éd.). À p = 2,5 % : T = 250 → [3, 11] ; T = 1 000 → [16, 35].

À vérifier dans les sources avant de les citer dans le README : seuils de Z2 (≈ −0,70 à 5 %,
≈ −1,8 à 0,01 %, Acerbi et Székely, 2014) ; seuils FRTB par desk (12 exceptions à 99 %, 30 à 97,5 %).

---

## 8. Résolution (2026-10-14)

Réécriture complète dans `src/var_es/` (D1), décisions D2–D10 appliquées telles que
recommandées en §4. Chaque ligne renvoie au test qui la verrouille.

| Constat | Résolution | Verrou |
|---|---|---|
| C1 figures du build biaisé | Toutes les figures et tables régénérées par `scripts/run_results.py` ; le tableau du README est réécrit par le script | `tests/test_docs.py` |
| C2 pas de 2008 | Snapshot 2004-11-18 → 2025-12-31 (5 313 dates), téléchargé depuis la session ; rendements 2014–2024 identiques à l'ancien snapshot à 2·10⁻⁶ près | `tests/test_data.py` |
| M1 agrégation | Rendements simples, rebalancement quotidien, identité testée contre un chemin de richesse | `test_portfolio_return_is_exact_under_daily_rebalancing` |
| M2 verdicts fragiles | p-values partout ; optimiseurs déterministes ; régression figée | `tests/test_regression.py` |
| M3 W = 250 | W = 500 (principal), W = 250 en sensibilité (`results/sensitivity_window250.csv`) | — |
| M4 zones mises à l'échelle | Comptages glissants sur 250 jours exactement | `test_rolling_zones_counts_the_last_250_days_including_today` |
| M5 GARCH-t in-sample | FHS-GARCH(1,1), ré-estimé tous les 21 jours sur l'historique expansif, filtre identique à `arch` | `test_garch_filter_matches_arch_conditional_volatility`, test de non-look-ahead |
| M6 MC redondant | Marges t par actif + copule gaussienne (scores normaux), nombres aléatoires communs, bruit publié | `test_monte_carlo_reduces_to_student_t_for_one_asset` |
| M7 reproductibilité | Manifeste SHA-256 vérifié à chaque chargement ; `.gitattributes` contre la conversion CRLF | `test_tampered_file_is_refused` |
| m1 31/12 manquant | Borne de fin inclusive (`end + 1 jour`) | `test_main_snapshot_matches_its_manifest` |
| m2 fig. 04 | Figure supprimée (hors des résultats clés) | — |
| m3 périmètre README | 4 résultats de la spec seulement | — |
| m4 puissance à 99 % | Signalée ; le niveau 97,5 % et le Z2 complètent | — |
| m5 temps de calcul | 462 s pour l'étude complète ; notebook < 5 s (lit `results/`) | — |
| Quantile historique | Type 6 (Hyndman–Fan) : taux d'exception nominal ; le type 1 donnerait 1,20 % au lieu de 1 % | `test_type6_quantile_gives_the_nominal_exceedance_probability` |

### Hypothèses du §6 confrontées aux résultats

- **H1 réfutée en partie.** Les quatre méthodes non conditionnelles regroupent leurs exceptions,
  comme prévu, mais FHS rejette aussi l'indépendance à 99 % (p < 0,001) : ses 5 paires
  d'exceptions consécutives tombent toutes dans des ventes brutales (avr. 2013, août 2015,
  fév. et mars 2020, avr. 2025). À 97,5 %, FHS est au niveau de l'i.i.d. sur 10 jours.
- **H2 réfutée.** À 99 %, le regroupement domine : LR_ind = 99 % de LR_cc pour l'historique et
  FHS, 54–76 % pour normale, t et MC.
- **H3 réfutée.** Z2 rejette l'ES Student-t (p = 0,0012), pas seulement la normale.
