# PROGRESS — market-risk-var-es

## 2026-10-13 — Phase 0 (préliminaire)

Fait
- Audit des données (5 CSV, 2014-01-02 → 2024-12-30, 2 767 lignes, aucune valeur manquante).
- Réplication indépendante des chiffres des figures : `scripts/phase0_replication.py`.
- `AUDIT.md` : 2 critiques, 7 majeurs, 5 mineurs ; écarts avec la spec ; plan des phases 1–6.
- `scripts/download_data.py` : téléchargement 2004-11-18 → 2025-12-31, snapshot figé, manifeste SHA-256.

Non fait
- Lecture du code du repo (inaccessible depuis la session).

Chiffres de référence (convention actuelle du repo, W = 250, OOS 2019–2024, VaR 99 %)
- Historique N = 23, Kupiec p = 0,057, IND p = 0,048 ; Normale N = 27, p = 0,006.
- Région de non-rejet de Kupiec à T = 1 509 : [9, 23].

Décisions ouvertes : D1–D10 (`AUDIT.md` §4).

Suite
1. Vérifier le repo public (C1) ; corriger tout de suite s'il affiche le build biaisé.
2. Exécuter `scripts/download_data.py` en local ; committer `data/`.
3. Terminer l'audit du code dans Claude Code ; valider D1–D10.
4. Phase 1.
