# Data Dictionary – creditcard.csv

Source : Kaggle "Credit Card Fraud Detection" (ULB Machine Learning Group, `mlg-ulb/creditcardfraud`).
Transactions par carte bancaire européenne. Les variables V1–V28 sont des composantes PCA
(anonymisées pour confidentialité).

| Colonne | Type | Description | Remarque pipeline |
|---|---|---|---|
| Time | numérique | Secondes écoulées depuis la première transaction du dataset | Dimension temporelle : à étudier pour le choix du split (Phase 4) |
| V1 … V28 | numérique | Composantes principales issues d'une PCA ; sémantique inconnue | Déjà décorrélées par construction ; échelle propre à la PCA |
| Amount | numérique | Montant de la transaction | Très asymétrique : transformation log ou RobustScaler à évaluer |
| Class | binaire | Cible : 0 = légitime, 1 = fraude | Classe positive très minoritaire |

Notes :
- Les valeurs réelles (comptages, statistiques, outliers) figurent dans `docs/data_understanding_report.md`
  et `reports/metrics/eda_summary.json`.
- Le fichier brut `data/raw/creditcard.csv` est en lecture seule et n'est pas versionné.
