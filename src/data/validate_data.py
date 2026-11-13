"""Fonctions d'audit de qualité et d'analyse statistique (sans modifier les données)."""
import numpy as np
import pandas as pd


def quality_report(df: pd.DataFrame, target: str = "Class") -> dict:
    features = df.drop(columns=[target], errors="ignore")
    numeric = features.select_dtypes(include=[np.number])
    freq_top = {c: float(df[c].value_counts(normalize=True, dropna=False).iloc[0]) for c in features}
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        "non_numeric_features": [c for c in features if c not in numeric.columns],
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicates_exact": int(df.duplicated().sum()),
        "duplicates_features_only": int(features.duplicated().sum()),
        "constant_columns": [c for c in features if df[c].nunique(dropna=False) <= 1],
        "near_zero_variance": [c for c, f in freq_top.items() if f > 0.99],
        "infinite_values": int(np.isinf(numeric).sum().sum()),
        "target_values": sorted(df[target].dropna().unique().tolist()),
        "invalid_target_rows": int((~df[target].isin([0, 1])).sum()),
        "negative_amount_rows": int((df["Amount"] < 0).sum()) if "Amount" in df else None,
        "negative_time_rows": int((df["Time"] < 0).sum()) if "Time" in df else None,
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 2),
    }


def class_distribution(df: pd.DataFrame, target: str = "Class") -> dict:
    counts = df[target].value_counts().sort_index()
    n_legit, n_fraud = int(counts.get(0, 0)), int(counts.get(1, 0))
    return {
        "legitimate": n_legit,
        "fraud": n_fraud,
        "fraud_percentage": round(100 * n_fraud / (n_legit + n_fraud), 4),
        "imbalance_ratio": round(n_legit / n_fraud, 2) if n_fraud else None,
    }


def descriptive_stats(df: pd.DataFrame, target: str = "Class") -> pd.DataFrame:
    x = df.drop(columns=[target])
    desc = x.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc["median"] = x.median()
    desc["skewness"] = x.skew()
    desc["kurtosis"] = x.kurt()
    return desc


def iqr_outlier_summary(df: pd.DataFrame, target: str = "Class") -> pd.DataFrame:
    """% de valeurs hors [Q1-1.5*IQR, Q3+1.5*IQR], calculé sur TOUT le jeu,
    puis ventilé par classe. Aucune ligne n'est supprimée."""
    rows = []
    for c in df.columns.drop(target):
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)
        rows.append({
            "feature": c,
            "outlier_pct_all": 100 * mask.mean(),
            "outlier_pct_legit": 100 * mask[df[target] == 0].mean(),
            "outlier_pct_fraud": 100 * mask[df[target] == 1].mean(),
        })
    return pd.DataFrame(rows).set_index("feature")


def correlation_with_target(df: pd.DataFrame, target: str = "Class") -> pd.Series:
    return df.corr(numeric_only=True)[target].drop(target).sort_values(key=np.abs, ascending=False)
