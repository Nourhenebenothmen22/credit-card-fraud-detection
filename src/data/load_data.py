"""Chargement du dataset brut (lecture seule)."""
import hashlib
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/creditcard.csv")
TARGET = "Class"


def file_sha256(path: Path = RAW_PATH) -> str:
    """Empreinte SHA-256 pour prouver que le brut n'a pas changé."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")
    return pd.read_csv(path)
