"""Data loading and output helpers."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterable

import pandas as pd


ERGODIC_COEFF_COLUMNS = [
    "period",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "c9",
    "c10",
    "c11",
]

SITE_COEFF_COLUMNS = ["period", "c", "vc", "f4", "f5"]


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for {path}. Use .csv or .xlsx.")


def _require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="raise")
    if out[list(columns)].isna().any().any():
        raise ValueError(f"{table_name} contains NaN values in required numeric columns.")
    return out


def _normalize_period_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "period" not in out.columns and "Unnamed: 0" in out.columns:
        out = out.rename(columns={"Unnamed: 0": "period"})
    return out


def load_ergodic_coeffs(path: str | Path | None = None) -> pd.DataFrame:
    """Load ergodic model coefficients.

    If ``path`` is omitted, bundled coefficients converted from ``coeffs.xlsx``
    are used.
    """

    if path is None:
        with resources.files("nz_nonergodic_model.data").joinpath(
            "ergodic_coeffs.csv"
        ).open("r", encoding="utf-8") as handle:
            coeffs = pd.read_csv(handle)
    else:
        coeffs = _read_table(path)

    coeffs = _normalize_period_column(coeffs)
    _require_columns(coeffs, ERGODIC_COEFF_COLUMNS, "ergodic coefficients")
    coeffs = _coerce_numeric(coeffs[ERGODIC_COEFF_COLUMNS], ERGODIC_COEFF_COLUMNS, "ergodic coefficients")
    return coeffs.sort_values("period").reset_index(drop=True)


def load_site_coeffs(path: str | Path | None = None) -> pd.DataFrame:
    """Load SS14 site-correction coefficients."""

    if path is None:
        with resources.files("nz_nonergodic_model.data").joinpath(
            "ss14_site_coeffs.csv"
        ).open("r", encoding="utf-8") as handle:
            coeffs = pd.read_csv(handle)
    else:
        coeffs = _read_table(path)

    coeffs = _normalize_period_column(coeffs)
    _require_columns(coeffs, SITE_COEFF_COLUMNS, "site coefficients")
    coeffs = _coerce_numeric(coeffs[SITE_COEFF_COLUMNS], SITE_COEFF_COLUMNS, "site coefficients")
    return coeffs.sort_values("period").reset_index(drop=True)


def load_records(path: str | Path) -> pd.DataFrame:
    """Load prediction records from CSV or Excel."""

    records = _read_table(path)
    if "Mw" not in records.columns and "mag" in records.columns:
        records = records.rename(columns={"mag": "Mw"})
    _require_columns(records, ["Mw", "Rrup"], "input records")
    records = _coerce_numeric(records, ["Mw", "Rrup"], "input records")
    if "Vs30" in records.columns:
        records = _coerce_numeric(records, ["Vs30"], "input records")
    if "PGAr" in records.columns:
        records = _coerce_numeric(records, ["PGAr"], "input records")
    return records.reset_index(drop=True)


def write_predictions(predictions: pd.DataFrame, path: str | Path) -> Path:
    """Write predictions to CSV and create the parent directory if needed."""

    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Prediction output currently supports .csv only.")
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path
