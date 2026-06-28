"""Final Type-2 non-ergodic model helpers.

The trained INLA outputs can be large, so they are read from an external model
directory instead of bundled into the Python package by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .model import _period_label, _select_periods


DEFAULT_PATH_LABEL = "spatial_iid_cells"


def period_token(period: float) -> str:
    """Return the period token used by the existing Type-2 output files."""

    period = float(period)
    if abs(period - round(period)) < 1e-10 and period >= 1:
        text = f"{period:.1f}"
    else:
        text = f"{period:g}"
    return text.replace(".", "_")


def period_tag(period: float) -> str:
    return f"T{period_token(period)}"


@dataclass(frozen=True)
class Type2PeriodModel:
    """Type-2 non-ergodic coefficients for one oscillator period."""

    period: float
    period_tag: str
    path_label: str
    coefficients: pd.DataFrame
    cells: pd.DataFrame
    hyperparameters: pd.DataFrame | None = None

    @property
    def dc0_mean(self) -> float:
        if "dc_0_mean" in self.coefficients.columns:
            vals = pd.to_numeric(self.coefficients["dc_0_mean"], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.iloc[0])
        if self.hyperparameters is not None and {"parameter", "mean"}.issubset(self.hyperparameters.columns):
            row = self.hyperparameters[self.hyperparameters["parameter"].eq("dc_0")]
            if not row.empty:
                return float(row["mean"].iloc[0])
        return 0.0


class Type2NonErgodicModel:
    """Reader and predictor for final Type-2 ``spatial_iid_cells`` outputs."""

    def __init__(
        self,
        model_dir: str | Path,
        path_label: str = DEFAULT_PATH_LABEL,
        periods: Sequence[float] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.path_label = path_label
        self._models: dict[float, Type2PeriodModel] = {}

        if periods is None:
            periods = self._discover_periods()
        for period in periods:
            model = self._load_period_model(float(period))
            self._models[model.period] = model

    @property
    def periods(self) -> list[float]:
        return sorted(self._models)

    def period_model(self, period: float) -> Type2PeriodModel:
        available = np.asarray(self.periods, dtype=float)
        matches = np.where(np.isclose(available, float(period), rtol=0.0, atol=1e-9))[0]
        if len(matches) == 0:
            raise ValueError(
                f"Period {period:g} is not available in {self.model_dir}. "
                f"Available periods: {', '.join(f'{p:g}' for p in available)}"
            )
        return self._models[float(available[matches[0]])]

    def predict_adjustments(
        self,
        records: pd.DataFrame,
        cell_distance_matrix: pd.DataFrame | str | Path | None = None,
        periods: Sequence[float] | float | None = None,
        include_record_terms: bool = True,
    ) -> pd.DataFrame:
        """Return long-format non-ergodic adjustment terms.

        For records used in training, ``include_record_terms=True`` reads the
        exact source/site/path terms from ``*_inla_coefficients.csv`` by ``rsn``.
        For new records, set ``include_record_terms=False`` and pass a
        cell-distance matrix with ``c.<cellid>`` columns to calculate
        ``dc_0 + path`` from trained cell attenuation coefficients.
        """

        selected_models = self._select_models(periods)
        cellmat = None
        if cell_distance_matrix is not None:
            cellmat = load_cell_distance_matrix(cell_distance_matrix)
            cellmat = _align_cell_matrix(records, cellmat)

        rows: list[dict[str, object]] = []
        base_columns = [column for column in ["record_id", "rsn", "eqid", "ssn"] if column in records.columns]

        for record_index, record in records.reset_index(drop=True).iterrows():
            base = {column: record[column] for column in base_columns}
            for model in selected_models:
                row = {
                    **base,
                    "period": model.period,
                    "dc_0_mean": model.dc0_mean,
                    "dc_1e_mean": np.nan,
                    "dc_1as_mean": np.nan,
                    "dc_1bs_mean": np.nan,
                    "c_cap_path_mean": np.nan,
                    "delta_nonergodic_mean": np.nan,
                }

                coeff_row = _match_coefficients_row(record, model.coefficients) if include_record_terms else None
                if coeff_row is not None:
                    for column in ["dc_0_mean", "dc_1e_mean", "dc_1as_mean", "dc_1bs_mean", "c_cap_path_mean"]:
                        if column in coeff_row.index:
                            row[column] = float(coeff_row[column])
                    row["delta_nonergodic_mean"] = sum(
                        float(row[column])
                        for column in ["dc_0_mean", "dc_1e_mean", "dc_1as_mean", "dc_1bs_mean", "c_cap_path_mean"]
                    )
                    if "c_cap_path_sig" in coeff_row.index:
                        row["c_cap_path_sig"] = float(coeff_row["c_cap_path_sig"])
                elif cellmat is not None:
                    path_mean, path_sig = compute_path_term_for_record(cellmat.iloc[record_index], model.cells)
                    row["c_cap_path_mean"] = path_mean
                    row["c_cap_path_sig"] = path_sig
                    row["delta_nonergodic_mean"] = float(row["dc_0_mean"]) + path_mean
                rows.append(row)

        return pd.DataFrame(rows)

    def _select_models(self, periods: Sequence[float] | float | None) -> list[Type2PeriodModel]:
        period_df = pd.DataFrame({"period": self.periods})
        selected = _select_periods(period_df, periods)
        return [self.period_model(period) for period in selected["period"].to_numpy(float)]

    def _discover_periods(self) -> list[float]:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Non-ergodic model directory not found: {self.model_dir}")
        periods = []
        for child in self.model_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("T"):
                continue
            text = child.name[1:].replace("_", ".")
            try:
                periods.append(float(text))
            except ValueError:
                continue
        if not periods:
            raise FileNotFoundError(f"No period directories like T0_2 were found in {self.model_dir}")
        return sorted(set(periods))

    def _load_period_model(self, period: float) -> Type2PeriodModel:
        tag = period_tag(period)
        prefix = f"nz_type2_{tag}_{self.path_label}"
        period_dir = self.model_dir / tag
        coeff_path = period_dir / f"{prefix}_inla_coefficients.csv"
        cell_path = period_dir / f"{prefix}_inla_catten.csv"
        hyper_path = period_dir / f"{prefix}_inla_hyperparameters_quantiles.csv"

        if not coeff_path.exists():
            raise FileNotFoundError(f"Missing Type-2 coefficients: {coeff_path}")
        if not cell_path.exists():
            raise FileNotFoundError(f"Missing Type-2 cell attenuation file: {cell_path}")

        coefficients = pd.read_csv(coeff_path)
        cells = pd.read_csv(cell_path)
        hyperparameters = pd.read_csv(hyper_path) if hyper_path.exists() else None
        _validate_coefficients(coefficients, coeff_path)
        _validate_cells(cells, cell_path)
        return Type2PeriodModel(
            period=period,
            period_tag=tag,
            path_label=self.path_label,
            coefficients=coefficients,
            cells=cells,
            hyperparameters=hyperparameters,
        )


def load_cell_distance_matrix(path_or_df: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load a wide ``c.<cellid>`` distance matrix or sparse row/col/data CSV."""

    if isinstance(path_or_df, pd.DataFrame):
        matrix = path_or_df.copy()
    else:
        path = Path(path_or_df)
        matrix = pd.read_csv(path)

    if {"row", "col", "data"}.issubset(matrix.columns):
        matrix = sparse_cell_distance_to_wide(matrix)

    cell_cols = _cell_columns(matrix)
    if not cell_cols:
        raise ValueError("Cell-distance matrix must contain columns named like c.54.")
    matrix[cell_cols] = matrix[cell_cols].apply(pd.to_numeric, errors="raise")
    return matrix


def sparse_cell_distance_to_wide(sparse: pd.DataFrame) -> pd.DataFrame:
    """Convert sparse row/col/data matrix to a wide matrix with ``c.<col>`` cells.

    Sparse files generated by older scripts use 1-based row and column IDs.
    The resulting column names follow the Type2 convention ``c.<cellid>``.
    """

    sparse = sparse.copy()
    sparse[["row", "col", "data"]] = sparse[["row", "col", "data"]].apply(pd.to_numeric, errors="raise")
    n_rows = int(sparse["row"].max())
    wide = pd.DataFrame(index=np.arange(1, n_rows + 1))
    for cell_id, group in sparse.groupby("col", sort=True):
        values = np.zeros(n_rows, dtype=float)
        rows = group["row"].to_numpy(dtype=int) - 1
        values[rows] = group["data"].to_numpy(dtype=float)
        wide[f"c.{int(cell_id)}"] = values
    wide.insert(0, "row", np.arange(1, n_rows + 1))
    return wide.reset_index(drop=True)


def compute_path_terms(cell_distance_matrix: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Calculate path attenuation mean/sigma from a cell-distance matrix."""

    cellmat = load_cell_distance_matrix(cell_distance_matrix)
    rows = []
    for _, distances in cellmat.iterrows():
        path_mean, path_sig = compute_path_term_for_record(distances, cells)
        rows.append({"c_cap_path_mean": path_mean, "c_cap_path_sig": path_sig})
    return pd.DataFrame(rows)


def compute_path_term_for_record(distances: pd.Series, cells: pd.DataFrame) -> tuple[float, float]:
    cell_cols = [column for column in distances.index if _is_cell_column(column)]
    if not cell_cols:
        raise ValueError("No c.<cellid> columns found in distance row.")

    cell_lookup = cells.set_index(cells["cellid"].astype(int))
    mean = 0.0
    variance = 0.0
    for column in cell_cols:
        distance = float(distances[column])
        if distance == 0.0 or not np.isfinite(distance):
            continue
        cell_id = int(column.split(".", 1)[1])
        if cell_id not in cell_lookup.index:
            continue
        cell = cell_lookup.loc[cell_id]
        mean += distance * float(cell["c_cap_mean"])
        variance += (distance * float(cell["c_cap_sig"])) ** 2
    return mean, float(np.sqrt(variance))


def apply_nonergodic_adjustments(
    predictions: pd.DataFrame,
    adjustments: pd.DataFrame,
    base_column: str = "log10_y_site",
) -> pd.DataFrame:
    """Merge non-ergodic terms into prediction output and add final columns."""

    if base_column not in predictions.columns:
        raise ValueError(f"Base prediction column not found: {base_column}")
    merge_keys = [key for key in ["record_id", "rsn", "eqid", "ssn", "period"] if key in predictions.columns and key in adjustments.columns]
    if "period" not in merge_keys:
        raise ValueError("Both predictions and adjustments must contain a period column.")
    id_keys = [key for key in merge_keys if key != "period"]
    if not id_keys:
        raise ValueError(
            "Non-ergodic adjustment merge requires a record identifier such as rsn, record_id, or eqid+ssn."
        )
    out = predictions.merge(adjustments, on=merge_keys, how="left", suffixes=("", "_ne"))
    out["log10_y_nonergodic"] = out[base_column] + out["delta_nonergodic_mean"]
    out["y_nonergodic_g"] = np.power(10.0, out["log10_y_nonergodic"])
    return out


def _match_coefficients_row(record: pd.Series, coefficients: pd.DataFrame) -> pd.Series | None:
    for key_set in (["rsn"], ["record_id"], ["eqid", "ssn"]):
        if all(key in record.index and key in coefficients.columns for key in key_set):
            mask = np.ones(len(coefficients), dtype=bool)
            for key in key_set:
                mask &= coefficients[key].astype(str).to_numpy() == str(record[key])
            matches = coefficients.loc[mask]
            if not matches.empty:
                return matches.iloc[0]
    return None


def _align_cell_matrix(records: pd.DataFrame, cellmat: pd.DataFrame) -> pd.DataFrame:
    if "rsn" in records.columns and "rsn" in cellmat.columns:
        keyed = cellmat.set_index(cellmat["rsn"].astype(str), drop=False)
        keys = records["rsn"].astype(str)
        missing = [key for key in keys if key not in keyed.index]
        if missing:
            raise ValueError(f"Cell-distance matrix is missing rsn values: {missing[:5]}")
        return keyed.loc[keys].reset_index(drop=True)
    if len(cellmat) != len(records):
        raise ValueError("Cell-distance matrix must have the same row count as records when no rsn key is present.")
    return cellmat.reset_index(drop=True)


def _validate_coefficients(df: pd.DataFrame, path: Path) -> None:
    required = ["dc_0_mean", "dc_1e_mean", "dc_1as_mean", "dc_1bs_mean", "c_cap_path_mean"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _validate_cells(df: pd.DataFrame, path: Path) -> None:
    required = ["cellid", "c_cap_mean", "c_cap_sig"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _cell_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if _is_cell_column(column)]


def _is_cell_column(column: object) -> bool:
    text = str(column)
    if not text.startswith("c."):
        return False
    try:
        int(text.split(".", 1)[1])
    except ValueError:
        return False
    return True
