"""Ground-motion prediction calculations."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from .io import load_ergodic_coeffs, load_site_coeffs


R1_KM = 7.57
R2_KM = 59.62
VREF_M_PER_S = 760.0
F3_G = 0.1


def _as_1d_float_array(values: Sequence[float] | pd.Series | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _select_periods(coeffs: pd.DataFrame, periods: Sequence[float] | float | None) -> pd.DataFrame:
    if periods is None:
        return coeffs.reset_index(drop=True)

    wanted = np.atleast_1d(np.asarray(periods, dtype=float))
    selected = []
    available = coeffs["period"].to_numpy(dtype=float)
    for period in wanted:
        matches = np.where(np.isclose(available, period, rtol=0.0, atol=1e-9))[0]
        if len(matches) == 0:
            raise ValueError(
                f"Period {period:g} is not available. "
                f"Available periods: {', '.join(f'{p:g}' for p in available)}"
            )
        selected.append(coeffs.iloc[matches[0]])
    return pd.DataFrame(selected).reset_index(drop=True)


def predict_ergodic(
    mw: Sequence[float] | pd.Series | np.ndarray,
    rrup: Sequence[float] | pd.Series | np.ndarray,
    coeffs: pd.DataFrame | None = None,
    periods: Sequence[float] | float | None = None,
) -> pd.DataFrame:
    """Predict rock ergodic ground motion using the MATLAB ``GMM_multi`` equation.

    Returns a wide table with one input record per row and period-specific
    ``log10_y_rock_*`` and ``y_rock_g_*`` columns.
    """

    mw_array = _as_1d_float_array(mw, "Mw")
    rrup_array = _as_1d_float_array(rrup, "Rrup")
    if mw_array.shape != rrup_array.shape:
        raise ValueError("Mw and Rrup must have the same length.")
    if (rrup_array < 0).any():
        raise ValueError("Rrup must be non-negative.")

    if coeffs is None:
        coeffs = load_ergodic_coeffs()
    coeffs = _select_periods(coeffs, periods)

    c = {f"c{i}": coeffs[f"c{i}"].to_numpy(dtype=float) for i in range(1, 12)}
    periods_array = coeffs["period"].to_numpy(dtype=float)

    mw_mat = mw_array[:, None]
    rrup_mat = rrup_array[:, None]
    radius = np.sqrt(rrup_mat**2 + c["c11"][None, :] ** 2)

    term1 = c["c1"][None, :] + c["c2"][None, :] * mw_mat + c["c3"][None, :] * mw_mat**2
    term2 = (c["c4"][None, :] + c["c5"][None, :] * mw_mat) * np.minimum(
        np.log10(radius), math.log10(R1_KM)
    )
    term3 = (c["c6"][None, :] + c["c7"][None, :] * mw_mat) * np.maximum(
        np.minimum(np.log10(radius / R1_KM), math.log10(R2_KM / R1_KM)), 0.0
    )
    term4 = (c["c8"][None, :] + c["c9"][None, :] * mw_mat) * np.maximum(
        np.log10(radius / R2_KM), 0.0
    )
    term5 = c["c10"][None, :] * radius

    log10_y = term1 + term2 + term3 + term4 + term5
    y = 10.0**log10_y

    out = pd.DataFrame(index=np.arange(len(mw_array)))
    for index, period in enumerate(periods_array):
        label = _period_label(period)
        out[f"log10_y_rock_{label}"] = log10_y[:, index]
        out[f"y_rock_g_{label}"] = y[:, index]
    return out


def ss14_site_terms(
    vs30: Sequence[float] | pd.Series | np.ndarray,
    pgar: Sequence[float] | pd.Series | np.ndarray | float,
    site_coeffs: pd.DataFrame | None = None,
    periods: Sequence[float] | float | None = None,
) -> pd.DataFrame:
    """Calculate Stewart & Seyhan (2014) site terms in natural-log units."""

    vs30_array = _as_1d_float_array(vs30, "Vs30")
    if (vs30_array <= 0).any():
        raise ValueError("Vs30 must be positive.")

    pgar_array = np.asarray(pgar, dtype=float)
    if pgar_array.ndim == 0:
        pgar_array = np.full_like(vs30_array, float(pgar_array))
    else:
        pgar_array = pgar_array.reshape(-1)
    if pgar_array.shape != vs30_array.shape:
        raise ValueError("PGAr must be a scalar or have the same length as Vs30.")
    if (pgar_array < 0).any() or not np.isfinite(pgar_array).all():
        raise ValueError("PGAr must be finite and non-negative.")

    if site_coeffs is None:
        site_coeffs = load_site_coeffs()
    site_coeffs = _select_periods(site_coeffs, periods)

    coeff = {
        name: site_coeffs[name].to_numpy(dtype=float)
        for name in ["c", "vc", "f4", "f5"]
    }
    periods_array = site_coeffs["period"].to_numpy(dtype=float)
    vs30_mat = vs30_array[:, None]
    pgar_mat = pgar_array[:, None]

    flin = np.where(
        vs30_mat < coeff["vc"][None, :],
        coeff["c"][None, :] * np.log(vs30_mat / VREF_M_PER_S),
        coeff["c"][None, :] * np.log(coeff["vc"][None, :] / VREF_M_PER_S),
    )
    f2 = coeff["f4"][None, :] * (
        np.exp(coeff["f5"][None, :] * (np.minimum(vs30_mat, VREF_M_PER_S) - 360.0))
        - np.exp(coeff["f5"][None, :] * (VREF_M_PER_S - 360.0))
    )
    fnl = np.where(
        vs30_mat >= VREF_M_PER_S,
        0.0,
        f2 * np.log((pgar_mat + F3_G) / F3_G),
    )
    site_term_ln = flin + fnl

    out = pd.DataFrame(index=np.arange(len(vs30_array)))
    for index, period in enumerate(periods_array):
        label = _period_label(period)
        out[f"site_term_ln_{label}"] = site_term_ln[:, index]
        out[f"site_factor_{label}"] = np.exp(site_term_ln[:, index])
    return out


def predict_dataframe(
    records: pd.DataFrame,
    coeffs: pd.DataFrame | None = None,
    site_coeffs: pd.DataFrame | None = None,
    apply_site_correction: bool = False,
    pgar: float | None = None,
    periods: Sequence[float] | float | None = None,
) -> pd.DataFrame:
    """Predict ground motions for input records and return long-format output."""

    if "Mw" not in records.columns and "mag" in records.columns:
        records = records.rename(columns={"mag": "Mw"})

    required = ["Mw", "Rrup"]
    missing = [column for column in required if column not in records.columns]
    if missing:
        raise ValueError(f"records is missing required columns: {missing}")

    if coeffs is None:
        coeffs = load_ergodic_coeffs()
    coeffs = _select_periods(coeffs, periods)
    periods_array = coeffs["period"].to_numpy(dtype=float)

    rock_wide = predict_ergodic(records["Mw"], records["Rrup"], coeffs=coeffs)

    site_wide = None
    if apply_site_correction:
        if "Vs30" not in records.columns:
            raise ValueError("Vs30 is required when apply_site_correction=True.")
        if pgar is None:
            if "PGAr" not in records.columns:
                raise ValueError("PGAr column or pgar argument is required for site correction.")
            pgar_values = records["PGAr"]
        else:
            pgar_values = float(pgar)
        if site_coeffs is None:
            site_coeffs = load_site_coeffs()
        site_coeffs = _select_periods(site_coeffs, periods_array)
        site_wide = ss14_site_terms(records["Vs30"], pgar_values, site_coeffs=site_coeffs)

    id_columns = [
        column
        for column in ["record_id", "rsn", "eqid", "ssn", "event_id", "station_id"]
        if column in records.columns
    ]
    base_columns = id_columns + [column for column in ["Mw", "Rrup", "Vs30", "PGAr"] if column in records.columns]

    rows = []
    for record_index, record in records.reset_index(drop=True).iterrows():
        base = {column: record[column] for column in base_columns}
        for period in periods_array:
            label = _period_label(period)
            row = {
                **base,
                "period": period,
                "log10_y_rock": rock_wide.at[record_index, f"log10_y_rock_{label}"],
                "y_rock_g": rock_wide.at[record_index, f"y_rock_g_{label}"],
            }
            if site_wide is not None:
                site_term = site_wide.at[record_index, f"site_term_ln_{label}"]
                site_factor = site_wide.at[record_index, f"site_factor_{label}"]
                row.update(
                    {
                        "site_term_ln": site_term,
                        "site_factor": site_factor,
                        "log10_y_site": row["log10_y_rock"] + site_term / math.log(10.0),
                        "y_site_g": row["y_rock_g"] * site_factor,
                    }
                )
            rows.append(row)

    return pd.DataFrame(rows)


def _period_label(period: float) -> str:
    return f"T{period:g}".replace("-", "m").replace(".", "p")
