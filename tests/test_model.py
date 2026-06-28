import math
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nz_nonergodic_model import (
    Type2NonErgodicModel,
    apply_nonergodic_adjustments,
    predict_dataframe,
    predict_ergodic,
    ss14_site_terms,
)
from nz_nonergodic_model.io import load_ergodic_coeffs


class ModelTests(unittest.TestCase):
    def test_predict_ergodic_matches_gmm_multi_formula_for_one_period(self):
        coeffs = load_ergodic_coeffs()
        period_coeffs = coeffs[coeffs["period"].eq(0.2)].reset_index(drop=True)
        out = predict_ergodic([6.5], [10.0], coeffs=period_coeffs)

        row = period_coeffs.iloc[0]
        r1 = 7.57
        r2 = 59.62
        radius = math.sqrt(10.0**2 + row["c11"] ** 2)
        expected = (
            row["c1"]
            + row["c2"] * 6.5
            + row["c3"] * 6.5**2
            + (row["c4"] + row["c5"] * 6.5) * min(math.log10(radius), math.log10(r1))
            + (row["c6"] + row["c7"] * 6.5)
            * max(min(math.log10(radius / r1), math.log10(r2 / r1)), 0.0)
            + (row["c8"] + row["c9"] * 6.5) * max(math.log10(radius / r2), 0.0)
            + row["c10"] * radius
        )

        self.assertAlmostEqual(out.at[0, "log10_y_rock_T0p2"], expected, places=12)
        self.assertAlmostEqual(out.at[0, "y_rock_g_T0p2"], 10.0**expected, places=12)

    def test_ss14_site_term_is_zero_at_reference_vs30_for_any_pgar(self):
        out = ss14_site_terms([760.0], [0.1], periods=[0.2])

        self.assertAlmostEqual(out.at[0, "site_term_ln_T0p2"], 0.0, places=12)
        self.assertAlmostEqual(out.at[0, "site_factor_T0p2"], 1.0, places=12)

    def test_predict_dataframe_outputs_long_site_corrected_rows(self):
        records = pd.DataFrame(
            {
                "record_id": ["eq001"],
                "Mw": [6.5],
                "Rrup": [10.0],
                "Vs30": [400.0],
                "PGAr": [0.1],
            }
        )
        out = predict_dataframe(records, apply_site_correction=True, periods=[0.2])

        self.assertEqual(len(out), 1)
        self.assertIn("site_term_ln", out.columns)
        self.assertIn("y_site_g", out.columns)
        self.assertAlmostEqual(
            out.at[0, "log10_y_site"],
            out.at[0, "log10_y_rock"] + out.at[0, "site_term_ln"] / math.log(10.0),
            places=12,
        )

    def test_type2_adjustments_match_spatial_iid_coefficients_when_available(self):
        model_dir = Path(r"G:\26research\non-ergodic\New-Zealand\code\output_type2")
        coeff_file = model_dir / "T0_2" / "nz_type2_T0_2_spatial_iid_cells_inla_coefficients.csv"
        if not coeff_file.exists():
            self.skipTest("Local Type-2 output files are not available.")

        records = pd.read_csv(coeff_file, nrows=1)
        records["Mw"] = 6.5
        records["Vs30"] = 760.0
        records["PGAr"] = 0.1

        type2 = Type2NonErgodicModel(model_dir, periods=[0.2])
        adjustments = type2.predict_adjustments(records, periods=[0.2])
        expected = sum(
            float(records.at[0, column])
            for column in ["dc_0_mean", "dc_1e_mean", "dc_1as_mean", "dc_1bs_mean", "c_cap_path_mean"]
        )

        self.assertAlmostEqual(adjustments.at[0, "delta_nonergodic_mean"], expected, places=12)

        base = predict_dataframe(records, apply_site_correction=True, periods=[0.2])
        out = apply_nonergodic_adjustments(base, adjustments)
        self.assertAlmostEqual(
            out.at[0, "log10_y_nonergodic"],
            out.at[0, "log10_y_site"] + expected,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
