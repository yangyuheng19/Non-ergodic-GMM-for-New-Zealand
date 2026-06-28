from pathlib import Path
import sys
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nz_nonergodic_model import (  # noqa: E402
    Type2NonErgodicModel,
    apply_nonergodic_adjustments,
    predict_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Type-2 non-ergodic prediction example.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Folder containing output_type2/T*/nz_type2_T*_spatial_iid_cells_inla_*.csv.",
    )
    parser.add_argument("--period", type=float, default=0.2, help="Oscillator period in seconds.")
    parser.add_argument("--nrows", type=int, default=3, help="Number of example rows to read.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    token = f"{args.period:g}".replace(".", "_")
    if args.period >= 1 and abs(args.period - round(args.period)) < 1e-10:
        token = f"{args.period:.1f}".replace(".", "_")
    tag = f"T{token}"
    input_file = model_dir / tag / f"nz_type2_{tag}_spatial_iid_cells_prepared_flatfile.csv"

    records = pd.read_csv(input_file, nrows=args.nrows)
    records = records.rename(columns={"mag": "Mw"})
    if "PGAr" not in records.columns:
        records["PGAr"] = 0.1

    base = predict_dataframe(records, apply_site_correction=True, periods=[args.period])

    type2 = Type2NonErgodicModel(model_dir, periods=[args.period])
    adjustments = type2.predict_adjustments(records, periods=[args.period])
    predictions = apply_nonergodic_adjustments(base, adjustments, base_column="log10_y_site")

    output_path = PROJECT_ROOT / "outputs" / "nonergodic_quickstart_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(predictions[["rsn", "period", "log10_y_site", "delta_nonergodic_mean", "log10_y_nonergodic"]])
    print(f"Wrote {len(predictions)} rows to {output_path}")


if __name__ == "__main__":
    main()
