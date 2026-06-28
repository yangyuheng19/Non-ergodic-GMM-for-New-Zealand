"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_ergodic_coeffs, load_records, load_site_coeffs, write_predictions
from .model import predict_dataframe
from .nonergodic import Type2NonErgodicModel, apply_nonergodic_adjustments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nz-nonergodic",
        description="Predict New Zealand ergodic and Type-2 non-ergodic ground motions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run predictions for a CSV or Excel input table.")
    predict.add_argument("--input", required=True, help="Input .csv/.xlsx file with Mw and Rrup columns.")
    predict.add_argument("--output", required=True, help="Output CSV path.")
    predict.add_argument("--coeffs", help="Optional ergodic coefficient .csv/.xlsx file.")
    predict.add_argument("--site-coeffs", help="Optional SS14 site coefficient .csv/.xlsx file.")
    predict.add_argument(
        "--nonergodic-model-dir",
        help=(
            "Optional Type-2 output_type2 directory containing T*/"
            "nz_type2_T*_spatial_iid_cells_inla_*.csv files."
        ),
    )
    predict.add_argument(
        "--cell-distance-matrix",
        help="Optional wide c.<cellid> or sparse row/col/data CSV for new-record path terms.",
    )
    predict.add_argument(
        "--nonergodic-base",
        default="log10_y_site",
        choices=["log10_y_site", "log10_y_rock"],
        help="Base column for adding non-ergodic corrections. Defaults to log10_y_site.",
    )
    predict.add_argument(
        "--apply-site-correction",
        action="store_true",
        help="Apply SS14 site correction using Vs30 and PGAr.",
    )
    predict.add_argument(
        "--pgar",
        type=float,
        help="Constant rock PGA in g for all rows when the input has no PGAr column.",
    )
    predict.add_argument(
        "--period",
        type=float,
        action="append",
        dest="periods",
        help="Period to predict. Repeat for multiple periods. Defaults to all periods.",
    )
    predict.set_defaults(func=run_predict)
    return parser


def run_predict(args: argparse.Namespace) -> Path:
    records = load_records(args.input)
    coeffs = load_ergodic_coeffs(args.coeffs)
    site_coeffs = load_site_coeffs(args.site_coeffs) if args.apply_site_correction else None
    predictions = predict_dataframe(
        records,
        coeffs=coeffs,
        site_coeffs=site_coeffs,
        apply_site_correction=args.apply_site_correction,
        pgar=args.pgar,
        periods=args.periods,
    )
    if args.nonergodic_model_dir:
        if args.nonergodic_base == "log10_y_site" and "log10_y_site" not in predictions.columns:
            raise ValueError(
                "--nonergodic-base log10_y_site requires --apply-site-correction. "
                "Use --nonergodic-base log10_y_rock to add corrections to rock predictions."
            )
        ne_model = Type2NonErgodicModel(args.nonergodic_model_dir, periods=args.periods)
        adjustments = ne_model.predict_adjustments(
            records,
            cell_distance_matrix=args.cell_distance_matrix,
            periods=args.periods,
            include_record_terms=True,
        )
        predictions = apply_nonergodic_adjustments(
            predictions,
            adjustments,
            base_column=args.nonergodic_base,
        )
    output = write_predictions(predictions, args.output)
    print(f"Wrote {len(predictions)} rows to {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
