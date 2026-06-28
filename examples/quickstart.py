from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nz_nonergodic_model import predict_dataframe  # noqa: E402


def main() -> None:
    input_path = PROJECT_ROOT / "examples" / "example_input.csv"
    output_path = PROJECT_ROOT / "outputs" / "quickstart_predictions.csv"

    records = pd.read_csv(input_path)
    predictions = predict_dataframe(records, apply_site_correction=True, periods=[0.2, 1.0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Wrote {len(predictions)} rows to {output_path}")


if __name__ == "__main__":
    main()
