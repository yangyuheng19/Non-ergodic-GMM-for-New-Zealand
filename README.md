# nz-nonergodic-model

Python utilities for using the New Zealand non-ergodic ground-motion model.

The workflow is:

1. calculate the ergodic rock prediction;
2. apply the Stewart & Seyhan (2014) site correction;
3. add the trained Type-2 non-ergodic correction:
   `dc_0 + dc_1e + dc_1as + dc_1bs + c_cap_path`.

This repository is for using the trained model. It does not rerun INLA.

## Install

From this directory:

```bash
pip install -e .
```

## Input

The prediction input file can be `.csv` or `.xlsx`.

Required columns:

- `Mw`: moment magnitude
- `Rrup`: closest rupture distance in km

Required only when applying site correction:

- `Vs30`: time-averaged shear-wave velocity in the upper 30 m, m/s
- `PGAr`: reference rock PGA in g for the SS14 nonlinear term

Optional columns:

- `record_id` or `rsn`: identifier copied to the output
- `eqid`: earthquake/source ID
- `ssn`: station ID

For full Type-2 non-ergodic predictions on records used in model fitting, the
input should include `rsn` so the code can match the trained source, site, and
path terms in the Type-2 output files.

## Run

Rock ergodic prediction:

```bash
nz-nonergodic predict --input examples/example_input.csv --output outputs/rock_predictions.csv
```

With SS14 site correction:

```bash
nz-nonergodic predict --input examples/example_input.csv --output outputs/site_predictions.csv --apply-site-correction
```

With the trained Type-2 non-ergodic model:

```bash
nz-nonergodic predict \
  --input path/to/input_records.csv \
  --output outputs/nonergodic_predictions.csv \
  --apply-site-correction \
  --nonergodic-model-dir path/to/output_type2 \
  --period 0.2
```

The `output_type2` folder should contain files like:

```text
output_type2/
`-- T0_2/
    |-- nz_type2_T0_2_spatial_iid_cells_inla_coefficients.csv
    |-- nz_type2_T0_2_spatial_iid_cells_inla_catten.csv
    `-- nz_type2_T0_2_spatial_iid_cells_prepared_flatfile.csv
```

You can also provide a constant `PGAr` if the input file does not contain a
`PGAr` column:

```bash
nz-nonergodic predict --input examples/example_input_without_pgar.csv --output outputs/site_predictions.csv --apply-site-correction --pgar 0.1
```

## Output

The default output is long format with one row per input record and period.

Key columns:

- `period`: oscillator period in seconds
- `log10_y_rock`: ergodic prediction in log10 units
- `y_rock_g`: ergodic spectral acceleration in g
- `site_term_ln`: SS14 site term in natural-log units, when requested
- `site_factor`: `exp(site_term_ln)`, when requested
- `log10_y_site`: site-corrected prediction in log10 units, when requested
- `y_site_g`: site-corrected spectral acceleration in g, when requested
- `delta_nonergodic_mean`: total Type-2 non-ergodic correction in log10 units
- `log10_y_nonergodic`: final non-ergodic prediction in log10 units
- `y_nonergodic_g`: final non-ergodic prediction in g

The model correction columns are:

- `dc_0_mean`: period-specific intercept correction
- `dc_1e_mean`: spatially varying source term
- `dc_1as_mean`: spatially varying site term
- `dc_1bs_mean`: station-specific site term
- `c_cap_path_mean`: path attenuation term accumulated along crossed cells

## Default coefficients

The package includes:

- `ergodic_coeffs.csv`: coefficients converted from `coeffs.xlsx`
- `ss14_site_coeffs.csv`: SS14 coefficients for the periods available in the
  ergodic coefficient table

Custom coefficient files can be supplied with:

```bash
nz-nonergodic predict --input examples/example_input.csv --output outputs/custom.csv --coeffs path/to/coeffs.xlsx --site-coeffs path/to/ss14.csv
```

## API example

```python
import pandas as pd
from nz_nonergodic_model import (
    Type2NonErgodicModel,
    apply_nonergodic_adjustments,
    predict_dataframe,
)

records = pd.read_csv("path/to/input_records.csv")
base = predict_dataframe(records, apply_site_correction=True, periods=[0.2])

type2 = Type2NonErgodicModel("path/to/output_type2", periods=[0.2])
adjustments = type2.predict_adjustments(records, periods=[0.2])
predictions = apply_nonergodic_adjustments(base, adjustments, base_column="log10_y_site")
predictions.to_csv("outputs/predictions.csv", index=False)
```

For a runnable example:

```bash
python examples/nonergodic_quickstart.py --model-dir path/to/output_type2 --period 0.2
```

## Tests

```bash
python -m unittest discover -s tests
```



