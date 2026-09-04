# PV-probabilistic-day-ahead-forecasting

Reproducible experiments for the article ``Hard feasibility constraints and soft physical information in probabilistic day-ahead photovoltaic
forecasting.''

The repository contains preprocessing, physical feature construction, global LightGBM and XGBoost quantile models, Chronos-2 comparisons, physical projection, conformal calibration, dependence-aware uncertainty analysis, and manuscript figures. The main scientific comparison uses simulated NREL systems and measured PVOD telemetry. EMSx and ECMWF--Jacumba are retained as supplementary experiments with separate forecasting contracts.

Frozen machine-readable summaries are stored in `results/`. Raw and processed datasets are intentionally excluded because their redistribution follows the original source licences.

The review-specific analyses include nine-quantile evaluation, feasibility auditing, projection-threshold sensitivity, block-length sensitivity, conditional calibration, three-seed Chronos-2 adaptation budgets, a validation-grid search for the PVOD neural baselines, and the exploratory PVOD physical-reference diagnostic. The physical-reference diagnostic can be reproduced with:

```bash
python scripts/run_pvod_physical_mechanism_diagnostic.py
```

The projection and neural-baseline checks are reproduced with `scripts/run_projection_threshold_sensitivity.py` and `scripts/run_pvod_neural_baseline_sweep.py`.

Install the environment declared in `requirements.txt`, restore the source datasets using the provenance and pairing information in `docs/ingestion_status.tex` and `docs/audit_findings.tex`, and run the experiment scripts from the repository root. The test suite is available through `pytest -q`.

Author: Krzysztof Siwek, `krzysztof.siwek@pw.edu.pl`.

The code is released under the MIT License.
