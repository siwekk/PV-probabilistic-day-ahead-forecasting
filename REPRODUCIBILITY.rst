Reproducibility guide
=====================

Scope
-----

The repository preserves the programs used for source auditing, feature-panel
construction, model fitting, physical projection, conformal calibration,
dependence-aware uncertainty analysis and article generation. The datasets do
not describe one identical forecasting task. Their contracts and supported
claims are documented in the manuscript and configuration files.

Data boundary
-------------

Raw data, processed panels, frozen predictions, trained checkpoints and server
connection files are not committed. Raw resources belong under ``data/raw/``.
Derived panels belong under ``data/processed/`` or the paths recorded in the
study configurations. Compact JSON summaries are retained in ``results/``.

The EMSx and PVDAQ acquisition utilities are provided in ``scripts/``. NREL,
PVOD and ECMWF inputs must be obtained according to their source terms and the
provenance notes in ``data/source_manifest.tex`` and ``docs/``.

Principal workflow
------------------

The exact commands depend on which source data are available. The principal
article workflow is organised as follows::

  python scripts/audit_q1_sources.py
  python scripts/build_nrel_q1_panel.py
  python scripts/build_pvod_day_ahead_panel.py
  python scripts/run_nrel_physics_ladder.py
  python scripts/run_nrel_site_holdout.py
  python scripts/run_nrel_state_holdout.py
  python scripts/run_nine_quantile_benchmarks.py --dataset all
  python scripts/run_nrel_chronos2.py
  python scripts/run_nrel_chronos2_adaptation_sweep.py
  python scripts/run_ecmwf_vintage_benchmark.py
  python scripts/run_extended_uncertainty_analysis.py
  python scripts/build_feasibility_audit.py
  python scripts/complete_point_and_reliability.py
  python scripts/make_final_review_figures.py
  python scripts/build_all_method_tables.py

Before launching a long run, inspect the command help and the corresponding
JSON configuration. Some programs expect server-side paths recorded in the
frozen study design. Chronos-2 adaptation and full tuning require a GPU and are
substantially more expensive than the boosting experiments.

Statistical contract
--------------------

The principal probabilistic comparison uses quantiles 0.05, 0.10, 0.20, 0.30,
0.50, 0.70, 0.80, 0.90 and 0.95. Model differences use paired site and circular
delivery-day block resampling. Seven days is the primary block length, with 3,
14 and 28 days used for sensitivity analysis. State-transfer conclusions are
restricted to the three simulated states and are not national claims.

Verification
------------

Run the fast checks with::

  python -m compileall scripts
  python -m pytest

Then compare regenerated compact JSON files and manuscript tables with the
tracked results. Small numerical differences may occur across CPU instruction
sets, GPU kernels and library builds. The model versions and random seeds should
not be changed during confirmatory reproduction.
