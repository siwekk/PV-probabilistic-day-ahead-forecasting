Hard feasibility constraints and soft physical information in probabilistic 
day-ahead photovoltaic forecasting
=============================================================================

This repository contains the programs and manuscript sources used for the
article ``Hard feasibility constraints and soft physical information 
in probabilistic day-ahead photovoltaic forecasting``.

The study separates two roles of physical information. Hard projection enforces
non-negative, capacity-bounded, night-time-feasible and non-crossing quantile
forecasts. Physical predictors, clear-sky references and residual targets may
improve prediction, but their value depends on the consistency of the source
forecast, plant metadata, target scaling and physical representation.

Author
------

Krzysztof Siwek, Warsaw University of Technology

Contact: krzysztof.siwek@pw.edu.pl

Repository: https://github.com/siwekk/PV-probabilistic-day-ahead-forecasting

Repository contents
-------------------

``scripts/``
  Data audits, panel construction, model training, calibration, statistical
  uncertainty analysis, table construction and article figure generation.

``configs/``
  Frozen study designs and chronological split definitions.

``results/``
  Compact machine-readable JSON results. Large predictions, checkpoints and
  logs are intentionally excluded from version control.

``data/source_manifest.tex``
  Source provenance and data-contract notes. Raw and processed data are not
  redistributed.

Installation
------------

Python 3.11 or 3.12 is recommended. Create an isolated environment and install
the CPU dependencies::

  python -m venv .venv
  .venv/Scripts/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

On Linux or macOS, activate the environment with
``source .venv/bin/activate``. Chronos-2 experiments require a compatible GPU
environment and the optional packages listed in
``requirements-gpu-cu126.txt``. The exact CUDA build may need adjustment for
the local driver.

Reproduction
------------

The experiments use several datasets with different forecast contracts. Read
``REPRODUCIBILITY.rst`` and ``data/source_manifest.tex`` before running a
pipeline. Raw files must be placed under ``data/raw/``. That directory is
excluded from version control.

The final confirmatory analyses are produced by the nine-quantile benchmark,
extended uncertainty, feasibility-audit, reliability and figure scripts. Each
program resolves paths relative to the repository root and writes derived
outputs under ``results/``.


Data and model licenses
-----------------------

The MIT license applies only to the original software in this repository. It
does not relicense third-party datasets, pretrained model weights or publisher
materials. Users must obtain those resources from their original providers and
follow their terms.

Citation
--------

Citation metadata are provided in ``CITATION.cff``. The article DOI and final
bibliographic record should be added after publication.
