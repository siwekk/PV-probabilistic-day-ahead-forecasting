from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from review_metrics import QUANTILES, interval_score, metrics, projection_audit, rearrangement_audit


def test_perfect_quantiles_have_zero_point_error() -> None:
    target = np.array([0.2, 0.5, 0.8])
    prediction = np.repeat(target[:, None], len(QUANTILES), axis=1)
    result = metrics(target, prediction)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["crps_9q"] == 0.0
    assert result["r2"] == 1.0


def test_rearrangement_removes_crossings_without_increasing_mean_pinball() -> None:
    frame = pd.DataFrame({"daylight": [True, True], "target_power_normalized": [0.3, 0.7]})
    raw = np.asarray([
        [0.4, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.7, 0.9],
        [0.2, 0.1, 0.4, 0.3, 0.8, 0.7, 0.6, 0.9, 1.0],
    ])
    audit = rearrangement_audit(frame, raw)
    assert audit["crossing_row_rate_before"] == 1.0
    assert audit["crossing_row_rate_after"] == 0.0
    assert audit["mean_pinball_change_after_minus_before"] <= 1e-15


def test_interval_score_penalises_misses() -> None:
    target = np.array([0.5, 1.5])
    low = np.array([0.0, 0.0])
    high = np.array([1.0, 1.0])
    score = interval_score(target, low, high, alpha=0.10)
    assert score[0] == 1.0
    assert score[1] > score[0]


def test_projection_removes_all_audited_violations() -> None:
    frame = pd.DataFrame(
        {
            "daylight": [True, False],
            "target_power_normalized": [0.5, 0.0],
        }
    )
    raw = np.array(
        [
            [0.1, 0.2, 0.4, 0.3, 0.5, 0.7, 0.8, 0.9, 1.3],
            [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ]
    )
    result = projection_audit(frame, raw)
    assert result["rows_affected_by_any_correction_rate"] == 1.0
    assert all(value == 0.0 for value in result["projected"].values())
