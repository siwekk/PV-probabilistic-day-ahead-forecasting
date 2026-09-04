"""Metrics and dependence-aware comparisons for the review experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = np.asarray([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])


def pinball_matrix(y: np.ndarray, qhat: np.ndarray) -> np.ndarray:
    error = y[:, None] - qhat
    return np.maximum(QUANTILES[None, :] * error, (QUANTILES[None, :] - 1.0) * error)


def interval_score(y: np.ndarray, low: np.ndarray, high: np.ndarray, alpha: float) -> np.ndarray:
    return high - low + 2.0 / alpha * (low - y) * (y < low) + 2.0 / alpha * (y - high) * (y > high)


def metrics(y: np.ndarray, qhat: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, float)
    qhat = np.asarray(qhat, float)
    median = qhat[:, 4]
    losses = pinball_matrix(y, qhat)
    extended_q = np.r_[0.0, QUANTILES, 1.0]
    extended_loss = np.column_stack([losses[:, 0], losses, losses[:, -1]])
    crps = 2.0 * np.trapz(extended_loss, x=extended_q, axis=1)
    alpha_pairs = [(0.60, 3, 5), (0.40, 2, 6), (0.20, 1, 7), (0.10, 0, 8)]
    interval_scores = {alpha: interval_score(y, qhat[:, lo], qhat[:, hi], alpha) for alpha, lo, hi in alpha_pairs}
    wis = (0.5 * np.abs(y - median) + sum(alpha / 2.0 * interval_scores[alpha] for alpha, _, _ in alpha_pairs)) / 4.5
    denominator = np.abs(y) + np.abs(median)
    smape = np.divide(2.0 * np.abs(y - median), denominator, out=np.zeros_like(y), where=denominator > 0)
    target_variation = np.sum((y - y.mean()) ** 2)
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - median))),
        "rmse": float(np.sqrt(np.mean((y - median) ** 2))),
        "smape_percent": float(100.0 * np.mean(smape)),
        "r2": float(1.0 - np.sum((y - median) ** 2) / target_variation) if target_variation > 0 else float("nan"),
        "mean_pinball": float(losses.mean()),
        "crps_9q": float(crps.mean()),
        "wis": float(wis.mean()),
    }
    for alpha, lo, hi in alpha_pairs:
        nominal = 1.0 - alpha
        coverage = np.mean((y >= qhat[:, lo]) & (y <= qhat[:, hi]))
        suffix = str(int(round(100 * nominal)))
        result[f"coverage_{suffix}"] = float(coverage)
        result[f"coverage_error_{suffix}"] = float(coverage - nominal)
        result[f"width_{suffix}"] = float(np.mean(qhat[:, hi] - qhat[:, lo]))
        result[f"interval_score_{suffix}"] = float(interval_scores[alpha].mean())
    return result


def diagnostics(frame: pd.DataFrame, raw: np.ndarray, ordered: np.ndarray, upper: float = 1.2) -> dict[str, float]:
    daylight = frame["daylight"].to_numpy(bool)
    return {
        "any_quantile_crossing_rate_raw": float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1))),
        "negative_quantile_rate_raw": float(np.mean(raw < 0.0)),
        "above_capacity_bound_rate_raw": float(np.mean(raw > upper)),
        "night_nonzero_median_rate_ordered": float(np.mean(np.abs(ordered[~daylight, 4]) > 1e-8)) if np.any(~daylight) else 0.0,
    }


def rearrangement_audit(frame: pd.DataFrame, raw: np.ndarray) -> dict:
    """Measure the isolated effect of discrete increasing rearrangement."""
    raw = np.asarray(raw, float)
    daylight = frame["daylight"].to_numpy(bool)
    y = frame["target_power_normalized"].to_numpy(float)[daylight]
    raw_day = raw[daylight]
    rearranged_day = np.sort(raw_day, axis=1)
    raw_losses = pinball_matrix(y, raw_day)
    rearranged_losses = pinball_matrix(y, rearranged_day)
    raw_metrics = metrics(y, raw_day)
    rearranged_metrics = metrics(y, rearranged_day)

    return {
        "method": "Discrete increasing rearrangement by sorting the nine predictions within each row.",
        "daylight_rows": int(daylight.sum()),
        "crossing_row_rate_before": float(np.mean(np.any(np.diff(raw_day, axis=1) < 0, axis=1))),
        "crossing_row_rate_after": float(np.mean(np.any(np.diff(rearranged_day, axis=1) < 0, axis=1))),
        "mean_absolute_quantile_displacement": float(np.mean(np.abs(rearranged_day - raw_day))),
        "mean_absolute_displacement_by_quantile": {
            f"q{int(round(100 * quantile)):02d}": float(value)
            for quantile, value in zip(QUANTILES, np.mean(np.abs(rearranged_day - raw_day), axis=0))
        },
        "mean_pinball_before": float(raw_losses.mean()),
        "mean_pinball_after": float(rearranged_losses.mean()),
        "mean_pinball_change_after_minus_before": float(rearranged_losses.mean() - raw_losses.mean()),
        "crps_9q_before": float(raw_metrics["crps_9q"]),
        "crps_9q_after": float(rearranged_metrics["crps_9q"]),
        "crps_9q_change_after_minus_before": float(rearranged_metrics["crps_9q"] - raw_metrics["crps_9q"]),
        "pinball_before_by_quantile": {
            f"q{int(round(100 * quantile)):02d}": float(value)
            for quantile, value in zip(QUANTILES, raw_losses.mean(axis=0))
        },
        "pinball_after_by_quantile": {
            f"q{int(round(100 * quantile)):02d}": float(value)
            for quantile, value in zip(QUANTILES, rearranged_losses.mean(axis=0))
        },
        "pinball_change_by_quantile": {
            f"q{int(round(100 * quantile)):02d}": float(value)
            for quantile, value in zip(QUANTILES, rearranged_losses.mean(axis=0) - raw_losses.mean(axis=0))
        },
    }


def projection_audit(frame: pd.DataFrame, raw: np.ndarray, upper: float = 1.2) -> dict:
    """Audit ordering and physical projection without conformal calibration."""
    raw = np.asarray(raw, float)
    daylight = frame["daylight"].to_numpy(bool)
    ordered = np.sort(raw, axis=1)
    projected = np.clip(ordered, 0.0, upper)
    projected[~daylight, :] = 0.0
    crossing = np.any(np.diff(raw, axis=1) < 0, axis=1)
    negative = np.any(raw < 0.0, axis=1)
    above = np.any(raw > upper, axis=1)
    night_median = np.zeros(len(frame), dtype=bool)
    night_upper = np.zeros(len(frame), dtype=bool)
    night_median[~daylight] = np.abs(ordered[~daylight, 4]) > 1e-8
    night_upper[~daylight] = np.abs(ordered[~daylight, -1]) > 1e-8
    affected = crossing | negative | above | night_median | night_upper

    def rates(values: np.ndarray) -> dict[str, float]:
        return {
            "negative_quantile_value_rate": float(np.mean(values < 0.0)),
            "above_capacity_quantile_value_rate": float(np.mean(values > upper)),
            "nonzero_night_median_rate": float(np.mean(np.abs(values[~daylight, 4]) > 1e-8)) if np.any(~daylight) else 0.0,
            "nonzero_night_upper_rate": float(np.mean(np.abs(values[~daylight, -1]) > 1e-8)) if np.any(~daylight) else 0.0,
            "adjacent_crossing_row_rate": float(np.mean(np.any(np.diff(values, axis=1) < 0, axis=1))),
        }

    y = frame["target_power_normalized"].to_numpy(float)
    raw_day = metrics(y[daylight], ordered[daylight])
    projected_day = metrics(y[daylight], projected[daylight])
    return {
        "raw": rates(raw),
        "projected": rates(projected),
        "rows_affected_by_any_correction_rate": float(np.mean(affected)),
        "daylight_metric_change_projected_minus_raw": {
            name: float(projected_day[name] - raw_day[name])
            for name in ["mae", "crps_9q", "wis", "interval_score_90"]
        },
        "daylight_raw": {name: raw_day[name] for name in ["mae", "crps_9q", "wis", "interval_score_90"]},
        "daylight_projected": {name: projected_day[name] for name in ["mae", "crps_9q", "wis", "interval_score_90"]},
    }


def conditional_metrics(frame: pd.DataFrame, prediction: np.ndarray, groups: dict[str, pd.Series]) -> dict:
    output = {}
    y = frame["target_power_normalized"].to_numpy(float)
    daylight = frame["daylight"].to_numpy(bool)
    for name, labels in groups.items():
        labels = pd.Series(labels, index=frame.index).astype(str).to_numpy()
        output[name] = {}
        for value in sorted(set(labels)):
            mask = (labels == value) & daylight
            if mask.sum() >= 30:
                output[name][value] = metrics(y[mask], prediction[mask])
    return output


def hierarchical_block_bootstrap(
    frame: pd.DataFrame,
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    site_col: str,
    day_col: str,
    draws: int = 4000,
    block_days: int = 7,
    seed: int = 20260829,
) -> dict[str, float | int]:
    work = frame[[site_col, day_col]].copy()
    work["difference"] = np.asarray(losses_a) - np.asarray(losses_b)
    daily = work.groupby([site_col, day_col], observed=True)["difference"].mean().reset_index()
    sites = sorted(daily[site_col].unique())
    by_site = {site: daily.loc[daily[site_col] == site].sort_values(day_col)["difference"].to_numpy() for site in sites}
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        selected_sites = rng.choice(sites, size=len(sites), replace=True)
        site_means = []
        for site in selected_sites:
            values = by_site[site]
            required = len(values)
            sampled = []
            while sum(len(x) for x in sampled) < required:
                start = int(rng.integers(0, len(values)))
                sampled.append(values[(start + np.arange(block_days)) % len(values)])
            site_means.append(np.concatenate(sampled)[:required].mean())
        estimates[draw] = np.mean(site_means)
    return {
        "draws": draws,
        "block_days": block_days,
        "estimate": float(daily.groupby(site_col, observed=True)["difference"].mean().mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }
