"""Tests for MetricsCalculator. Run: python test_metrics.py (or pytest)."""

import numpy as np

from metrics_calculator import MetricsCalculator as M


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_rmse_and_mae_known_values():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([2.0, 2.0, 4.0])
    # errors: 1, 0, 1 -> MAE = 2/3, RMSE = sqrt(2/3)
    assert approx(M.calculate_mae(actual, predicted), 2 / 3)
    assert approx(M.calculate_rmse(actual, predicted), np.sqrt(2 / 3))


def test_perfect_forecast_is_zero_error():
    a = np.array([10.0, 20.0, 30.0])
    metrics = M.calculate_all_metrics(a, a.copy())
    assert approx(metrics["rmse"], 0.0)
    assert approx(metrics["mae"], 0.0)
    assert approx(metrics["smape"], 0.0)


def test_smape_is_finite_when_actual_is_zero():
    actual = np.array([0.0, 100.0])
    predicted = np.array([5.0, 100.0])
    # MAPE would divide by zero on the first point; sMAPE stays finite.
    smape = M.calculate_smape(actual, predicted)
    assert np.isfinite(smape)
    assert smape > 0


def test_all_metrics_has_the_four_keys():
    m = M.calculate_all_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert set(m) == {"rmse", "mape", "mae", "smape"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all metrics tests passed")
