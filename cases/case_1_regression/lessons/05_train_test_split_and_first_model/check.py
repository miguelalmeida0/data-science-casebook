"""Self-check for Lesson 5. Run with `uv run pytest` in this directory.

Set LESSON_MODULE=solution to check the reference solution instead of
task.py (used by CI, not by students).
"""

import importlib.util
import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

_MODULE_NAME = os.environ.get("LESSON_MODULE", "task")
_LESSON_DIR = Path(__file__).parent
_MODULE_PATH = _LESSON_DIR / f"{_MODULE_NAME}.py"
_UNIQUE_NAME = f"lesson_{_LESSON_DIR.parent.parent.name}_{_LESSON_DIR.name}_{_MODULE_NAME}"

_spec = importlib.util.spec_from_file_location(_UNIQUE_NAME, _MODULE_PATH)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


def test_load_shipments_returns_493_rows():
    df = lesson.load_shipments()
    assert len(df) == 493


def test_split_shipments_produces_expected_sizes():
    df = lesson.load_shipments()
    train_df, test_df = lesson.split_shipments(df)
    assert len(train_df) == 394
    assert len(test_df) == 99
    assert len(train_df) + len(test_df) == len(df)


def test_impute_driver_experience_uses_train_median_only():
    df = lesson.load_shipments()
    train_df, test_df = lesson.split_shipments(df)
    assert train_df["driver_experience_years"].isna().sum() == 7
    assert test_df["driver_experience_years"].isna().sum() == 1

    train_df, test_df = lesson.impute_driver_experience(train_df, test_df)
    assert train_df["driver_experience_years"].isna().sum() == 0
    assert test_df["driver_experience_years"].isna().sum() == 0
    assert abs(train_df["driver_experience_years"].median() - 12.0) < 1e-9


def test_fit_model_returns_fitted_linear_regression():
    df = lesson.load_shipments()
    train_df, test_df = lesson.split_shipments(df)
    train_df, test_df = lesson.impute_driver_experience(train_df, test_df)
    model = lesson.fit_model(train_df)
    assert isinstance(model, LinearRegression)
    assert len(model.coef_) == 4


def test_model_predictions_match_expected_metrics_on_test_set():
    df = lesson.load_shipments()
    train_df, test_df = lesson.split_shipments(df)
    train_df, test_df = lesson.impute_driver_experience(train_df, test_df)
    model = lesson.fit_model(train_df)
    predicted = lesson.predict_delay(model, test_df)
    actual = test_df["delay_minutes"]

    mae = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    assert abs(mae - 10.2127) < 1e-3
    assert abs(rmse - 12.8064) < 1e-3


def test_model_beats_train_mean_baseline_on_test_set():
    df = lesson.load_shipments()
    train_df, test_df = lesson.split_shipments(df)
    train_df, test_df = lesson.impute_driver_experience(train_df, test_df)
    model = lesson.fit_model(train_df)
    predicted = lesson.predict_delay(model, test_df)
    actual = test_df["delay_minutes"]

    model_mae = mean_absolute_error(actual, predicted)
    model_rmse = root_mean_squared_error(actual, predicted)

    train_mean_delay = train_df["delay_minutes"].mean()
    baseline_pred = pd.Series(train_mean_delay, index=test_df.index)
    baseline_mae = mean_absolute_error(actual, baseline_pred)
    baseline_rmse = root_mean_squared_error(actual, baseline_pred)

    assert abs(baseline_mae - 12.0751) < 1e-3
    assert abs(baseline_rmse - 15.1869) < 1e-3
    assert model_mae < baseline_mae
    assert model_rmse < baseline_rmse
