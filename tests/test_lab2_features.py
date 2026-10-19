"""
LAB 2 TESTS: the engineered features behave like features should.

These pass from the start (nothing is implemented yet) and keep passing as
long as every feature you implement follows the rules:
  one value per policy, numeric, no missing values, and computed ONLY from
  months 1..24 (the temporal fence).

Run:   python -m pytest -q tests/test_lab2_features.py
"""
import numpy as np
import pandas as pd
import pytest

from src import features

HISTORY = features.load_history()
POLICIES = sorted(HISTORY["policy_id"].unique())


@pytest.fixture(scope="module")
def built():
    return features.build_features(HISTORY)


def test_one_row_per_policy(built):
    assert len(built) == len(POLICIES), "Feature table must have exactly one row per policy"
    assert sorted(built["policy_id"]) == POLICIES


def test_features_are_numeric_and_complete(built):
    for col in built.columns:
        if col == "policy_id":
            continue
        assert pd.api.types.is_numeric_dtype(built[col]), f"{col} is not numeric"
        assert built[col].notna().all(), f"{col} has missing values (check the 'never happened' case)"
        assert np.isfinite(built[col]).all(), f"{col} has infinite values (check your denominators)"


def test_temporal_fence_no_future_data():
    """THE AS-OF CONTROL. We append months 25..30 with absurd values. If any
    feature changes, that feature used data from after the decision."""
    future = HISTORY[HISTORY["month"] > features.DECISION_MONTH - 6].copy()
    future["month"] = future["month"] + 6                 # becomes 25..30
    future["miles"] = 1.0
    future["trips"] = 1.0
    future["hard_brakes"] = 500
    future["speeding_events"] = 500
    future["night_miles_share"] = 0.99
    future["claim_filed"] = 1
    poisoned = pd.concat([HISTORY, future], ignore_index=True)

    clean = features.build_features(HISTORY).set_index("policy_id").sort_index()
    dirty = features.build_features(poisoned).set_index("policy_id").sort_index()
    leaking = [c for c in clean.columns if not np.allclose(clean[c].values, dirty[c].values, equal_nan=True)]
    assert not leaking, f"These features used months after the decision: {leaking}. Apply as_of() first."


def test_implemented_features_are_registered():
    for name in features.implemented_features(HISTORY):
        assert name in features.FAMILY, f"{name} has no family in features.FAMILY"
