"""
LAB 6 SUBMISSION TESTS: is model_config.json a complete, honest submission?

These run in CI (Stage B) and whenever you run the whole suite. They are not
part of the pipeline's stage 0, so an unfinished submission never stops you
from running v1 and v2.

Run:   python -m pytest -q tests/test_lab6_submission.py
"""
import json
from pathlib import Path

from src import features
from src.train import ALLOWED_MODEL_TYPES

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.load(open(ROOT / "model_config.json"))
MIN_ENGINEERED_IN_MODEL = 2
MIN_IMPLEMENTED = 4
MIN_FAMILIES = 4


def test_author_is_filled_in():
    assert CONFIG.get("author", "").strip() not in ("", "YOUR NAME HERE"), "Put your name in model_config.json"


def test_model_type_is_allowed():
    assert CONFIG["model_type"] in ALLOWED_MODEL_TYPES


def test_enough_features_implemented():
    done = features.implemented_features()
    assert len(done) >= MIN_IMPLEMENTED, f"Implement at least {MIN_IMPLEMENTED} engineered features; found {done}"
    fams = {features.FAMILY[d] for d in done}
    assert len(fams) >= MIN_FAMILIES, f"Features must span at least {MIN_FAMILIES} families; found {sorted(fams)}"


def test_model_uses_engineered_features():
    used = [f for f in CONFIG["features"] if f in features.ENGINEERED]
    assert len(used) >= MIN_ENGINEERED_IN_MODEL, (
        f"Your model must use at least {MIN_ENGINEERED_IN_MODEL} of your engineered features; it uses {used}")
    not_built = [f for f in used if f not in features.implemented_features()]
    assert not not_built, f"Config uses features that are not implemented: {not_built}"
