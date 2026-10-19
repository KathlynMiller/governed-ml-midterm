"""
LAB 3 TEST: every suspect column has a ruling, and the policy acts on it.

The policy lists suspect_columns. For each one you must write a ruling into
column_rulings: "leakage" or "allowed". A column ruled leakage must appear in
leakage_columns (so the pipeline drops it). A column ruled allowed must not.
The numbers in src/leakage_checks.py help; the data dictionary's
when_recorded column decides. Justify each ruling in MIDTERM.md.

Run:   python -m pytest -q tests/test_lab3_leakage.py
"""
from src.prepare import load_policy, prepare_data

POLICY = load_policy()
VALID = {"leakage", "allowed"}


def test_every_suspect_column_has_a_ruling():
    rulings = POLICY.get("column_rulings", {})
    missing = [c for c in POLICY["suspect_columns"] if c not in rulings]
    bad = {c: r for c, r in rulings.items() if r not in VALID}
    assert not missing, f"No ruling yet for: {missing}. Add them to column_rulings in governance/policy.json."
    assert not bad, f"Rulings must be one of {sorted(VALID)}: {bad}"


def test_rulings_are_enforced_by_the_policy_lists():
    rulings = POLICY.get("column_rulings", {})
    not_dropped = [c for c, r in rulings.items() if r == "leakage" and c not in POLICY["leakage_columns"]]
    wrongly_dropped = [c for c, r in rulings.items() if r == "allowed" and c in POLICY["leakage_columns"]]
    assert not not_dropped, f"Ruled leakage but not in leakage_columns: {not_dropped}"
    assert not wrongly_dropped, f"Ruled allowed but listed as leakage: {wrongly_dropped}"


def test_rulings_match_the_processed_table():
    rulings = POLICY.get("column_rulings", {})
    if not rulings:
        return
    processed = prepare_data(policy=POLICY)
    present_leaks = [c for c, r in rulings.items() if r == "leakage" and c in processed.columns]
    assert not present_leaks, f"Ruled leakage but still in the training table: {present_leaks}"
