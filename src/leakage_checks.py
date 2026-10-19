"""
LAB 3 TOOL: THE THREE LEAKAGE TESTS, IN PYTHON

You ran these in Excel in the features session. Here they are as code, run
against the raw applicant table so you can see every column before the
policy filters it.

    python src/leakage_checks.py

For every numeric column it prints:
  corr        absolute correlation with the label (too good = suspicious)
  fill_pos    share of claim-free drivers with a non-zero / non-blank value
  fill_neg    share of claim-filing drivers with a non-zero / non-blank value
              (a column filled almost only for one class is suspicious)
  recorded    WHEN the column was recorded, from the data dictionary
              (the timeline test: anything after the decision is leakage,
               whatever the numbers say)

The tool does not decide for you. Two of the three suspect columns listed in
the policy are legitimate or leakage for reasons the numbers alone do not
settle. Read the 'recorded' column. Then write your ruling into
governance/policy.json under column_rulings and justify it in MIDTERM.md.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prepare import LABEL, load_dictionary, load_policy, load_raw  # noqa: E402


def main():
    raw = load_raw()
    policy = load_policy()
    dic = load_dictionary().set_index("field")
    y = raw[LABEL]
    rows = []
    for col in raw.select_dtypes("number").columns:
        if col == LABEL:
            continue
        x = raw[col]
        filled = x.notna() & (x != 0)
        rows.append({
            "column": col,
            "corr": round(abs(x.corr(y)), 3) if x.notna().sum() > 1 else float("nan"),
            "fill_pos": round(filled[y == 1].mean(), 3),
            "fill_neg": round(filled[y == 0].mean(), 3),
            "recorded": dic.loc[col, "when_recorded"] if col in dic.index else "?",
            "suspect": "SUSPECT" if col in policy.get("suspect_columns", []) else "",
        })
    table = pd.DataFrame(rows).sort_values("corr", ascending=False)
    pd.set_option("display.width", 140)
    print(f"\nLeakage checks on data/policies_raw.csv   label = {LABEL}   correlation limit = {policy['max_feature_label_correlation']}\n")
    print(table.to_string(index=False))
    print("\nSuspect columns awaiting a ruling:", policy.get("suspect_columns", []))
    print("Current rulings:", policy.get("column_rulings", {}))
    print("\nRemember the timeline test: when was it recorded, relative to the decision?")


if __name__ == "__main__":
    main()
