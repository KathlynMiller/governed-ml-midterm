"""
STAGE 2: DATA PREPARATION

Turns policies_raw.csv plus the engineered driving features into one
training-ready table.

Compared with Week 5, this file is already policy driven: every drop list is
read from governance/policy.json. Nothing here needs editing in the midterm.
If a data test fails, the fix is in the POLICY, not in this code. That is the
point: the code enforces whatever the committee wrote down. When the committee
forgot something, the detective tests tell you, and you complete the policy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import features

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "policies_raw.csv"
POLICY_PATH = ROOT / "governance" / "policy.json"
DICTIONARY_PATH = ROOT / "data" / "data_dictionary.csv"

LABEL = "claim_free"
SPLIT_SEED = 690


def load_policy(path=POLICY_PATH):
    """The policy is data, not code: it is the committee's decision, versioned
    in the repo like everything else."""
    with open(path) as f:
        return json.load(f)


def load_dictionary(path=DICTIONARY_PATH):
    """The data dictionary: what every raw column is and WHEN it was recorded."""
    return pd.read_csv(path)


def load_raw(path=RAW_PATH):
    return pd.read_csv(path, dtype={"zip": str})


def drop_listed(df, columns, reason):
    """Drop whichever of the listed columns exist. Never crash on a missing one."""
    present = [c for c in columns if c in df.columns]
    return df.drop(columns=present)


def drop_identifiers(df, policy):
    """Remove direct and quasi identifiers listed in policy['pii_columns']."""
    return drop_listed(df, policy["pii_columns"], "pii")


def remove_leakage(df, policy):
    """Remove columns recorded after the decision, listed in policy['leakage_columns']."""
    return drop_listed(df, policy["leakage_columns"], "leakage")


def drop_sensitive_except_protected(df, policy):
    """Sensitive attributes are never model inputs. Exactly ONE of them, the
    protected_attribute, stays in the table so the evaluation stage can audit
    fairness across its groups. You keep it to TEST, never to DECIDE."""
    to_drop = [c for c in policy["sensitive_attributes"] if c != policy["protected_attribute"]]
    return drop_listed(df, to_drop, "sensitive")


def impute_missing(df):
    """Fill missing numeric values with the column median. Documented and dull."""
    out = df.copy()
    for col in ["prior_insurer_years", "annual_mileage_declared"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    return out


def add_basic_features(df):
    """Simple recodings of declared fields. These are NOT the engineered
    driving features; those come from src/features.py (Lab 2)."""
    out = df.copy()
    out["coverage_level"] = out["coverage_type"].map({"basic": 0, "standard": 1, "full": 2})
    out["log_vehicle_value"] = np.log(out["vehicle_value"])
    out["mileage_per_year_licensed"] = out["annual_mileage_declared"] / (out["years_licensed"] + 1.0)
    return out


def merge_driving_features(df):
    """Join the engineered features built from the 24 months BEFORE the decision."""
    feats = features.build_features(features.load_history())
    if feats is None or feats.shape[1] <= 1:
        return df
    return df.merge(feats, on="policy_id", how="left")


def prepare_data(raw_path=RAW_PATH, policy=None):
    """The full preparation stage, in order. Returns the processed table."""
    policy = policy or load_policy()
    df = load_raw(raw_path)
    df = drop_identifiers(df, policy)
    df = remove_leakage(df, policy)
    df = drop_sensitive_except_protected(df, policy)
    df = impute_missing(df)
    df = add_basic_features(df)
    df = merge_driving_features(df)
    return df


def split(df):
    """Fixed-seed stratified split so every run, and every student, is comparable."""
    return train_test_split(df, test_size=0.3, random_state=SPLIT_SEED, stratify=df[LABEL])


if __name__ == "__main__":
    data = prepare_data()
    print(f"processed rows={len(data)} columns={len(data.columns)}")
    print(list(data.columns))
