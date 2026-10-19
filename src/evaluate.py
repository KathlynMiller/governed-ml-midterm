"""
STAGE 4: EVALUATION

Measures the model on the held-out test split, overall AND for each group of
the protected attribute (area: urban vs rural).

Metrics produced
  auc            : ranking quality (0.5 = coin flip, 1.0 = perfect)
  accuracy       : share of correct approve/decline calls at the threshold
  approval_rate  : share of applicants offered the standard rate
  tpr            : true positive rate = share of drivers who stayed claim-free
                   that the model approved (good risks not turned away)
  fpr            : false positive rate = share of drivers who DID file a claim
                   that the model approved anyway (bad risks let through).
                   Already computed per group. Nothing in the gate uses it yet.
                   That is Lab 5.
  approval_disparity_ratio : lowest group approval rate / highest ("four-fifths")
  tpr_ratio      : lowest group TPR / highest group TPR ("equal opportunity")
  fpr_ratio      : lowest group FPR / highest group FPR. When this is far below
                   1, the model is strict with one group's bad risks and lenient
                   with the other's. Together with tpr_ratio this is the
                   "equalized odds" idea.
"""
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def group_metrics(y_true, scores, approved):
    positives = y_true == 1
    negatives = y_true == 0
    return {
        "n": int(len(y_true)),
        "auc": round(float(roc_auc_score(y_true, scores)), 4),
        "approval_rate": round(float(approved.mean()), 4),
        "tpr": round(float(approved[positives].mean()), 4),
        "fpr": round(float(approved[negatives].mean()), 4) if negatives.any() else 0.0,
    }


def ratio(values):
    return round(min(values) / max(values), 4) if max(values) > 0 else 0.0


def evaluate(model, config, test_df, group_col="area", label="claim_free"):
    scores = model.predict_proba(test_df[config["features"]])[:, 1]
    approved = scores >= config["decision_threshold"]
    y = test_df[label].values

    groups = {}
    for g in sorted(test_df[group_col].unique()):
        mask = (test_df[group_col] == g).values
        groups[g] = group_metrics(y[mask], scores[mask], approved[mask])

    return {
        "model_version": config["model_version"],
        "author": config.get("author", ""),
        "model_type": config["model_type"],
        "features": config["features"],
        "decision_threshold": config["decision_threshold"],
        "n_test": int(len(y)),
        "overall_auc": round(float(roc_auc_score(y, scores)), 4),
        "overall_accuracy": round(float(accuracy_score(y, approved)), 4),
        "groups": groups,
        "approval_disparity_ratio": ratio([m["approval_rate"] for m in groups.values()]),
        "tpr_ratio": ratio([m["tpr"] for m in groups.values()]),
        "fpr_ratio": ratio([m["fpr"] for m in groups.values()]),
    }


def save_metrics(metrics, path=None):
    ARTIFACTS.mkdir(exist_ok=True)
    path = path or ARTIFACTS / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
