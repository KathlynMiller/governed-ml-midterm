"""
STAGE 6: THE MODEL CARD

Every run, pass or blocked, produces a model card: a one-page, human-readable
record of what the model is, how it performed, and what the gate decided.
In CI this file is uploaded as a build artifact. That makes it evidence.
"""
from datetime import datetime, timezone
from pathlib import Path

from src import features

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def build_card(config, metrics, report, policy, run_info):
    eng = [f for f in config["features"] if f in features.ENGINEERED]
    lines = []
    lines.append(f"# Model Card: Sierra Mutual Underwriting Model {config['model_version']}")
    lines.append("")
    lines.append(f"**Gate decision: {report['decision']}**  (policy v{policy['policy_version']}, owner: {policy['owner']})")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  commit `{run_info.get('commit', 'local')}`  |  data sha256 `{run_info.get('data_hash', '')[:12]}`  |  policy sha256 `{run_info.get('policy_hash', '')}`")
    lines.append("")
    lines.append("## Model details")
    lines.append(f"- Author: {config.get('author', '')}")
    lines.append(f"- Description: {config.get('description', '')}")
    lines.append(f"- Algorithm: {config['model_type']}  |  decision threshold: {config['decision_threshold']}")
    lines.append(f"- Features ({len(config['features'])}): {', '.join(config['features'])}")
    lines.append(f"- Engineered driving features used ({len(eng)}): {', '.join(eng) if eng else 'none'}")
    for f in eng:
        lines.append(f"  - {f}: family {features.FAMILY.get(f, 'unknown')}")
    lines.append("")
    lines.append("## Intended use")
    lines.append("- Recommends standard-rate approval or manual review for new one-year auto policies, reviewed by an underwriter.")
    lines.append("- Not for pricing, renewal, or claims handling. Not validated for commercial fleets.")
    lines.append("")
    lines.append("## Performance (held-out test set)")
    lines.append(f"- Overall AUC {metrics['overall_auc']:.4f}  |  accuracy {metrics['overall_accuracy']:.4f}  |  n = {metrics['n_test']}")
    lines.append("")
    lines.append("| Group | n | AUC | Approval rate | TPR (safe drivers approved) | FPR (claim filers approved) |")
    lines.append("|---|---|---|---|---|---|")
    for g, m in metrics["groups"].items():
        lines.append(f"| {g} | {m['n']} | {m['auc']:.4f} | {m['approval_rate']:.4f} | {m['tpr']:.4f} | {m['fpr']:.4f} |")
    lines.append("")
    lines.append(f"- Approval disparity ratio: {metrics['approval_disparity_ratio']:.4f} (policy minimum {policy['min_approval_disparity_ratio']})")
    lines.append(f"- TPR ratio: {metrics['tpr_ratio']:.4f} (policy minimum {policy['min_tpr_ratio']})")
    fpr_min = policy.get("min_fpr_ratio", "not governed")
    lines.append(f"- FPR ratio: {metrics['fpr_ratio']:.4f} (policy minimum {fpr_min})")
    lines.append("")
    lines.append("## Gate checks")
    lines.append("| Check | Value | Threshold | Result |")
    lines.append("|---|---|---|---|")
    for r in report["checks"]:
        lines.append(f"| {r['check']} | {r['value']} | {r['threshold']} | {'pass' if r['passed'] else 'FAIL'} |")
    lines.append("")
    lines.append("## Data lineage")
    lines.append("- Applicant table: data/policies_raw.csv, filtered by governance/policy.json (pii, leakage, sensitive lists).")
    lines.append("- Driving features: data/driving_history.csv, months 1 to 24 only, built by src/features.py.")
    lines.append("- data/history_after_decision.csv is never read by the pipeline.")
    lines.append("")
    lines.append("## Known limitations")
    lines.append("- Trained on historical claim outcomes that may encode past underwriting practice.")
    lines.append("- Fairness is evaluated for area (urban vs rural) only. Other groups have not been tested.")
    lines.append("- Synthetic data: for teaching only.")
    lines.append("")
    return "\n".join(lines)


def save_card(text, path=None):
    ARTIFACTS.mkdir(exist_ok=True)
    path = path or ARTIFACTS / "model_card.md"
    with open(path, "w") as f:
        f.write(text)
    return path
