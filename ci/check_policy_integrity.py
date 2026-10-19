"""
CI STAGE A: POLICY INTEGRITY CHECK (separation of duties, midterm edition)

Week 5 rule: a model pull request may not touch governance at all.
Midterm rule: you are also the governance team, so you MAY change the policy,
the gate, and the gate tests, under two conditions enforced here:

  1. Every governance change is REPORTED in the CI log and routed to the
     policy owner for review (CODEOWNERS does the routing).
  2. A change may TIGHTEN but never LOOSEN. Concretely:
       - any policy key starting with  min_  may not decrease
       - any policy key starting with  max_  may not increase
       - pii_columns, leakage_columns, prohibited_features and
         sensitive_attributes may not lose entries
       - the policy_version must change if any numeric threshold changes
     A pull request that loosens anything FAILS here, before any model runs.

  3. Some files are HARD PROTECTED: the pipeline machinery, the data, the
     test files that grade Labs 1, 2, 3 and 6, and CI itself. Any change to
     those fails immediately.

Usage in CI:   python ci/check_policy_integrity.py <base_commit_sha>
"""
import json
import subprocess
import sys

HARD_PROTECTED = [
    "ci/", ".github/", ".devcontainer/", "data/",
    "run_pipeline.py", "src/prepare.py", "src/train.py", "src/evaluate.py", "src/model_card.py",
    "src/leakage_checks.py", "tests/conftest.py",
    "tests/test_lab1_compliance.py", "tests/test_lab2_features.py",
    "tests/test_lab3_leakage.py", "tests/test_lab6_submission.py",
]
GOVERNANCE_FILES = ["governance/policy.json", "src/gate.py", "tests/test_lab5_gate.py"]
LIST_KEYS = ["pii_columns", "leakage_columns", "prohibited_features", "sensitive_attributes"]


def run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print("git command failed:", " ".join(cmd))
        print(out.stderr.strip())
        print("Failing closed: a control that cannot run must not report success.")
        sys.exit(1)
    return out.stdout


def changed_files(base):
    return [f for f in run(["git", "diff", "--name-only", base, "HEAD"]).splitlines() if f.strip()]


def file_at(ref, path):
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def compare_policies(base):
    old_txt, new_txt = file_at(base, "governance/policy.json"), file_at("HEAD", "governance/policy.json")
    if old_txt is None or new_txt is None:
        return ["policy.json missing on one side of the comparison"]
    old, new = json.loads(old_txt), json.loads(new_txt)
    problems, tightened, added = [], [], []
    for k, v in old.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nv = new.get(k)
            if nv is None:
                problems.append(f"threshold {k} was removed")
            elif k.startswith("min_") and nv < v:
                problems.append(f"{k} LOOSENED from {v} to {nv}")
            elif k.startswith("max_") and nv > v:
                problems.append(f"{k} LOOSENED from {v} to {nv}")
            elif nv != v:
                tightened.append(f"{k}: {v} -> {nv}")
    for k in LIST_KEYS:
        lost = sorted(set(old.get(k, [])) - set(new.get(k, [])))
        gained = sorted(set(new.get(k, [])) - set(old.get(k, [])))
        if lost:
            problems.append(f"{k} lost entries {lost}")
        if gained:
            added.append(f"{k} gained {gained}")
    for k, v in new.items():
        if k not in old and isinstance(v, (int, float)) and not isinstance(v, bool):
            added.append(f"new threshold {k} = {v}")
    if tightened and new.get("policy_version") == old.get("policy_version"):
        problems.append("numeric thresholds changed but policy_version did not")
    if tightened:
        print("  thresholds tightened:")
        for t in tightened:
            print(f"    {t}")
    if added:
        print("  protections added:")
        for a in added:
            print(f"    {a}")
    return problems


def main():
    if len(sys.argv) < 2 or not sys.argv[1]:
        print("No base commit supplied (not a pull request). Policy integrity check skipped.")
        return
    base = sys.argv[1]
    files = changed_files(base)
    print(f"Files changed in this pull request: {len(files)}")
    for f in files:
        print(f"  {f}")

    hard = [f for f in files if any(f.startswith(p) for p in HARD_PROTECTED)]
    if hard:
        print("\nHARD PROTECTED FILE CHANGED:")
        for f in hard:
            print(f"  {f}")
        print("\nThe pipeline machinery, the data, and the grading tests may not be changed in a pull request.")
        sys.exit(1)

    gov = [f for f in files if f in GOVERNANCE_FILES]
    if not gov:
        print("\nNo governance change. Integrity check passed.")
        return

    print("\nGOVERNANCE CHANGE DETECTED (routed to policy owner for review):")
    for f in gov:
        print(f"  {f}")
    problems = compare_policies(base) if "governance/policy.json" in gov else []
    if problems:
        print("\nREFUSED: this pull request LOOSENS governance:")
        for p in problems:
            print(f"  - {p}")
        print("\nA model author may tighten a control or add one. Loosening needs the AI Risk Committee, in its own pull request.")
        sys.exit(1)
    print("\nGovernance change is a tightening or an addition. Allowed, pending owner review. Integrity check passed.")


if __name__ == "__main__":
    main()
