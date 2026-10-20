# BANA 690A Midterm: Written Answers

**Name: Kathlyn Miller-Francis
**Last-name group (A to H, I to P, Q to Z): M
**GitHub username: KathlynMiller

This file is half of your exam. The code and the pipeline grade the other half.
Write in full sentences. Cite evidence from THIS repository: a file name, a line,
a number from `artifacts/gate_report.json` or `artifacts/model_card.md`.
Open book. Individual. Keep each answer short and specific.

---

## Lab 1: Make the data compliant

**1.1** All four failures were detective controls: `test_no_zip_like_column` detected `zip`; `test_no_date_like_column` detected `date_of_birth`; `test_no_suspiciously_predictive_column` detected `adjuster_fraud_review` at correlation 0.845 above the 0.60 limit; and `test_policy_agrees_with_data_dictionary` found those columns missing from the policy lists. The list controls (`test_no_pii_columns` and `test_no_known_leakage_columns`) passed; detective controls find omissions the lists do not already cover.

**1.2** The data dictionary classifies `zip` and `date_of_birth` as quasi-identifiers, so I added them to `pii_columns`. It classifies `adjuster_fraud_review` as an outcome recorded after the policy year, so I added it to `leakage_columns`. The Lab 1 test run then reported 11 passed.

**1.3** `src/prepare.py` already removes the columns listed by the policy; the failures showed that the policy lists were incomplete. Updating the policy fixes the missing governance decisions without changing the policy-driven preparation code.

---

## Lab 2: Build features from driving history

Record every feature you implemented. Four minimum, four families minimum, including both families assigned to your last-name group.

| Feature name | Family | Formula in one sentence | Why it should predict claim_free |
|---|---|---|---|
| claims_per_month | frequency | Sum `claim_filed` over months 1-24 divided by 24. | Frequent prior claims can indicate higher future claim risk. |
| months_since_last_claim | recency | 25 minus the latest claim month, or 25 if no claim occurred. | A more recent claim can indicate current driving risk. |
| speeding_velocity | velocity | Recent-six-month speeding events per 1,000 miles divided by the 24-month rate plus 0.1. | A rise in speeding relative to a driver's baseline can signal worsening risk. |
| miles_volatility | volatility | Population standard deviation of monthly miles divided by mean monthly miles over months 1-24. | Unstable driving exposure may be associated with less predictable risk. |
| max_hard_brakes_per_1k | extremes | Maximum of each month's `hard_brakes / miles * 1000` over months 1-24, with a small mileage stabilizer. | A driver's worst rate of harsh braking may signal hazardous behavior. |
| hard_brake_trend | trend | Second-half aggregate hard-brakes-per-1,000-miles rate minus first-half rate. | A positive trend indicates worsening harsh-braking behavior. |
| night_share_mean | ratio | Average `night_miles_share` across months 1-24. | Greater night-driving exposure may be associated with increased claim risk. |

All seven defined feature families are implemented, covering the two families assigned to my I-to-P surname group even though the slides specifying that pair are not present in the repository. All features apply `as_of()`; velocity also uses `recent()` for months 19-24, with `DECISION_MONTH` set to 24.

**2.1** The temporal fence test adds poisoned history for months 25 to 30 and checks that the features do not change; features must use only months 1 to 24, before the decision. A feature that failed would rely on information unavailable at application time, so offline evaluation could be misleading and the feature could not be computed correctly in production.

---

## Lab 3: Hunt the leakage

Three suspect columns. For each, give your ruling and the evidence.

| Column | Ruling (leakage or allowed) | Correlation with label | When recorded | Outcome / rural proxy evidence and reasoning |
|---|---|---|---|---|
| prior_claims_3y | Allowed | 0.243 | At application | Generated from aggressiveness and new-driver status (`data/generate_data.py:86`), not rural status; rural/urban means are 0.229/0.253. It is pre-decision risk history. |
| renewal_offered | Leakage | 0.549 | End of policy year | Generated conditionally on `claim_free` (`data/generate_data.py:141`), with fill rates 0.959 for claim-free and 0.500 for claim-filing drivers. It is post-decision and indirectly reflects the outcome's rural penalty (rural/urban means 0.706/0.866). |
| annual_mileage_declared | Allowed | 0.280 | At application | The generator adds `2600 * rural` (`data/generate_data.py:73`); measured rural/urban means are 13,675/11,002. It is available at decision time but is a rural proxy that warrants fairness monitoring. |

**3.1** `renewal_offered` is leakage even though its correlation is 0.549, below the 0.60 threshold. Correlation only measures association strength, not when the value becomes known; the data-dictionary timeline test catches that it is recorded at the end of the policy year.

---

## Lab 4: Run v1 and v2, explain the block

**4.1** Both normal runs passed Stage 0 (18 tests). v1 (logistic regression, six features) had overall AUC 0.7046; rural AUC/approval/TPR/FPR were 0.6617/0.7749/0.8647/0.6860 and urban values were 0.6960/0.8951/0.9430/0.7277. Its approval-disparity, TPR, and FPR ratios were 0.8657/0.9170/0.9427; `DECISION: PASS`. v2 (gradient boosting, seven features) had overall AUC 0.7224; rural AUC/approval/TPR/FPR were 0.5937/0.5146/0.5588/0.4709 and urban values were 0.6835/0.9196/0.9490/0.8168. Its ratios were 0.5596/0.5888/0.5765; `DECISION: BLOCKED`.

**4.2** The key config difference is that v2 adds `territory_risk_index` and uses gradient boosting. The index is generated as `42 + 30 * rural + noise` at `data/generate_data.py:136`, so it acts as a rural proxy; the historical label applies an unjustified rural penalty (`-1.10 * rural`) at `data/generate_data.py:132`. v2's overall AUC is higher (0.7224 vs. 0.7046), but it fails rural AUC at 0.5937 vs. 0.64 (0.0463 below), rural TPR at 0.5588 vs. 0.70 (0.1412 below), approval disparity at 0.5596 vs. 0.80 (0.2404 below), and TPR ratio at 0.5888 vs. 0.80 (0.2112 below).

**4.3** The preventive check only asks whether a feature's name appears in the prohibited list. `territory_risk_index` is not listed, so it passes, even though it tracks rurality. The detective checks compare model outcomes for rural and urban drivers; they catch the harm: rural AUC is 0.5937 (below 0.64), rural TPR is 0.5588 (below 0.70), approval disparity is 0.5596 (below 0.80), and the TPR ratio is 0.5888 (below 0.80). The name check screens known prohibited inputs; group outcome checks reveal proxy effects the list cannot anticipate.

---

## Lab 5: Change the governance

**5.1** Not completed: I have not added `min_fpr_ratio` or selected a threshold in `governance/policy.json`.

**5.2** No threshold value has been chosen, so there is no selected value to defend. A false positive is a driver who would remain claim-free but is routed away from the standard rate; this can cost Sierra Mutual a customer and unfairly burden the driver with manual review or a higher rate. Unequal FPR means similarly situated drivers who would file a claim are treated differently by area, creating unequal access and burdens as well as different business costs.

**5.3** Not completed: `test_gate_blocks_fpr_gap` still calls `pytest.skip`. The intended test should prove directly that `run_gate` blocks an FPR ratio below the policy threshold and passes one at or above it; running the pipeline alone would not isolate or verify both sides of that gate rule.

---

## Lab 6: Beat the gate

**6.1** There is no final v3 gate result: `model_config.json` is still a placeholder and no engineered features are implemented. The diagnostic v2 run (with tests bypassed) reported overall AUC 0.7224, disparity ratio 0.5596, TPR ratio 0.5888, FPR ratio 0.5765, and `DECISION: BLOCKED`.

**6.2** None. `src/features.py` currently reports zero implemented features, and the model config contains only non-engineered features, so there is no feature-importance comparison to report.

**6.3** Yes. The diagnostic v2 config includes `territory_risk_index`; with tests bypassed, overall AUC rose from v1's 0.7046 to 0.7224, but v2 was blocked for poor rural subgroup results and disparate approvals. A higher average score does not make a model deployable when it fails governance and fairness checks.

---

## Lab 7: Request deployment

Fill in the pull request template on GitHub. Then answer here:

**7.1** The repository routes reviews of governance files to `@pouyan1361` in `.github/CODEOWNERS`, identifies the Model Risk Officer as policy owner in `governance/policy.json`, and runs the workflow on pull requests to `main` (`.github/workflows/governance-gate.yml`). The CI evidence is the Stage A report of changed files and policy protections. I do not have a GitHub run here confirming the specific "awaiting approval" pause.

**7.2** No approved GitHub run or Stage A output is available to quote. The success-path message in `ci/check_policy_integrity.py` is: "Governance change is a tightening or an addition. Allowed, pending owner review. Integrity check passed." However, this working tree also changes `tests/test_lab1_compliance.py`, which Stage A lists as hard-protected; a PR containing that file change would be refused before reaching that success message.

---

## Lab 8 (bonus): Break it and watch it stop

**8.1** No second branch or local v2 run without bypassing tests is recorded. The diagnostic v2 run used `--skip-tests`; it printed overall AUC 0.7224 and `DECISION: BLOCKED`. The normal v2 command had previously stopped at Stage 0 before training.

**8.2** No GitHub Stage A output is available. For a threshold-loosening PR, the refusal heading in `ci/check_policy_integrity.py` is `REFUSED: this pull request LOOSENS governance:`; the actual run-specific refusal line is not recorded.

**8.3** The deployment gate alone would not stop a loosened policy threshold because it evaluates metrics against whatever thresholds the policy currently contains. Stage A separately detects and refuses governance loosening; both controls are needed so a model must meet policy and the policy itself cannot be silently weakened.

---

## Declaration

I completed this exam individually, using only the permitted materials.

**Signature (type your name):**

---

## Change / Terminal Log

For each change or fix, record the date, what changed, the terminal command run,
and its output. For documentation-only changes with no command run, say so.

### 2026-10-20

**Change:** Added a change/fix log to `tests/test_lab1_compliance.py`.

**Command:** `python -m pytest -q tests/test_lab1_compliance.py`

**Terminal output:**

```text
FAILED test_no_zip_like_column: ZIP-like column: ['zip']
FAILED test_no_date_like_column: date-like column: ['date_of_birth']
FAILED test_no_suspiciously_predictive_column: adjuster_fraud_review correlation: 0.845 (limit: 0.6)
FAILED test_policy_agrees_with_data_dictionary: missing pii_columns: ['zip', 'date_of_birth']; missing leakage_columns: ['adjuster_fraud_review']
4 failed, 7 passed in 3.68s
```

### 2026-10-20

**Change:** Added this change/terminal log and its recording convention.

**Command:** Not run (documentation-only change).

### 2026-10-20

**Change:** Recorded the Python dependency installation output.

**Command:** `pip install -r requirements.txt`

**Terminal output:**

```text
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pandas>=1.5 in /home/vscode/.local/lib/python3.11/site-packages (from -r requirements.txt (line 1)) (3.0.6)
Requirement already satisfied: numpy>=1.23 in /home/vscode/.local/lib/python3.11/site-packages (from -r requirements.txt (line 2)) (2.4.6)
Requirement already satisfied: scikit-learn>=1.2 in /home/vscode/.local/lib/python3.11/site-packages (from -r requirements.txt (line 3)) (1.9.1)
Requirement already satisfied: pytest>=7.0 in /home/vscode/.local/lib/python3.11/site-packages (from -r requirements.txt (line 4)) (9.1.1)
Requirement already satisfied: python-dateutil>=2.8.2 in /home/vscode/.local/lib/python3.11/site-packages (from pandas>=1.5->-r requirements.txt (line 1)) (2.9.0.post0)
Requirement already satisfied: scipy>=1.10.0 in /home/vscode/.local/lib/python3.11/site-packages (from scikit-learn>=1.2->-r requirements.txt (line 3)) (1.17.1)
Requirement already satisfied: joblib>=1.4.0 in /home/vscode/.local/lib/python3.11/site-packages (from scikit-learn>=1.2->-r requirements.txt (line 3)) (1.6.0)
Requirement already satisfied: narwhals>=2.0.1 in /home/vscode/.local/lib/python3.11/site-packages (from scikit-learn>=1.2->-r requirements.txt (line 3)) (2.27.1)
Requirement already satisfied: threadpoolctl>=3.5.0 in /home/vscode/.local/lib/python3.11/site-packages (from scikit-learn>=1.2->-r requirements.txt (line 3)) (3.7.0)
Requirement already satisfied: iniconfig>=1.0.1 in /home/vscode/.local/lib/python3.11/site-packages (from pytest>=7.0->-r requirements.txt (line 4)) (2.3.1)
Requirement already satisfied: packaging>=22 in /usr/local/lib/python3.11/site-packages (from pytest>=7.0->-r requirements.txt (line 4)) (26.3)
Requirement already satisfied: pluggy<2,>=1.5 in /home/vscode/.local/lib/python3.11/site-packages (from pytest>=7.0->-r requirements.txt (line 4)) (1.7.0)
Requirement already satisfied: pygments>=2.7.2 in /home/vscode/.local/lib/python3.11/site-packages (from pytest>=7.0->-r requirements.txt (line 4)) (2.21.0)
Requirement already satisfied: cloudpickle>=3.0 in /home/vscode/.local/lib/python3.11/site-packages (from joblib>=1.4.0->scikit-learn>=1.2->-r requirements.txt (line 3)) (3.1.2)
Requirement already satisfied: six>=1.5 in /home/vscode/.local/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas>=1.5->-r requirements.txt (line 1)) (1.17.0)

[notice] A new release of pip is available: 24.0 -> 26.2.1
[notice] To update, run: pip install --upgrade pip
```

**Exit code:** `0`

### 2026-10-20

**Change:** Added the missing Lab 1 policy entries and completed the evidence-based written answers.

**Command:** `python -m pytest -q tests/test_lab1_compliance.py`

**Terminal output:**

```text
...........                                                              [100%]
11 passed in 1.30s
```

**Command:** `python -m pytest -q tests/`

**Terminal output:**

```text
FAILED test_every_suspect_column_has_a_ruling
FAILED test_author_is_filled_in
FAILED test_enough_features_implemented
FAILED test_model_uses_engineered_features
4 failed, 22 passed, 1 skipped in 1.79s
```

**Diagnostic pipeline commands:**

```text
python run_pipeline.py --skip-tests --config configs/v1.json
overall AUC 0.7046; DECISION: PASS

python run_pipeline.py --skip-tests --config configs/v2.json
overall AUC 0.7224; rural AUC 0.5937; rural TPR 0.5588
disparity ratio 0.5596; TPR ratio 0.5888; FPR ratio 0.5765
DECISION: BLOCKED
```

These diagnostic runs bypassed Stage 0 and are not completed submissions.

**Leakage check command:** `python src/leakage_checks.py`

**Relevant output:** `prior_claims_3y` correlation 0.243, application; `renewal_offered` 0.549, end of policy year; `annual_mileage_declared` 0.280, application.

**Command:** Not run for the written-answer changes (documentation-only).

### 2026-10-20

**Change:** Clarified Lab 1 detective-versus-list control types and data-dictionary reasons.

**Command:** Not run (documentation-only change).

### 2026-10-20

**Change:** Implemented all seven Lab 2 features from pre-decision driving history, with guarded denominators and the maximum monthly hard-braking rate for the extremes feature.

**Command:** `python src/features.py`

**Relevant output:**

```text
implemented: ['claims_per_month', 'months_since_last_claim', 'speeding_velocity', 'miles_volatility', 'max_hard_brakes_per_1k', 'hard_brake_trend', 'night_share_mean']
families: ['extremes', 'frequency', 'ratio', 'recency', 'trend', 'velocity', 'volatility']
```

**Command:** `python -m pytest -q tests/test_lab2_features.py`

**Terminal output:**

```text
....                                                                     [100%]
4 passed in 0.79s
```

**Edge-case check:** The synthetic zero-mile case stayed finite with zero rates; a 10-hard-brake/1-mile month produced 10,000 hard brakes per 1,000 miles, confirming max-of-monthly-ratios behavior.

### 2026-10-20

**Change:** Matched recency to `(DECISION_MONTH + 1) - last_claim_month` and set the velocity baseline stabilizer to `0.1`; expanded the Lab 3 notes with outcome and rural-proxy evidence.

**Command:** `python -m pytest -q tests/test_lab2_features.py`

**Terminal output:**

```text
....                                                                     [100%]
4 passed in 0.79s
```

**Formula check output:** `month1=24`, `month24=1`, `never=25`, `zero_baseline_velocity=0.0`.

**Command:** `python -m pytest -q tests/test_lab3_leakage.py`

**Terminal output:**

```text
...                                                                      [100%]
3 passed in 1.31s
```

### 2026-10-20

**Change:** Replaced the Lab 4 diagnostic-only answers with results from both normal pipeline runs and expanded the proxy explanation.

**Commands:** `python run_pipeline.py --config configs/v1.json`; `python run_pipeline.py --config configs/v2.json`

**Results:** Both passed Stage 0 with 18 tests. v1 AUC 0.7046, disparity 0.8657, TPR ratio 0.9170, FPR ratio 0.9427, `DECISION: PASS`. v2 AUC 0.7224, disparity 0.5596, TPR ratio 0.5888, FPR ratio 0.5765, `DECISION: BLOCKED` (rural AUC 0.5937; rural TPR 0.5588).

### 2026-10-20

**Change:** Reworded the Lab 4.3 explanation to distinguish the name-based preventive check from group-outcome detective checks.

**Command:** Not run (documentation-only change).
