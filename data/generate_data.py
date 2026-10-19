"""
Generates the synthetic auto insurance dataset used in the midterm.

Everything here is fictional. Names, license numbers, emails, phones and
addresses are fabricated. You do not need to run this file: the three CSV
files are already in the repo. It is included so you can see exactly how the
data, and its built-in problems, were created. Reading it is allowed and
encouraged: an auditor reads the data lineage too.

The business
  Sierra Mutual sells one-year auto policies. At application time the
  underwriter must decide whether to offer the STANDARD rate (approve) or
  route the applicant to manual review and a higher rate (decline).
  The model predicts claim_free: 1 if the driver filed NO at-fault claim
  during the policy year, 0 if they did. Score >= threshold means approve.

Three files
  policies_raw.csv            one row per applicant, known at application time
                              PLUS several columns that should not be there
  driving_history.csv         24 months of telematics BEFORE the application
                              (month 1 is oldest, month 24 is the month before
                              the decision). This is where you build features.
  history_after_decision.csv  months 25 to 30: telematics DURING the policy
                              year. It exists so you can prove your features
                              never touch it. Correct code never reads it.

Built-in problems (on purpose)
  1. Direct identifiers in the raw file (name, license number, email, phone,
     street address).
  2. Quasi-identifiers in the raw file (zip and date_of_birth). The shipped
     policy forgot them.
  3. Two post-year columns that leak the label: claim_paid_amount (listed in
     the shipped policy) and adjuster_fraud_review (NOT listed).
  4. A third column, renewal_offered, that is recorded at the END of the
     policy year. It is leakage too, but it is not correlated strongly enough
     to trip the automatic correlation check. Only a timeline argument finds it.
  5. prior_claims_3y is strongly related to the label but is known at
     application time. It is legitimate. Not everything strong is leakage.
  6. The historical label carries a penalty against rural drivers that their
     driving does not justify (think: past underwriting practice).
  7. A vendor "territory_risk_index" is mostly a rural signal, so a model can
     learn the rural penalty without seeing the area column. That is a proxy.
"""
import numpy as np
import pandas as pd

SEED = 690
N = 4000
HIST_MONTHS = 24
AFTER_MONTHS = 6

FIRST = ["Alex", "Maria", "James", "Linh", "Omar", "Sofia", "Daniel", "Aisha", "Kevin", "Priya",
         "Carlos", "Emma", "Hiro", "Fatima", "Noah", "Grace", "Mateo", "Olivia", "Ravi", "Chloe"]
LAST = ["Garcia", "Nguyen", "Smith", "Patel", "Kim", "Lopez", "Johnson", "Chen", "Martinez", "Brown",
        "Hernandez", "Davis", "Singh", "Wilson", "Ramirez", "Taylor", "Ali", "Moore", "Lee", "Clark"]
STREETS = ["Sierra Dr", "Granite Ave", "Aspen Ct", "Ridge Rd", "Meadow Ln", "Canyon Blvd", "Summit Way"]
COVERAGE = ["basic", "standard", "full"]
VEHICLE_CLASS = ["sedan", "suv", "pickup", "compact"]


def generate(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    # ----- who the applicant is -----
    area = np.where(rng.random(n) < 0.70, "urban", "rural")
    rural = (area == "rural").astype(float)
    gender = rng.choice(["F", "M"], n)

    years_licensed = np.clip(rng.gamma(3.0, 5.0, n), 0.5, 55).round(1)
    new_driver = (years_licensed < 3).astype(float)
    vehicle_age = np.clip(rng.gamma(2.5, 2.4, n), 0, 25).round(0)
    vehicle_value = np.clip(np.exp(rng.normal(9.9 - 0.045 * vehicle_age, 0.45, n)), 1500, 90000).round(-2)
    annual_mileage_declared = np.clip(rng.normal(11000 + 2600 * rural, 3400, n), 1500, 40000).round(-2)
    prior_insurer_years = np.clip(rng.gamma(1.8, 2.2, n), 0, 30).round(1)
    coverage_type = rng.choice(COVERAGE, n, p=[0.25, 0.45, 0.30])
    vehicle_class = rng.choice(VEHICLE_CLASS, n, p=[0.35, 0.30, 0.15, 0.20])

    # ----- hidden driving style (never observed directly) -----
    # aggressiveness drives hard brakes and speeding; night_tendency drives night miles.
    aggressiveness = rng.normal(0, 1, n) + 0.35 * new_driver
    night_tendency = rng.normal(0, 1, n)
    # some drivers are getting worse over the 24 months (deterioration trend)
    deterioration = rng.normal(0, 1, n)

    # prior at-fault claims in the 3 years before the history window (known at application)
    prior_claims_3y = rng.poisson(np.clip(0.25 + 0.18 * aggressiveness + 0.1 * new_driver, 0.02, 2.0))

    # ----- 24 months of telematics before the decision -----
    monthly_base_miles = annual_mileage_declared / 12.0
    rows = []
    for m in range(1, HIST_MONTHS + AFTER_MONTHS + 1):
        t = (m - 1) / (HIST_MONTHS - 1)          # 0 at month 1, 1 at month 24
        season = 1.0 + 0.08 * np.sin(2 * np.pi * m / 12.0)
        miles = np.clip(monthly_base_miles * season * np.exp(rng.normal(0, 0.22, n)), 50, None).round(0)
        trips = np.clip((miles / 9.5) * np.exp(rng.normal(0, 0.15, n)), 5, None).round(0)
        agg_now = aggressiveness + 0.6 * deterioration * t
        hard_brakes = rng.poisson(np.clip(np.exp(0.9 + 0.55 * agg_now) * miles / 1000.0, 0.05, None))
        speeding = rng.poisson(np.clip(np.exp(0.2 + 0.65 * agg_now) * miles / 1000.0, 0.02, None))
        night_share = np.clip(0.14 + 0.07 * night_tendency + rng.normal(0, 0.04, n), 0.0, 0.8).round(3)
        # a prior at-fault claim in this month (rare), more likely for aggressive drivers
        p_claim = np.clip(0.006 * np.exp(0.5 * agg_now), 0.001, 0.08)
        claim_this_month = (rng.random(n) < p_claim).astype(int)
        rows.append(pd.DataFrame({
            "policy_id": [f"POL{i:05d}" for i in range(1, n + 1)],
            "month": m,
            "miles": miles,
            "trips": trips,
            "hard_brakes": hard_brakes,
            "speeding_events": speeding,
            "night_miles_share": night_share,
            "claim_filed": claim_this_month,
        }))
    history = pd.concat(rows, ignore_index=True)
    before = history[history["month"] <= HIST_MONTHS].copy()
    after = history[history["month"] > HIST_MONTHS].copy()

    # ----- true risk from driving behavior and known facts -----
    recent = before[before["month"] > HIST_MONTHS - 6].groupby("policy_id")
    recent_speeding_rate = (recent["speeding_events"].sum() / recent["miles"].sum() * 1000).reindex(
        [f"POL{i:05d}" for i in range(1, n + 1)]).values
    z = (-0.95 * aggressiveness
         - 0.55 * deterioration
         - 0.50 * (np.log(annual_mileage_declared) - np.log(11500)) / 0.3
         - 0.55 * prior_claims_3y
         - 0.50 * new_driver
         - 0.30 * night_tendency
         + 0.10 * (prior_insurer_years - 4) / 3.5
         - 0.08 * (vehicle_value / 1000 - 20) / 12)

    # Historical label: claim_free in the policy year, with a rural penalty that
    # the driving does not justify (the bias the model can inherit).
    logit = 1.35 + z - 1.10 * rural + rng.normal(0, 1.05, n)
    claim_free = (logit > 0).astype(int)

    # Vendor territory risk index: mostly an area signal, plus noise
    territory_risk_index = np.clip(42 + 30 * rural + rng.normal(0, 8, n), 0, 100).round(1)

    # ----- columns recorded AFTER the decision (leakage) -----
    paid = np.where(claim_free == 0, np.exp(rng.normal(8.1, 0.8, n)).round(-1), 0.0)
    adjuster_fraud_review = np.where(claim_free == 0, rng.random(n) < 0.82, rng.random(n) < 0.02).astype(int)
    renewal_offered = np.where(claim_free == 1, rng.random(n) < 0.96, rng.random(n) < 0.52).astype(int)

    # ----- identifiers and quasi-identifiers -----
    names = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]
    license_no = [f"{rng.choice(list('ABCDEFGH'))}{rng.integers(1000000, 9999999)}" for _ in range(n)]
    email = [f"{nm.lower().replace(' ', '.')}{rng.integers(1, 999)}@example.com" for nm in names]
    phone = [f"(818) 555-{rng.integers(1000, 10000):04d}" for _ in range(n)]
    street = [f"{rng.integers(10, 9999)} {rng.choice(STREETS)}" for _ in range(n)]
    zips = np.where(rural == 1, rng.integers(95900, 96199, n), rng.integers(90001, 91999, n)).astype(str)
    birth = pd.to_datetime("1950-01-01") + pd.to_timedelta(rng.integers(365 * 10, 365 * 58, n), unit="D")

    policies = pd.DataFrame({
        "policy_id": [f"POL{i:05d}" for i in range(1, n + 1)],
        "applicant_name": names,
        "drivers_license_no": license_no,
        "email": email,
        "phone": phone,
        "street_address": street,
        "zip": zips,
        "date_of_birth": birth.strftime("%Y-%m-%d"),
        "gender": gender,
        "area": area,
        "years_licensed": years_licensed,
        "vehicle_age": vehicle_age,
        "vehicle_value": vehicle_value,
        "vehicle_class": vehicle_class,
        "annual_mileage_declared": annual_mileage_declared,
        "prior_insurer_years": prior_insurer_years,
        "coverage_type": coverage_type,
        "prior_claims_3y": prior_claims_3y,
        "territory_risk_index": territory_risk_index,
        "claim_paid_amount": paid,
        "adjuster_fraud_review": adjuster_fraud_review,
        "renewal_offered": renewal_offered,
        "claim_free": claim_free,
    })

    # a little realistic missingness in two declared fields
    miss = rng.random(n) < 0.04
    policies.loc[miss, "prior_insurer_years"] = np.nan
    miss2 = rng.random(n) < 0.03
    policies.loc[miss2, "annual_mileage_declared"] = np.nan

    return policies, before.reset_index(drop=True), after.reset_index(drop=True)


if __name__ == "__main__":
    import pathlib
    here = pathlib.Path(__file__).resolve().parent
    policies, before, after = generate()
    policies.to_csv(here / "policies_raw.csv", index=False)
    before.to_csv(here / "driving_history.csv", index=False)
    after.to_csv(here / "history_after_decision.csv", index=False)
    print(f"policies {policies.shape}  history {before.shape}  after {after.shape}")
    print(f"claim_free rate {policies.claim_free.mean():.3f}   rural share {(policies.area == 'rural').mean():.3f}")
    print(policies.groupby("area")["claim_free"].mean().round(3).to_dict())
    num = policies.select_dtypes("number")
    print("corr with label:", num.corrwith(policies.claim_free).abs().sort_values(ascending=False).round(3).head(8).to_dict())
