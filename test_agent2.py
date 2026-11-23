import pickle
import numpy as np
import pandas as pd

# ---- 1. Load models + scaler + FEATURE NAMES ----
with open("models/agent2_rf.pkl", "rb") as f:
    rf = pickle.load(f)
with open("models/hc_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# feature_names are stored as the 5th element of hc_data.pkl
with open("data/clean/hc_data.pkl", "rb") as f:
    _, _, _, _, feature_names = pickle.load(f)

print(f"Agent 2 loaded: {type(rf).__name__}")
print(f"  n_features_in_ = {rf.n_features_in_}")
print(f"Scaler loaded: {type(scaler).__name__}")
print(f"Feature names count = {len(feature_names)}")

# Sanity: confirm the columns we care about exist
for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
            "CNT_CHILDREN", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
    assert col in feature_names, f"MISSING COLUMN: {col}"
print("✅ All required columns present in feature_names.")

# ---- 2. Helper: build a DataFrame row ----
def make_row(income, credit, annuity, children, gender, car, realty):
    row = {name: 0.0 for name in feature_names}  # baseline = 0 for everything
    row["AMT_INCOME_TOTAL"] = float(income)
    row["AMT_CREDIT"]       = float(credit)
    row["AMT_ANNUITY"]      = float(annuity)
    row["CNT_CHILDREN"]     = float(children)
    row["CODE_GENDER"]      = 1.0 if gender == "M" else 0.0
    row["FLAG_OWN_CAR"]     = 1.0 if car == "Y" else 0.0
    row["FLAG_OWN_REALTY"]  = 1.0 if realty == "Y" else 0.0
    # NOTE: remaining ~72 columns stay at 0. In scaled space that's
    # (0 - mean) / std — a neutral-ish baseline. Good enough for a demo.
    return pd.DataFrame([row], columns=feature_names)

# ---- 3. Score GOOD profile ----
good = make_row(
    income=300_000, credit=50_000, annuity=3_000, children=0,
    gender="F", car="Y", realty="Y",
)
scaled_good = scaler.transform(good)
p_good = rf.predict_proba(scaled_good)[0]
print(f"\nGOOD profile:")
print(f"  P(repay=0)   = {p_good[0]:.4f}")
print(f"  P(default=1) = {p_good[1]:.4f}")

# ---- 4. Score BAD profile ----
bad = make_row(
    income=20_000, credit=1_000_000, annuity=80_000, children=3,
    gender="M", car="N", realty="N",
)
scaled_bad = scaler.transform(bad)
p_bad = rf.predict_proba(scaled_bad)[0]
print(f"\nBAD profile:")
print(f"  P(repay=0)   = {p_bad[0]:.4f}")
print(f"  P(default=1) = {p_bad[1]:.4f}")

# ---- 5. Assert the model behaves monotonically ----
print(f"\nΔP(default) BAD - GOOD = {p_bad[1] - p_good[1]:+.4f}")
assert p_bad[1] > p_good[1], (
    f"❌ Model still doesn't differentiate profiles. "
    f"GOOD={p_good[1]:.4f}, BAD={p_bad[1]:.4f}"
)
print("✅ Agent 2 behaves correctly — bad profile gets higher default risk.")