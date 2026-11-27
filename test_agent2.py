"""Smoke test for the isolated Agent 2 (Random Forest) contribution.

Run from the project root:

    python test_agent2.py

The script verifies that all saved inference artifacts load, that their feature
counts agree, and that the fitted scaler -> model pipeline produces predictions
for a synthetic applicant profile.  If the cleaned dataset is present locally it
also checks that a real test row survives a scaler round-trip unchanged; this
check is skipped when the (large) dataset has not been downloaded.
"""

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "agent2_rf.pkl"
SCALER_PATH = ROOT / "models" / "hc_scaler.pkl"
FEATURES_PATH = ROOT / "models" / "hc_feature_names.pkl"
DATA_PATH = ROOT / "data" / "clean" / "hc_data.pkl"


def load_pickle(path):
    assert path.is_file(), f"Missing required artifact: {path}"
    with path.open("rb") as handle:
        return pickle.load(handle)


# Artifacts were created with scikit-learn 1.7.2.  A newer installed version can
# still load them, but emits InconsistentVersionWarning; suppress only that warning.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", InconsistentVersionWarning)
    rf = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
feature_names = load_pickle(FEATURES_PATH)

assert type(rf).__name__ == "RandomForestClassifier", type(rf)
assert type(scaler).__name__ == "StandardScaler", type(scaler)
assert isinstance(feature_names, list) and len(feature_names) > 0
assert rf.n_features_in_ == scaler.n_features_in_ == len(feature_names), (
    f"Artifact feature mismatch: model={rf.n_features_in_}, "
    f"scaler={scaler.n_features_in_}, names={len(feature_names)}"
)
print(f"Model loaded:     {type(rf).__name__} ({rf.n_features_in_} features)")
print(f"Scaler loaded:    {type(scaler).__name__} ({scaler.n_features_in_} features)")
print(f"Feature names:    {len(feature_names)} entries")

for column in (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_CHILDREN",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
):
    assert column in feature_names, f"Missing expected feature: {column}"
print("Required borrower features are present.")


def synthetic_profile(income, credit, annuity, children, gender, car, realty):
    """Build a SYNTHETIC raw applicant row (not real customer data)."""
    row = {name: 0.0 for name in feature_names}
    row["AMT_INCOME_TOTAL"] = float(income)
    row["AMT_CREDIT"] = float(credit)
    row["AMT_ANNUITY"] = float(annuity)
    row["CNT_CHILDREN"] = float(children)
    row["CODE_GENDER"] = 1.0 if gender == "M" else 0.0
    row["FLAG_OWN_CAR"] = 1.0 if car == "Y" else 0.0
    row["FLAG_OWN_REALTY"] = 1.0 if realty == "Y" else 0.0
    return pd.DataFrame([row], columns=feature_names)


def default_probability(raw_frame):
    """Inference order must be: fitted scaler first, forest second."""
    scaled = scaler.transform(raw_frame)
    assert scaled.shape == (1, len(feature_names))
    probabilities = rf.predict_proba(scaled)[0]
    prediction = int(rf.predict(scaled)[0])
    assert np.isclose(probabilities.sum(), 1.0)
    return probabilities, prediction


good = synthetic_profile(
    income=300_000, credit=50_000, annuity=3_000, children=0,
    gender="F", car="Y", realty="Y",
)
bad = synthetic_profile(
    income=20_000, credit=1_000_000, annuity=80_000, children=3,
    gender="M", car="N", realty="N",
)
p_good, label_good = default_probability(good)
p_bad, label_bad = default_probability(bad)

print("\nSynthetic GOOD profile")
print(f"  P(repay=0)   = {p_good[0]:.4f}")
print(f"  P(default=1) = {p_good[1]:.4f}   predicted class = {label_good}")
print("Synthetic BAD profile")
print(f"  P(repay=0)   = {p_bad[0]:.4f}")
print(f"  P(default=1) = {p_bad[1]:.4f}   predicted class = {label_bad}")
print(f"Delta P(default) BAD - GOOD = {p_bad[1] - p_good[1]:+.4f}")
assert p_bad[1] > p_good[1], "Synthetic risky profile did not get higher default risk."

# Optional: validate a legitimate held-out row when the cleaned dataset exists.
# A Git LFS pointer is only ~134 bytes; ignore it rather than trying to unpickle it.
if DATA_PATH.is_file() and DATA_PATH.stat().st_size > 1000:
    with DATA_PATH.open("rb") as handle:
        _, X_test, _, y_test, _ = pickle.load(handle)
    scaled_sample = np.asarray(X_test)[:1]
    raw_sample = scaler.inverse_transform(scaled_sample)
    sample_frame = pd.DataFrame(raw_sample, columns=feature_names)
    rescaled = scaler.transform(sample_frame)
    assert np.allclose(rescaled, scaled_sample, atol=1e-6), "Scaler round-trip failed."
    probabilities, prediction = default_probability(sample_frame)
    direct_probability = float(rf.predict_proba(scaled_sample)[0][1])
    assert np.isclose(probabilities[1], direct_probability, atol=1e-6)
    true_label = np.asarray(y_test)[0]
    print(
        f"\nReal held-out row: true TARGET={int(true_label)}, "
        f"P(default)={probabilities[1]:.4f}, predicted class={prediction} "
        "(scaler round-trip verified)"
    )
else:
    print("\nCleaned dataset not present locally; optional real-row check skipped.")

print("\nPASS: Agent 2 artifacts load and the scaler -> model pipeline predicts.")
