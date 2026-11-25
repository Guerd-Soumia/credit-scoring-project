"""
Aura Credit Engine - interface of the Credit Scoring Project

Agent 1: RNN (LSTM)        -> fraud score from one transaction
Agent 2: Random Forest     -> default score from one applicant profile
Agent 3: rules + local LLM -> decision (approved / rejected) + justification

Run from the project folder:   streamlit run app.py
Expected folders:  data/clean/  and  models/
"""

import os
import pickle

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf

# --------------------------------------------------------------------------
# 0. SETTINGS (same values as 06_coordinator.ipynb)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRAUD_THRESHOLD = 0.5     # Agent 1: precision 0.80, recall 0.83 at this threshold
RF_THRESHOLD = 0.5        # Agent 2: precision 0.17, recall 0.61 at this threshold

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"

BAD_PHRASES = ["i can't", "i can\u2019t", "i cannot", "i'm sorry", "i am sorry",
               "can't assist", "as an ai", "loan officer", "here is", "here are"]


def path(*parts):
    return os.path.join(BASE_DIR, *parts)


# --------------------------------------------------------------------------
# 1. PAGE CONFIG + CSS
# --------------------------------------------------------------------------
st.set_page_config(page_title="Aura Credit Engine", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    .main-title {
        font-size: 3.2rem; font-weight: 800; margin-bottom: 0px;
        background: -webkit-linear-gradient(45deg, #00F0FF, #7000FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-title { color: #9CA3AF; font-size: 1.05rem; margin-top: 0; margin-bottom: 24px; font-weight: 300; }

    .glass-card {
        background: rgba(26, 28, 36, 0.7); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 22px 24px; margin: 14px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .decision-ok  { color: #16A34A; font-size: 2.1rem; font-weight: 800; margin: 4px 0; }
    .decision-bad { color: #FF4B4B; font-size: 2.1rem; font-weight: 800; margin: 4px 0; }
    .tag { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem;
           font-weight: 600; background: rgba(0,240,255,0.12); color: #00F0FF; }
    .tag-warn { background: rgba(245,158,11,0.15); color: #F59E0B; }

    .stNumberInput label { color: #00F0FF !important; font-size: 0.75rem !important; font-weight: 600 !important; }

    .stButton>button {
        background: linear-gradient(90deg, #00F0FF 0%, #7000FF 100%); color: white; border: none;
        border-radius: 8px; padding: 10px 20px; font-weight: 600; width: 100%;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,240,255,0.4); }

    .stTabs [data-baseweb="tab"] { background-color: #1A1C24; border-radius: 8px 8px 0 0; padding: 10px 18px; color: #9CA3AF; }
    .stTabs [aria-selected="true"] { background-color: #0E1117; color: #00F0FF !important; border-top: 3px solid #00F0FF; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 2. LOADING (models, feature names, test data)
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    m = {}
    if os.path.exists(path("models", "agent1_rnn.h5")):
        # compile=False: we only predict, so no need to rebuild loss/metrics
        m["rnn"] = tf.keras.models.load_model(path("models", "agent1_rnn.h5"), compile=False)
    if os.path.exists(path("models", "agent2_rf.pkl")):
        with open(path("models", "agent2_rf.pkl"), "rb") as f:
            m["rf"] = pickle.load(f)
    if os.path.exists(path("models", "hc_scaler.pkl")):
        with open(path("models", "hc_scaler.pkl"), "rb") as f:
            m["scaler"] = pickle.load(f)

    # Feature names: 02_clean saves them inside hc_data.pkl (5th element).
    # A separate models/hc_feature_names.pkl is used only if it exists.
    names = None
    if os.path.exists(path("models", "hc_feature_names.pkl")):
        with open(path("models", "hc_feature_names.pkl"), "rb") as f:
            names = pickle.load(f)
    elif os.path.exists(path("data", "clean", "hc_data.pkl")):
        with open(path("data", "clean", "hc_data.pkl"), "rb") as f:
            names = pickle.load(f)[4]
    if names is not None:
        m["feature_names"] = list(names)
    return m


@st.cache_resource
def load_test_data():
    """Only TEST rows are used in the demo (the models never saw them)."""
    d = {}
    try:
        with open(path("data", "clean", "cc_data.pkl"), "rb") as f:
            _, X_test, _, y_test = pickle.load(f)
        d["cc_X"], d["cc_y"] = X_test, y_test
    except Exception:
        pass
    try:
        with open(path("data", "clean", "hc_data.pkl"), "rb") as f:
            _, X_test, _, y_test, _ = pickle.load(f)     # X_test is already scaled
        d["hc_X"], d["hc_y"] = X_test, y_test
    except Exception:
        pass
    return d


models = load_models()
data = load_test_data()

CC_FEATURES = list(data["cc_X"].columns) if "cc_X" in data else \
    [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
HC_FEATURES = models.get("feature_names", [])


@st.cache_resource
def example_positions():
    """Pick clear example rows from the test sets (same idea as 06_coordinator)."""
    ex = {}
    if "rnn" in models and "cc_X" in data:
        y = data["cc_y"].values
        X = data["cc_X"].values
        ex["cc_normal"] = int(np.where(y == 0)[0][0])
        fraud_pos = np.where(y == 1)[0]
        probs = models["rnn"].predict(X[fraud_pos].reshape(-1, 1, X.shape[1]), verbose=0).flatten()
        ex["cc_fraud"] = int(fraud_pos[np.argmax(probs)])        # most obvious fraud
    if "rf" in models and "hc_X" in data:
        probs = models["rf"].predict_proba(data["hc_X"])[:, 1]
        ex["hc_safe"] = int(np.argmin(probs))
        ex["hc_risky"] = int(np.argmax(probs))
    return ex


# --------------------------------------------------------------------------
# 3. FEATURE GROUPS (Agent 2 inputs)
# --------------------------------------------------------------------------
def build_feature_groups(feature_names):
    groups = {
        "Financial": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"],
        "Socio-Professional": [
            "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
            "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "DAYS_BIRTH",
            "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "CODE_GENDER",
            "FLAG_OWN_CAR", "FLAG_OWN_REALTY"],
        "Housing & Region": [
            "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
            "HOUR_APPR_PROCESS_START", "WEEKDAY_APPR_PROCESS_START"],
        "Documents": [],
        "Credit Bureau": [],
        "Other": [],
    }
    names = set(feature_names)
    for g in groups:                       # keep only features that really exist
        groups[g] = [f for f in groups[g] if f in names]
    known = {f for fs in groups.values() for f in fs}
    for f in feature_names:
        if f in known:
            continue
        if f.startswith("FLAG_DOCUMENT"):
            groups["Documents"].append(f)
        elif f.startswith("AMT_REQ_CREDIT_BUREAU") or f.startswith("EXT_SOURCE"):
            groups["Credit Bureau"].append(f)
        elif f.startswith("FLAG_"):
            groups["Socio-Professional"].append(f)
        else:
            groups["Other"].append(f)
    return {g: fs for g, fs in groups.items() if fs}


# --------------------------------------------------------------------------
# 4. SESSION STATE (widget defaults)
# --------------------------------------------------------------------------
for f in CC_FEATURES:
    st.session_state.setdefault(f"rnn_{f}", 0.0)

if HC_FEATURES:
    means = getattr(models.get("scaler"), "mean_", None)
    for i, f in enumerate(HC_FEATURES):
        default = float(means[i]) if means is not None and len(means) == len(HC_FEATURES) else 0.0
        st.session_state.setdefault(f"rf_{f}", default)

st.session_state.setdefault("tx_label", None)
st.session_state.setdefault("rf_label", None)
st.session_state.setdefault("result", None)


# --------------------------------------------------------------------------
# 5. LOAD-EXAMPLE CALLBACKS (run before the widgets are redrawn)
# --------------------------------------------------------------------------
def load_transaction(pos, label):
    row = data["cc_X"].values[pos]
    for i, f in enumerate(CC_FEATURES):
        st.session_state[f"rnn_{f}"] = float(row[i])
    st.session_state["tx_label"] = label


def load_applicant(pos, label):
    scaled_row = data["hc_X"][pos].reshape(1, -1)
    raw_row = models["scaler"].inverse_transform(scaled_row)[0]     # back to real units
    for f, v in zip(HC_FEATURES, raw_row):
        st.session_state[f"rf_{f}"] = float(v)
    st.session_state["rf_label"] = label


def cb_tx_normal():
    load_transaction(example_positions()["cc_normal"], "NORMAL transaction (real test row)")


def cb_tx_fraud():
    load_transaction(example_positions()["cc_fraud"], "FRAUD transaction (real test row, highest score)")


def cb_tx_random():
    pos = int(np.random.randint(len(data["cc_X"])))
    kind = "FRAUD" if int(data["cc_y"].values[pos]) == 1 else "NORMAL"
    load_transaction(pos, f"RANDOM real test transaction (true label: {kind})")


def cb_rf_safe():
    load_applicant(example_positions()["hc_safe"], "SAFEST applicant (real test row)")


def cb_rf_risky():
    load_applicant(example_positions()["hc_risky"], "RISKIEST applicant (real test row)")


def cb_rf_random():
    pos = int(np.random.randint(len(data["hc_X"])))
    kind = "DEFAULT" if int(data["hc_y"].values[pos]) == 1 else "REPAID"
    load_applicant(pos, f"RANDOM real test applicant (true outcome: {kind})")


# --------------------------------------------------------------------------
# 6. AGENTS
# --------------------------------------------------------------------------
def run_agent1():
    x = np.array([st.session_state[f"rnn_{f}"] for f in CC_FEATURES], dtype=np.float32)
    return float(models["rnn"].predict(x.reshape(1, 1, -1), verbose=0)[0][0])


def run_agent2():
    row = {f: float(st.session_state[f"rf_{f}"]) for f in HC_FEATURES}
    raw = pd.DataFrame([row], columns=HC_FEATURES)
    scaled = models["scaler"].transform(raw)
    return float(models["rf"].predict_proba(scaled)[0][1]), scaled[0]


def agent3_rules(prob_fraud, prob_default):
    """Fraud is checked first. Returns (decision, code, reason)."""
    if prob_fraud > FRAUD_THRESHOLD:
        return "REJECTED", "fraud", "High fraud risk in transactions"
    if prob_default > RF_THRESHOLD:
        return "REJECTED", "default", "High default risk based on profile"
    return "APPROVED", "ok", "Low fraud risk and low default risk"


def factor_sentence(name, v):
    """Plain-language meaning of one standardized feature (v = std vs average)."""
    if name.startswith("EXT_SOURCE"):
        label = name.replace("EXT_SOURCE_", "external credit score ")
        if v > 0.5:
            return f"{label} is higher than average (favorable)"
        if v < -0.5:
            return f"{label} is lower than average (unfavorable)"
        return f"{label} is close to average"
    if name == "DAYS_EMPLOYED":      # negative days: more negative = employed for longer
        if v < -0.5:
            return "employment length is longer than average (favorable)"
        if v > 0.5:
            return "employment length is shorter than average (unfavorable)"
        return "employment length is close to average"
    if v > 0.5:
        return f"{name} is higher than average"
    if v < -0.5:
        return f"{name} is lower than average"
    return f"{name} is close to average"


def top_factors(scaled_row):
    """The 3 most important features of the Random Forest, in plain words."""
    idx = np.argsort(models["rf"].feature_importances_)[-3:][::-1]
    return [factor_sentence(HC_FEATURES[i], float(scaled_row[i])) for i in idx]


def facts_text(code, fraud_pct, default_pct, factors):
    """Exact facts, written by Python (never by the LLM)."""
    F, R = FRAUD_THRESHOLD * 100, RF_THRESHOLD * 100
    fac = "; ".join(factors)
    if code == "fraud":
        return (f"Fraud risk is {fraud_pct}, above the {F:.0f}% limit; "
                f"default risk is {default_pct}. Profile factors: {fac}.")
    if code == "default":
        return (f"Default risk is {default_pct}, above the {R:.0f}% limit; "
                f"fraud risk is {fraud_pct}. Profile factors: {fac}.")
    return (f"Fraud risk is {fraud_pct} (limit {F:.0f}%) and default risk is {default_pct} "
            f"(limit {R:.0f}%). Profile factors: {fac}.")


def template_text(code, fraud_pct, default_pct, factors):
    """Safe fallback justification, built only from real facts."""
    F, R = FRAUD_THRESHOLD * 100, RF_THRESHOLD * 100
    fac = "; ".join(factors)
    if code == "fraud":
        return (f"The application is rejected: the fraud risk of {fraud_pct} is above the "
                f"{F:.0f}% limit. The default risk is {default_pct}. Profile factors: {fac}.")
    if code == "default":
        return (f"The application is rejected: the default risk of {default_pct} is above the "
                f"{R:.0f}% limit. The fraud risk is low ({fraud_pct}). Profile factors: {fac}.")
    return (f"The application is approved: the fraud risk ({fraud_pct}) and the default risk "
            f"({default_pct}) are below the limits. Profile factors: {fac}.")


def clean_llm_text(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines and (lines[0].lower().startswith("here") or lines[0].endswith(":")):
        lines = lines[1:]
    return " ".join(lines).strip().strip('"')


def sentence_is_valid(sentence, decision, code):
    """Same checks as 06_coordinator: the LLM may only announce the decision."""
    low = sentence.lower()
    ok = 20 <= len(sentence) <= 250
    ok = ok and not any(ch.isdigit() for ch in sentence)
    ok = ok and not any(p in low for p in BAD_PHRASES)
    if decision == "APPROVED":
        ok = ok and "approv" in low and "reject" not in low and "declin" not in low
    else:
        ok = (ok and "reject" in low and "is approved" not in low
              and "been approved" not in low and "be approved" not in low)
        if code == "default":
            ok = ok and "fraud" not in low
        if code == "fraud":
            ok = ok and "default" not in low
    return ok


def ask_llm(decision, reason):
    verb = "approved" if decision == "APPROVED" else "rejected"
    prompt = (f"Write ONE short, professional sentence announcing that a loan application is {verb}.\n"
              f"The reason is: {reason.lower()}.\n"
              "Do not use any digits or numbers. Do not mention any other fact.\n"
              "Output only the sentence.")
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0.4, "num_predict": 60}},
                      timeout=120)
    r.raise_for_status()
    return clean_llm_text(r.json()["response"])


def justify(decision, code, reason, fraud_pct, default_pct, factors, use_llm=True):
    """LLM writes ONE sentence, Python adds the facts. Template if anything fails."""
    if use_llm:
        try:
            sentence = ask_llm(decision, reason)
            if sentence_is_valid(sentence, decision, code):
                return (f"{sentence} {facts_text(code, fraud_pct, default_pct, factors)}",
                        "LLM sentence + exact facts")
        except Exception:
            pass
    return template_text(code, fraud_pct, default_pct, factors), "template (fallback)"


@st.cache_data(ttl=10)
def ollama_running():
    try:
        requests.get(OLLAMA_URL, timeout=1.5)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# 7. HEADER
# --------------------------------------------------------------------------
st.markdown('<p class="main-title">AURA CREDIT ENGINE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Multi-agent credit scoring: fraud (RNN) + default risk '
            '(Random Forest) + decision (rules and local LLM)</p>', unsafe_allow_html=True)

with st.expander("How this demo works (read before presenting)"):
    st.markdown(
        "- **Agent 1 (RNN / LSTM)** reads one transaction (30 numbers) and gives a **fraud score**.\n"
        "- **Agent 2 (Random Forest)** reads one applicant profile (79 numbers) and gives a **default score**.\n"
        "- **Agent 3 (rules + local LLM)** takes the decision. **Rules decide**: fraud above 50% is "
        "rejected first, then default above 50% is rejected, otherwise approved. The local LLM only "
        "writes **one opening sentence**; every number comes from Python. If the LLM fails or its "
        "sentence does not pass the checks, a template text is used and the decision stays the same.\n"
        "- The two datasets describe **different people** and share no customer ID, so the customer "
        "in this demo is a **simulation** (one transaction + one profile chosen by you).\n"
        "- The Random Forest score is **not a real probability** (class_weight='balanced'): "
        "a high score means *risky compared with others*."
    )

tab1, tab2, tab3 = st.tabs([
    "1 | AGENT 1: TRANSACTION FRAUD (RNN)",
    "2 | AGENT 2: LOAN APPLICANT (RANDOM FOREST)",
    "3 | AGENT 3: FINAL DECISION (RULES + LLM)",
])

# ==========================================================================
# TAB 1: AGENT 1
# ==========================================================================
with tab1:
    st.markdown("### Transaction fraud check: Agent 1 (LSTM)")
    st.markdown(f"The RNN reads **{len(CC_FEATURES)} features** of one card transaction. "
                "V1 to V28 are PCA components (not interpretable); amount and time are standardized. "
                "You can edit every value.")

    if "rnn" not in models:
        st.error("Agent 1 is missing: run `03_agent1_rnn.ipynb` to create `models/agent1_rnn.h5`.")
    elif "cc_X" not in data:
        st.warning("`data/clean/cc_data.pkl` not found: run `02_clean.ipynb`. "
                   "You can still type the values by hand.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.button("Load NORMAL transaction", on_click=cb_tx_normal, key="b_tx_n")
        c2.button("Load obvious FRAUD", on_click=cb_tx_fraud, key="b_tx_f")
        c3.button("Random real transaction", on_click=cb_tx_random, key="b_tx_r")

    if st.session_state["tx_label"]:
        st.markdown(f"**Currently loaded:** `{st.session_state['tx_label']}`")

    cols = st.columns(5)
    for i, f in enumerate(CC_FEATURES):
        with cols[i % 5]:
            st.number_input(f, step=0.01, format="%.4f", key=f"rnn_{f}")

    st.write("---")
    if st.button("RUN FRAUD DIAGNOSTICS", key="run_a1", disabled="rnn" not in models):
        with st.spinner("Agent 1 analyzing the transaction..."):
            p1 = run_agent1()
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Agent 1: fraud score")
        m1, m2 = st.columns(2)
        m1.metric("Fraud probability", f"{p1 * 100:.2f}%")
        m2.metric("Limit", f"{FRAUD_THRESHOLD * 100:.0f}%")
        st.progress(min(max(p1, 0.0), 1.0))
        if p1 > FRAUD_THRESHOLD:
            st.error("FRAUD ALERT: block this transaction.")
        else:
            st.success("SAFE: transaction looks legitimate.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================================
# TAB 2: AGENT 2
# ==========================================================================
with tab2:
    st.markdown("### Loan applicant profile: Agent 2 (Random Forest)")

    missing = [n for n, k in [("`agent2_rf.pkl`", "rf"), ("`hc_scaler.pkl`", "scaler"),
                              ("the feature names", "feature_names")] if k not in models]
    if missing:
        st.error("Missing: " + ", ".join(missing) + ". Run `02_clean.ipynb` and `04_agent2_rf.ipynb`.")
    else:
        st.markdown(f"The Random Forest reads **{len(HC_FEATURES)} features** of one applicant. "
                    "Values are in real units (income, days...). Text categories (education, "
                    "occupation...) are stored as label-encoded numbers.")
        st.info(f"Decision threshold: **{RF_THRESHOLD * 100:.0f}%** (chosen in the threshold analysis: "
                "precision 0.17, recall 0.61). A default score above it is considered risky. "
                "The score is not a calibrated probability.")

        if "hc_X" in data:
            b1, b2, b3 = st.columns(3)
            b1.button("Load SAFEST applicant", on_click=cb_rf_safe, key="b_rf_s")
            b2.button("Load RISKIEST applicant", on_click=cb_rf_risky, key="b_rf_r")
            b3.button("Random real applicant", on_click=cb_rf_random, key="b_rf_x")
        else:
            st.warning("`data/clean/hc_data.pkl` not found: run `02_clean.ipynb` to enable the example buttons.")

        if st.session_state["rf_label"]:
            st.markdown(f"**Currently loaded:** `{st.session_state['rf_label']}`")

        groups = build_feature_groups(HC_FEATURES)
        gtabs = st.tabs(list(groups.keys()))
        for (g_name, feats), g_tab in zip(groups.items(), gtabs):
            with g_tab:
                gcols = st.columns(3)
                for i, f in enumerate(feats):
                    with gcols[i % 3]:
                        st.number_input(f, step=0.01, format="%.4f", key=f"rf_{f}")

        st.write("---")
        if st.button("RUN SOLVENCY ASSESSMENT", key="run_a2"):
            with st.spinner("Agent 2 analyzing the profile..."):
                p2, _ = run_agent2()
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Agent 2: default score")
            m1, m2 = st.columns(2)
            m1.metric("Default score", f"{p2 * 100:.2f}%")
            m2.metric("Limit", f"{RF_THRESHOLD * 100:.0f}%")
            st.progress(min(max(p2, 0.0), 1.0))
            if p2 > RF_THRESHOLD:
                st.error("High default risk (above the limit).")
            else:
                st.success("Acceptable profile (below the limit).")
            st.caption("High score = risky compared with other applicants, not an exact chance of default.")
            st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================================
# TAB 3: AGENT 3
# ==========================================================================
with tab3:
    st.markdown("### Final decision: Agent 3 (rules + local LLM)")
    st.markdown("Agent 3 runs **both agents on the values currently typed in tabs 1 and 2**, "
                "then applies the rules and writes the justification.")

    ready = all(k in models for k in ["rnn", "rf", "scaler", "feature_names"])
    if not ready:
        st.error("Some agents are missing (see the sidebar). Run the notebooks 02, 03 and 04 first.")
    else:
        if all(st.session_state[f"rnn_{f}"] == 0.0 for f in CC_FEATURES):
            st.warning("No transaction loaded: Agent 1 would score an all-zero transaction. "
                       "Load or type one in tab 1.")
        st.caption(f"Transaction: {st.session_state['tx_label'] or 'typed by hand'}   |   "
                   f"Applicant: {st.session_state['rf_label'] or 'typed by hand / average profile'}")

        use_llm = st.checkbox("Use the local LLM for the opening sentence "
                              "(otherwise the template text is used)", value=True)

        if st.button("RUN FULL PIPELINE", key="run_all"):
            with st.spinner("Agent 1 and Agent 2 are scoring..."):
                p_fraud = run_agent1()
                p_default, scaled_row = run_agent2()
            decision, code, reason = agent3_rules(p_fraud, p_default)
            fraud_pct, default_pct = f"{p_fraud * 100:.2f}%", f"{p_default * 100:.2f}%"
            factors = top_factors(scaled_row)
            with st.spinner("Agent 3 is writing the justification..."):
                text, source = justify(decision, code, reason, fraud_pct, default_pct,
                                       factors, use_llm=use_llm and ollama_running())
            st.session_state["result"] = {
                "fraud": p_fraud, "default": p_default, "decision": decision, "code": code,
                "reason": reason, "text": text, "source": source, "factors": factors}

        res = st.session_state["result"]
        if res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Pipeline result")
            a, b = st.columns(2)
            with a:
                st.metric("Agent 1: fraud risk", f"{res['fraud'] * 100:.2f}%",
                          help=f"Limit {FRAUD_THRESHOLD * 100:.0f}%")
                st.progress(min(max(res["fraud"], 0.0), 1.0))
                (st.error if res["fraud"] > FRAUD_THRESHOLD else st.success)(
                    "Flagged for fraud." if res["fraud"] > FRAUD_THRESHOLD else "Passed the fraud check.")
            with b:
                st.metric("Agent 2: default risk", f"{res['default'] * 100:.2f}%",
                          help=f"Limit {RF_THRESHOLD * 100:.0f}%")
                st.progress(min(max(res["default"], 0.0), 1.0))
                (st.error if res["default"] > RF_THRESHOLD else st.success)(
                    "High default risk." if res["default"] > RF_THRESHOLD else "Acceptable profile.")

            st.write("---")
            if res["decision"] == "APPROVED":
                st.markdown('<p class="decision-ok">LOAN APPROVED</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="decision-bad">LOAN REJECTED</p>', unsafe_allow_html=True)
            st.markdown(f"**Rule applied:** {res['reason']}")
            st.markdown("**Justification**")
            st.write(res["text"])
            tag = "tag" if res["source"].startswith("LLM") else "tag tag-warn"
            st.markdown(f'<span class="{tag}">Source: {res["source"]}</span>', unsafe_allow_html=True)
            st.markdown("**Top 3 profile factors (Random Forest)**")
            for fct in res["factors"]:
                st.markdown(f"- {fct}")
            st.caption("The decision is taken by the rules only; the LLM never changes it.")
            st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 8. SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#00F0FF;'>SYSTEM STATUS</h2>", unsafe_allow_html=True)
    st.markdown("---")

    llm_up = ollama_running()
    items = [
        ("Agent 1 (LSTM)", "rnn" in models, "READY", "MISSING"),
        ("Agent 2 (Random Forest)", "rf" in models, "READY", "MISSING"),
        ("Agent 3 (rules)", True, "READY", "MISSING"),
        ("Local LLM (Ollama)", llm_up, "READY", "OFFLINE: template used"),
        ("HC scaler", "scaler" in models, "READY", "MISSING"),
        ("HC feature names", "feature_names" in models, "READY", "MISSING"),
        ("Test data", "cc_X" in data and "hc_X" in data, "READY", "MISSING"),
    ]
    for label, ok, yes, no in items:
        color = "#16A34A" if ok else "#DC2626"
        st.markdown(f"<p style='margin:0.35rem 0;color:{color};font-weight:600;'>"
                    f"{label}: {yes if ok else no}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Decision rules**")
    st.markdown(f"1. Fraud score above {FRAUD_THRESHOLD * 100:.0f}%: REJECTED\n"
                f"2. Default score above {RF_THRESHOLD * 100:.0f}%: REJECTED\n"
                "3. Otherwise: APPROVED")
    st.caption("Fraud is checked first. Only test rows are used for the examples.")