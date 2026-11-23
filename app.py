
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Aura Credit Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00F0FF, #7000FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .sub-title {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-top: 0px;
        margin-bottom: 30px;
        font-weight: 300;
    }

    .glass-card {
        background: rgba(26, 28, 36, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }

    .stNumberInput label,
    .stSelectbox label,
    .stCheckbox label {
        color: #00F0FF !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00F0FF 0%, #7000FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 240, 255, 0.4);
    }

    .st-key-agent1_load_buttons [data-testid="column"]:nth-child(1) .stButton>button {
        background: #16A34A !important;
        color: white !important;
    }

    .st-key-agent1_load_buttons [data-testid="column"]:nth-child(2) .stButton>button {
        background: #DC2626 !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1A1C24;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #9CA3AF;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0E1117;
        color: #00F0FF !important;
        border-top: 3px solid #00F0FF;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. SESSION STATE ---
if 'rnn_fraud_proba' not in st.session_state:
    st.session_state['rnn_fraud_proba'] = 0.0

if 'tx_label' not in st.session_state:
    st.session_state['tx_label'] = None

if 'rnn_initialized' not in st.session_state:
    st.session_state['rnn_initialized'] = False

if 'rf_initialized' not in st.session_state:
    st.session_state['rf_initialized'] = False

if 'rf_label' not in st.session_state:
    st.session_state['rf_label'] = None


# --- 4. LOAD MODELS & DATA ---
@st.cache_resource
def load_models():
    m = {}

    if os.path.exists('models/agent1_rnn.h5'):
        m['rnn'] = tf.keras.models.load_model(
            'models/agent1_rnn.h5'
        )

    if os.path.exists('models/agent2_rf.pkl'):
        with open('models/agent2_rf.pkl', 'rb') as f:
            m['rf'] = pickle.load(f)

    if os.path.exists('models/agent3_lr.pkl'):
        with open('models/agent3_lr.pkl', 'rb') as f:
            m['lr'] = pickle.load(f)

    if os.path.exists('models/hc_scaler.pkl'):
        with open('models/hc_scaler.pkl', 'rb') as f:
            m['scaler'] = pickle.load(f)

    if os.path.exists('models/hc_feature_names.pkl'):
        with open('models/hc_feature_names.pkl', 'rb') as f:
            m['feature_names'] = pickle.load(f)

    return m


@st.cache_resource
def load_cc_data():
    with open('data/clean/cc_data.pkl', 'rb') as f:
        _, X_test, _, y_test = pickle.load(f)

    return X_test, y_test


@st.cache_resource
def load_cc_stats():
    with open('data/clean/cc_data.pkl', 'rb') as f:
        X_train, _, y_train, _ = pickle.load(f)

    normal = X_train[y_train == 0].values
    fraud = X_train[y_train == 1].values

    return {
        'normal_mean': normal.mean(axis=0),
        'normal_std': normal.std(axis=0),
        'fraud_mean': fraud.mean(axis=0),
        'fraud_std': fraud.std(axis=0),
        'normal_count': len(normal),
        'fraud_count': len(fraud)
    }


models = load_models()
cc_X_test, cc_y_test = load_cc_data()
cc_stats = load_cc_stats()


# --- 5. CREDIT CARD FEATURE NAMES (AGENT 1) ---
CC_FEATURE_NAMES = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'scaled_amount', 'scaled_time'
]


# --- 6. HOMECREDIT GROUPS (AGENT 2) ---
FEATURE_GROUPS = {
    "Financial": [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE"
    ],

    "Socio-Professional": [
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_INCOME_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY"
    ],

    "Housing & Region": [
        "REGION_POPULATION_RELATIVE",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "HOUR_APPR_PROCESS_START",
        "WEEKDAY_APPR_PROCESS_START"
    ],

    "Documents": [],
    "Credit Bureau": []
}


if 'feature_names' in models:
    known = set()

    for feats in FEATURE_GROUPS.values():
        known.update(feats)

    for f in models['feature_names']:
        if f in known:
            continue

        if f.startswith("FLAG_DOCUMENT"):
            FEATURE_GROUPS["Documents"].append(f)

        elif f.startswith("AMT_REQ_CREDIT_BUREAU") or f.startswith("EXT_SOURCE"):
            FEATURE_GROUPS["Credit Bureau"].append(f)

        elif f.startswith("FLAG_"):
            FEATURE_GROUPS["Socio-Professional"].append(f)

        else:
            FEATURE_GROUPS.setdefault("Other", []).append(f)


# --- 7. AGENT 1 HELPERS ---
def load_real_tx(kind):

    if kind == 'normal':
        idx = cc_y_test[cc_y_test == 0].index[0]
        row = cc_X_test.loc[idx].values
        label = "NORMAL (real)"

    else:
        idx = cc_y_test[cc_y_test == 1].index[0]
        row = cc_X_test.loc[idx].values
        label = "FRAUD (real)"

    for i, f in enumerate(CC_FEATURE_NAMES):
        st.session_state[f"rnn_{f}"] = float(row[i])

    st.session_state['tx_label'] = label
    st.rerun()


def set_rnn_transaction(kind='normal', randomize=False):

    if randomize:
        if kind == 'normal':
            vals = np.random.normal(
                cc_stats['normal_mean'],
                cc_stats['normal_std']
            )
            label = "RANDOM NORMAL"

        else:
            vals = np.random.normal(
                cc_stats['fraud_mean'],
                cc_stats['fraud_std']
            )
            label = "RANDOM FRAUD"

    else:
        if kind == 'normal':
            idx = cc_y_test[cc_y_test == 0].index[0]
            label = "NORMAL (real)"

        else:
            idx = cc_y_test[cc_y_test == 1].index[0]
            label = "FRAUD (real)"

        vals = cc_X_test.loc[idx].values

    for i, f in enumerate(CC_FEATURE_NAMES):
        st.session_state[f"rnn_{f}"] = float(vals[i])

    st.session_state['tx_label'] = label


def random_rnn(kind='mix'):
    set_rnn_transaction(kind=kind, randomize=True)
    st.rerun()


def build_rnn_input():
    vals = [
        st.session_state.get(f"rnn_{f}", 0.0)
        for f in CC_FEATURE_NAMES
    ]

    return np.array(vals, dtype=np.float32).reshape(1, 1, 30)


# --- 8. AGENT 2 HELPERS ---
def set_rf(feature, value):
    st.session_state[f"rf_{feature}"] = float(value)


def initialize_all_rf_features():
    """
    Initialize all Home Credit features using the fitted scaler's
    training means. This prevents unspecified features from remaining
    artificially initialized to zero.
    """

    feature_names = models['feature_names']
    scaler = models.get('scaler')

    means = getattr(scaler, "mean_", None)

    if means is None or len(means) != len(feature_names):
        raise RuntimeError(
            "The Home Credit scaler does not contain matching training means. "
            "Check hc_scaler.pkl and hc_feature_names.pkl."
        )

    for feature, value in zip(feature_names, means):
        st.session_state[f"rf_{feature}"] = float(value)


# SAFE APPLICANT

def load_safe_applicant():

    feature_names = models['feature_names']
    scaler = models['scaler']
    rf_model = models['rf']

    means = scaler.mean_

    best_row = None
    best_risk = float("inf")

    # Try 1,000 different financially strong applicant profiles
    for _ in range(1000):

        candidate = {
            f: float(v)
            for f, v in zip(feature_names, means)
        }

        # Financial profile
        income = np.random.uniform(250_000, 500_000)
        credit = np.random.uniform(50_000, 200_000)
        annuity = np.random.uniform(4_000, 12_000)

        safe_values = {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": credit * np.random.uniform(0.90, 1.0),
            "CNT_CHILDREN": np.random.randint(0, 2),
            "CNT_FAM_MEMBERS": np.random.randint(1, 3),
            "DAYS_BIRTH": np.random.uniform(-25_000, -15_000),
            "DAYS_EMPLOYED": np.random.uniform(-10_000, -2_000),
            "REGION_RATING_CLIENT": 1,
            "REGION_RATING_CLIENT_W_CITY": 1,
        }

        for feature, value in safe_values.items():
            if feature in candidate:
                candidate[feature] = float(value)

        # Strong external-source scores
        for f in feature_names:
            if f.startswith("EXT_SOURCE"):
                candidate[f] = np.random.uniform(0.75, 0.99)

            elif f.startswith("AMT_REQ_CREDIT_BUREAU"):
                candidate[f] = 0

        # Evaluate this candidate using the real model
        candidate_df = pd.DataFrame(
            [candidate],
            columns=feature_names
        )

        scaled_candidate = scaler.transform(candidate_df)

        risk = float(
            rf_model.predict_proba(scaled_candidate)[0][1]
        )

        # Keep the candidate with the lowest predicted risk
        if risk < best_risk:
            best_risk = risk
            best_row = candidate.copy()

        # Stop early if the profile is below the 25% threshold
        if best_risk <= 0.25:
            break

    # Load the best profile into Streamlit
    for feature, value in best_row.items():
        st.session_state[f"rf_{feature}"] = float(value)

    if best_risk <= 0.25:
        st.session_state['rf_label'] = "SAFE Applicant"
    else:
        st.session_state['rf_label'] = (
            f"Best Available Applicant ({best_risk * 100:.2f}% risk)"
        )

# RISKY APPLICANT
def load_risky_applicant():

    initialize_all_rf_features()

    set_rf("AMT_INCOME_TOTAL", 30_000)
    set_rf("AMT_CREDIT", 1_200_000)
    set_rf("AMT_ANNUITY", 80_000)
    set_rf("AMT_GOODS_PRICE", 1_000_000)

    set_rf("CNT_CHILDREN", 4)
    set_rf("CNT_FAM_MEMBERS", 6)

    set_rf("CODE_GENDER", 1)
    set_rf("FLAG_OWN_CAR", 0)
    set_rf("FLAG_OWN_REALTY", 0)

    set_rf("DAYS_BIRTH", -8_000)
    set_rf("DAYS_EMPLOYED", -100)

    for f in models['feature_names']:

        if f.startswith("EXT_SOURCE"):
            st.session_state[f"rf_{f}"] = 0.10

        if f.startswith("FLAG_DOCUMENT"):
            st.session_state[f"rf_{f}"] = 0

    st.session_state['rf_label'] = "RISKY Applicant"


# RANDOM APPLICANT
def load_random_applicant():

    flavor = np.random.choice(
        ['safe', 'risky', 'average'],
        p=[0.4, 0.4, 0.2]
    )

    if flavor == 'safe':

        income = np.random.uniform(200_000, 400_000)
        credit = np.random.uniform(50_000, 150_000)
        annuity = np.random.uniform(2_000, 8_000)
        children = np.random.randint(0, 2)

        gender = np.random.choice([0, 1], p=[0.6, 0.4])
        car = np.random.choice([0, 1], p=[0.2, 0.8])
        realty = np.random.choice([0, 1], p=[0.2, 0.8])

        dob = np.random.uniform(-25_000, -15_000)
        emp = np.random.uniform(-10_000, -3_000)

        ext = np.random.uniform(0.6, 0.95)
        doc = np.random.choice([0, 1], p=[0.1, 0.9])
        req = np.random.randint(0, 2)
        rating = np.random.randint(1, 3)

    elif flavor == 'risky':

        income = np.random.uniform(20_000, 60_000)
        credit = np.random.uniform(800_000, 1_500_000)
        annuity = np.random.uniform(50_000, 100_000)
        children = np.random.randint(2, 6)

        gender = np.random.choice([0, 1], p=[0.4, 0.6])
        car = np.random.choice([0, 1], p=[0.8, 0.2])
        realty = np.random.choice([0, 1], p=[0.8, 0.2])

        dob = np.random.uniform(-10_000, -7_000)
        emp = np.random.uniform(-500, 0)

        ext = np.random.uniform(0.02, 0.2)
        doc = np.random.choice([0, 1], p=[0.8, 0.2])
        req = np.random.randint(3, 8)
        rating = np.random.randint(2, 4)

    else:

        income = np.random.uniform(80_000, 180_000)
        credit = np.random.uniform(200_000, 600_000)
        annuity = np.random.uniform(15_000, 35_000)
        children = np.random.randint(0, 3)

        gender = np.random.choice([0, 1])
        car = np.random.choice([0, 1])
        realty = np.random.choice([0, 1])

        dob = np.random.uniform(-20_000, -10_000)
        emp = np.random.uniform(-5_000, -500)

        ext = np.random.uniform(0.3, 0.7)
        doc = np.random.choice([0, 1])
        req = np.random.randint(0, 4)
        rating = np.random.randint(1, 4)

    # Initialize ALL features first
    initialize_all_rf_features()

    # Override selected features with generated applicant values
    set_rf("AMT_INCOME_TOTAL", income)
    set_rf("AMT_CREDIT", credit)
    set_rf("AMT_ANNUITY", annuity)

    set_rf(
        "AMT_GOODS_PRICE",
        credit * np.random.uniform(0.85, 1.05)
    )

    set_rf("CNT_CHILDREN", children)
    set_rf(
        "CNT_FAM_MEMBERS",
        children + np.random.randint(1, 3)
    )

    set_rf("CODE_GENDER", gender)
    set_rf("FLAG_OWN_CAR", car)
    set_rf("FLAG_OWN_REALTY", realty)

    set_rf("DAYS_BIRTH", dob)
    set_rf("DAYS_EMPLOYED", emp)
    set_rf("REGION_RATING_CLIENT", rating)

    for f in models['feature_names']:

        if f.startswith("EXT_SOURCE"):
            st.session_state[f"rf_{f}"] = ext + np.random.uniform(-0.05, 0.05)

        if f.startswith("FLAG_DOCUMENT"):
            st.session_state[f"rf_{f}"] = doc

        if f.startswith("AMT_REQ_CREDIT_BUREAU"):
            st.session_state[f"rf_{f}"] = req

    st.session_state['rf_label'] = (
        f"RANDOM Applicant ({flavor.upper()})"
    )


def build_rf_input():

    row = {
        f: float(st.session_state.get(f"rf_{f}", 0.0))
        for f in models['feature_names']
    }

    return pd.DataFrame(
        [row],
        columns=models['feature_names']
    )


# --- 9. INIT ---
if not st.session_state['rnn_initialized'] and 'rnn' in models:

    for f in CC_FEATURE_NAMES:
        st.session_state.setdefault(f"rnn_{f}", 0.0)

    st.session_state['rnn_initialized'] = True


if not st.session_state['rf_initialized'] and 'feature_names' in models:

    scaler_means = getattr(models.get('scaler'), "mean_", None)

    for i, f in enumerate(models['feature_names']):

        default_value = (
            float(scaler_means[i])
            if scaler_means is not None
            else 0.0
        )

        st.session_state.setdefault(
            f"rf_{f}",
            default_value
        )

    st.session_state['rf_initialized'] = True


# --- 10. HEADER ---
st.markdown(
    '<p class="main-title">AURA CREDIT ENGINE</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Multi-Agent Decentralized Solvency & Fraud Analysis Pipeline</p>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs([
    "AGENT 1: TRANSACTION FRAUD (RNN)",
    "AGENTS 2 & 3: LOAN SOLVENCY (RF + LR)"
])


# ==========================================
# TAB 1: AGENT 1 (RNN)
# ==========================================
with tab1:

    st.markdown(
        "### Transaction Fraud Check — Agent 1 (LSTM)"
    )

    st.markdown(
        f"**Dataset:** Credit Card Fraud — "
        f"{cc_stats['normal_count']:,} normal / "
        f"{cc_stats['fraud_count']:,} fraud in training."
    )

    st.markdown(
        "The RNN uses **30 features** from the Credit Card dataset. "
        "You can edit any of them below."
    )

    with st.container(key="agent1_load_buttons"):

        btn_cols = st.columns(3)

        with btn_cols[0]:

            if st.button(
                "Load REAL Normal",
                key="btn_safe",
                use_container_width=True
            ):
                load_real_tx('normal')

        with btn_cols[1]:

            if st.button(
                "Load REAL Fraud",
                key="btn_fraud_load",
                use_container_width=True
            ):
                load_real_tx('fraud')

        with btn_cols[2]:

            if st.button(
                "Random FRAUD",
                key="btn_random_fraud",
                use_container_width=True
            ):
                random_rnn('fraud')

    if st.session_state.get('tx_label'):

        st.markdown(
            f"**Currently loaded:** `{st.session_state['tx_label']}`"
        )

    st.markdown(
        "#### Agent 1 Input Features (30 editable boxes)"
    )

    rnn_cols = st.columns(5)

    for i, f in enumerate(CC_FEATURE_NAMES):

        with rnn_cols[i % 5]:

            st.number_input(
                label=f,
                step=0.01,
                format="%.4f",
                key=f"rnn_{f}"
            )

    st.write("---")

    if st.button(
        "RUN FRAUD DIAGNOSTICS",
        key="btn_fraud",
        type="primary"
    ):

        if 'rnn' in models:

            with st.spinner(
                "Agent 1 analyzing the 30 features..."
            ):

                input_rnn = build_rnn_input()

                proba = float(
                    models['rnn'].predict(
                        input_rnn,
                        verbose=0
                    )[0][0]
                )

                st.session_state['rnn_fraud_proba'] = proba

            st.markdown(
                '<div class="glass-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "#### Agent 1 — LSTM Fraud Score"
            )

            st.markdown(
                f"**Fraud probability:** `{proba*100:.4f}%`"
            )

            st.markdown(
                f"**Raw score:** `{proba:.6f}`"
            )

            if proba > 0.5:

                st.error(
                    "FRAUD ALERT — BLOCK TRANSACTION"
                )

            else:

                conf = (1 - proba) * 100

                st.success(
                    "SAFE — APPROVE TRANSACTION"
                )

                st.info(
                    f"The model is **{conf:.4f}% confident** "
                    "that this transaction is legitimate."
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ==========================================
# TAB 2: AGENT 2 & 3
# ==========================================
with tab2:

    if 'feature_names' not in models:

        st.error(
            "Run `05_agent3_lr.ipynb` first to generate "
            "`hc_feature_names.pkl`."
        )

        st.stop()

    st.markdown(
        "### Loan Solvency Assessment"
    )

    st.markdown(
        "The Random Forest uses **79 features**. "
        "Use one of the 3 buttons below."
    )

    st.info(
        "Note on decision threshold: Because of the 92/8 class "
        "imbalance in Home Credit, the Random Forest was trained "
        "with `class_weight='balanced'`. Its effective decision "
        "threshold is **25%** (not 50%). Any default risk "
        "**> 25%** is considered RISKY."
    )

    # IMPORTANT:
    # Callbacks execute BEFORE Streamlit recreates the widgets.
    # They modify Agent 2 keys only, never Agent 1 keys.

    with st.container(key="agent2_load_buttons"):

        b1, b2, b3 = st.columns(3)

        with b1:

            st.button(
                "Load SAFE Applicant",
                key="btn_rf_safe",
                use_container_width=True,
                on_click=load_safe_applicant
            )

        with b2:

            st.button(
                "Load RISKY Applicant",
                key="btn_rf_risky",
                use_container_width=True,
                on_click=load_risky_applicant
            )

        with b3:

            st.button(
                "Random Applicant",
                key="btn_rf_random",
                use_container_width=True,
                on_click=load_random_applicant
            )

    if st.session_state.get('rf_label'):

        st.markdown(
            f"**Currently loaded:** `{st.session_state['rf_label']}`"
        )

    # AGENT 2 INPUT TABS
    group_names = [
        g for g in FEATURE_GROUPS
        if FEATURE_GROUPS[g]
    ]

    input_tabs = st.tabs(group_names)

    for g_name, g_tab in zip(group_names, input_tabs):

        with g_tab:

            cols = st.columns(3)

            for i, f in enumerate(FEATURE_GROUPS[g_name]):

                with cols[i % 3]:

                    st.number_input(
                        label=f,
                        step=0.01,
                        format="%.4f",
                        key=f"rf_{f}"
                    )

    st.write("---")

    if st.button(
        "RUN SOLVENCY ASSESSMENT",
        key="btn_loan",
        type="primary"
    ):

        with st.spinner(
            "Processing multi-agent pipeline..."
        ):

            raw_df = build_rf_input()

            scaled_df = models['scaler'].transform(raw_df)

            rf_proba = float(
                models['rf'].predict_proba(scaled_df)[0][1]
            )

            rnn_score = st.session_state['rnn_fraud_proba']

            # AGENT 3: LOGISTIC REGRESSION
            if 'lr' in models:

                lr_input = np.array([
                    [rnn_score, rf_proba]
                ])

                final_proba = float(
                    models['lr'].predict_proba(lr_input)[0][1]
                )

                reason = (
                    "Decision fused by Agent 3 "
                    "(Logistic Regression)."
                )

            else:

                final_proba = (
                    0.99
                    if (rnn_score > 0.5 or rf_proba > 0.25)
                    else 0.1
                )

                reason = (
                    "Agent 3 missing — fallback rule applied "
                    "(threshold 25%)."
                )

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Final Loan Decision Pipeline"
        )

        # 1. AGENT 1
        st.markdown(
            f"**1. Fraud Risk (Agent 1 — LSTM):** "
            f"`{rnn_score*100:.4f}%`"
        )

        if rnn_score > 0.5:

            st.error(
                "Flagged for fraud."
            )

        else:

            st.success(
                "Passed fraud check."
            )

        # 2. AGENT 2
        st.markdown(
            f"**2. Default Risk (Agent 2 — RF):** "
            f"`{rf_proba*100:.4f}%` "
            "*(threshold: 25%)*"
        )

        if rf_proba > 0.25:

            st.error(
                "High default risk (above 25% threshold)."
            )

        else:

            st.success(
                "Acceptable profile (below 25% threshold)."
            )

        st.write("---")

        # 3. AGENT 3
        st.markdown(
            f"**3. Final Decision (Agent 3 — LR):** "
            f"`{final_proba*100:.4f}%`"
        )

        if final_proba > 0.5:

            st.markdown(
                "<h2 style='color:#FF4B4B;'>LOAN REJECTED</h2>",
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                "<h2 style='color:#00F0FF;'>LOAN APPROVED</h2>",
                unsafe_allow_html=True
            )

        st.write(reason)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ==========================================
# SIDEBAR: SYSTEM STATUS
# ==========================================
with st.sidebar:

    st.markdown(
        "<h2 style='color:#00F0FF;'>SYSTEM STATUS</h2>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    status_items = [
        ("Agent 1 (LSTM)", "rnn" in models),
        ("Agent 2 (RF)", "rf" in models),
        ("Agent 3 (LR)", "lr" in models),
        ("HC Scaler", "scaler" in models),
        ("HC Features", "feature_names" in models)
    ]

    for label, available in status_items:

        status_color = (
            "#16A34A" if available else "#DC2626"
        )

        status_text = (
            "READY" if available else "MISSING"
        )

        st.markdown(
            f"<p style='margin: 0.35rem 0; "
            f"color: {status_color}; font-weight: 600;'>"
            f"{label}: {status_text}</p>",
            unsafe_allow_html=True
        )
