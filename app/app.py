import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tuned_logreg.pkl"

TOP_CATEGORIES = [
    "Biotechnology", "Software", "Curated Web", "Advertising", "E-Commerce",
    "Mobile", "Enterprise Software", "Games", "Analytics", "Health Care",
]
CATEGORY_COLS = {
    cat: "is_" + cat.lower().replace(" ", "").replace("-", "") for cat in TOP_CATEGORIES
}
CONTINENTS = ["North America", "Europe", "Asia", "South America", "Oceania", "Africa", "Other / Unknown"]
CONTINENT_COLS = {
    "North America": "is_northamerica", "Europe": "is_europe", "Asia": "is_asia",
    "South America": "is_southamerica", "Oceania": "is_oceania", "Africa": "is_africa",
}

FEATURE_ORDER = (
    ["funding_rounds", "funding_total_usd_log", "funding_missing",
     "age_last_funding_year", "funding_gap", "age_missing", "funding_velocity"]
    + list(CONTINENT_COLS.values())
    + list(CATEGORY_COLS.values())
    + ["is_othercategory"]
)

FRIENDLY_NAMES = {
    "funding_rounds": "Number of funding rounds",
    "funding_total_usd_log": "Total funding raised",
    "funding_missing": "Missing funding data",
    "age_last_funding_year": "Years to most recent round",
    "funding_gap": "Time between first and last round",
    "age_missing": "Missing founding date",
    "funding_velocity": "Funding velocity (rounds per year)",
    "is_northamerica": "Headquartered in North America",
    "is_europe": "Headquartered in Europe",
    "is_asia": "Headquartered in Asia",
    "is_southamerica": "Headquartered in South America",
    "is_oceania": "Headquartered in Oceania",
    "is_africa": "Headquartered in Africa",
    "is_othercategory": "Industry: Other",
}
for cat, col in CATEGORY_COLS.items():
    FRIENDLY_NAMES[col] = f"Industry: {cat}"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Startup Outcome Model", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');

:root {
    --paper: #fcfaf8; --ink: #18181b; --ink-muted: #3f3f46;
    --z400: #a1a1aa; --z300: #d4d4d8; --z200: #e4e4e7; --z100: #f4f4f5; --z500: #71717a;
}

.stApp { background-color: var(--paper) !important; color: var(--ink); font-family: 'Inter', sans-serif; }
.block-container { max-width: 640px; padding-top: 2.5rem; }

header[data-testid="stHeader"] { background: var(--paper) !important; box-shadow: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

.doc-title { font-family: 'Instrument Serif', serif; font-size: 2.6rem; line-height: 1.05; margin-bottom: 0.75rem; }
.doc-subtitle { color: var(--z500); max-width: 48ch; line-height: 1.5; margin-bottom: 2.5rem; }

.section-head { display: flex; align-items: center; gap: 1rem; margin: 2.5rem 0 1.25rem 0; }
.section-num { font-family: 'Instrument Serif', serif; font-style: italic; font-size: 1.3rem; color: var(--z400); }
.section-title { font-size: 0.8rem; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 600; }

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: transparent !important; border: none !important;
    border-bottom: 1px solid var(--z300) !important; border-radius: 0 !important;
    color: var(--ink) !important; box-shadow: none !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] * { color: var(--ink) !important; }
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within { border-bottom: 1px solid var(--ink) !important; }
div[data-testid="stNumberInput"] button { background: transparent !important; color: var(--ink) !important; }

label p {
    font-size: 11px !important; letter-spacing: 0.08em; text-transform: uppercase;
    font-weight: 500; color: var(--z500) !important;
}

.stButton > button {
    background: transparent !important; color: var(--ink) !important;
    border: 1px solid var(--ink) !important; border-radius: 2px;
    padding: 0.6rem 1.5rem; font-size: 0.85rem; font-weight: 500; box-shadow: none !important;
}
.stButton > button:hover { background: var(--ink) !important; color: var(--paper) !important; }
.stButton > button:focus:not(:active) { border: 1px solid var(--ink) !important; color: var(--ink) !important; }

.results-card { background: rgba(244,244,245,0.6); border: 1px solid rgba(0,0,0,0.05); padding: 2.2rem 2rem; margin-top: 2.5rem; }
.results-label { text-align: center; font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase; font-weight: 600; color: var(--z500); }
.results-verdict { text-align: center; font-family: 'Instrument Serif', serif; font-size: 3rem; margin: 0.25rem 0 1.75rem 0; }
.likelihood-row { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid var(--z200); padding-bottom: 0.5rem; margin-bottom: 1.25rem; }
.likelihood-pct { font-family: 'Instrument Serif', serif; font-size: 1.8rem; }
.bar-track { position: relative; height: 2px; background: var(--z200); margin: 0.75rem 0; }
.bar-marker { position: absolute; top: -6px; width: 2px; height: 14px; background: var(--ink); }
.bar-scale { display: flex; justify-content: space-between; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 600; color: var(--z400); }
.disclaimer { font-size: 10px; color: var(--z400); font-style: italic; margin-top: 1.5rem; border-top: 1px solid var(--z200); padding-top: 1rem; }
.factors-title { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600; color: var(--z500); margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--z200); }
.factor-row { display: flex; justify-content: space-between; font-size: 0.85rem; padding: 0.4rem 0; }
.factor-positive { color: #1f6f54; }
.factor-negative { color: #a6482e; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="doc-title">Startup Outcome Probability Model</div>
<div class="doc-subtitle">An analytical tool that estimates the success of early-stage startups based on their funding 
history, timeline, and characteristics.</div>
""", unsafe_allow_html=True)

with st.form("input_form"):
    st.markdown('<div class="section-head"><span class="section-num">01.</span><span class="section-title">Company Characteristics</span></div>', unsafe_allow_html=True)
    category = st.selectbox("Primary Category", TOP_CATEGORIES + ["Other"])
    continent = st.selectbox("Headquarters Region", CONTINENTS)

    st.markdown('<div class="section-head"><span class="section-num">02.</span><span class="section-title">Funding Architecture</span></div>', unsafe_allow_html=True)
    funding_total_usd = st.number_input("Total Funding Raised (USD)", min_value=0, value=2_000_000, step=50_000)
    funding_rounds = st.number_input("Number of Rounds", min_value=1, value=1, step=1)

    st.markdown('<div class="section-head"><span class="section-num">03.</span><span class="section-title">Temporal Dynamics</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    age_first_funding_year = c1.number_input("Years to First Round", value=0.5, step=0.1)
    age_last_funding_year = c2.number_input("Years to Most Recent Round", value=1.5, step=0.1)

    submitted = st.form_submit_button("Calculate Probability")

if submitted:
    row = {c: 0 for c in FEATURE_ORDER}
    funding_velocity = funding_rounds / max(age_last_funding_year, 0.25)
    funding_gap = max(age_last_funding_year - age_first_funding_year, 0)
    row.update({
        "funding_rounds": funding_rounds,
        "funding_total_usd_log": np.log1p(funding_total_usd),
        "funding_missing": 0,
        "age_last_funding_year": age_last_funding_year,
        "funding_gap": funding_gap,
        "age_missing": 0,
        "funding_velocity": funding_velocity,
    })
    if continent in CONTINENT_COLS:
        row[CONTINENT_COLS[continent]] = 1
    if category in CATEGORY_COLS:
        row[CATEGORY_COLS[category]] = 1
    else:
        row["is_othercategory"] = 1

    X = pd.DataFrame([row])[FEATURE_ORDER]

    model = load_model()
    prob_success = model.predict_proba(X)[0][1]
    pct = round(prob_success * 100, 1)
    verdict = "Leans successful" if prob_success >= 0.5 else "Leans closed"

    # Per-prediction explainability: contribution of each feature to the log-odds
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    X_scaled = scaler.transform(X)
    contributions = X_scaled[0] * clf.coef_[0]
    contrib_series = pd.Series(contributions, index=FEATURE_ORDER)
    top_factors = contrib_series.reindex(contrib_series.abs().sort_values(ascending=False).index).head(4)

    factor_rows = ""
    for feat, val in top_factors.items():
        label = FRIENDLY_NAMES.get(feat, feat)
        direction = "factor-positive" if val > 0 else "factor-negative"
        arrow = "pushes toward successful" if val > 0 else "pushes toward closed"
        factor_rows += f'<div class="factor-row"><span>{label}</span><span class="{direction}">{arrow}</span></div>'

    st.markdown(f"""
    <div class="results-card">
        <div class="results-label">Estimated Model Verdict</div>
        <div class="results-verdict">{verdict}</div>
        <div class="likelihood-row">
            <span style="font-size:0.85rem; font-weight:500;">Success Likelihood</span>
            <span class="likelihood-pct">{pct}%</span>
        </div>
        <div class="bar-track"><div class="bar-marker" style="left: calc({pct}% - 1px);"></div></div>
        <div class="bar-scale"><span>Closed</span><span>Successful</span></div>
        <div class="factors-title">Top factors behind this estimate</div>
        {factor_rows}
        <div class="disclaimer">Baseline model, first pass. Not investment advice.</div>
    </div>
    """, unsafe_allow_html=True)