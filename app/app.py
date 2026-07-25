import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_logreg.pkl"

FEATURE_ORDER = [
    "age_first_funding_year", "age_last_funding_year",
    "age_first_milestone_year", "age_last_milestone_year",
    "relationships", "funding_rounds", "milestones",
    "is_CA", "is_NY", "is_MA", "is_TX", "is_otherstate",
    "is_software", "is_web", "is_mobile", "is_enterprise", "is_advertising",
    "is_gamesvideo", "is_ecommerce", "is_biotech", "is_consulting", "is_othercategory",
    "has_VC", "has_angel", "has_roundA", "has_roundB", "has_roundC", "has_roundD",
    "avg_participants", "is_top500", "no_milestone_reached", "funding_total_usd_log",
]

CATEGORIES = {
    "Software": "is_software", "Web": "is_web", "Mobile": "is_mobile",
    "Enterprise": "is_enterprise", "Advertising": "is_advertising",
    "Games / Video": "is_gamesvideo", "Ecommerce": "is_ecommerce",
    "Biotech": "is_biotech", "Consulting": "is_consulting", "Other": "is_othercategory",
}
STATES = {
    "California": "is_CA", "New York": "is_NY", "Massachusetts": "is_MA",
    "Texas": "is_TX", "Other": "is_otherstate",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Startup Outcome Model", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500&display=swap');

:root {
    --ink: #14181c;
    --paper: #f2f4f3;
    --surface: #ffffff;
    --line: #d8dcda;
    --grow: #1f6f54;
    --risk: #a6482e;
    --muted: #6b7280;
}

.stApp { background-color: var(--paper); color: var(--ink); font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--ink); font-weight: 600; }
.stButton > button {
    background-color: var(--ink); color: var(--paper); border-radius: 3px;
    border: none; font-family: 'Inter', sans-serif; font-weight: 500;
    padding: 0.6rem 1.4rem;
}
.stButton > button:hover { background-color: var(--grow); color: white; }

.ledger-card {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 4px; padding: 1.75rem;
}
.ledger-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
}
.ledger-verdict {
    font-family: 'Fraunces', serif; font-size: 1.9rem; margin: 0.25rem 0 1rem 0;
}
.ledger-bar {
    position: relative; height: 10px; border-radius: 5px;
    background: linear-gradient(to right, var(--risk), var(--grow));
    margin: 0.5rem 0 0.75rem 0;
}
.ledger-marker {
    position: absolute; top: -6px; width: 3px; height: 22px;
    background: var(--ink);
}
.ledger-scale {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Startup Outcome Model")
st.markdown(
    "<span style='color: var(--muted)'>Estimates the likelihood of acquisition vs. closure "
    "from early-stage company characteristics. Baseline logistic regression, first pass.</span>",
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.3, 1])

with left:
    with st.form("input_form"):
        st.markdown("### Company")
        category = st.selectbox("Primary category", list(CATEGORIES.keys()))
        state = st.selectbox("Headquarters", list(STATES.keys()))
        is_top500 = st.checkbox("Ranked in startup world top 500")

        st.markdown("### Funding")
        funding_total_usd = st.number_input("Total funding raised (USD)", min_value=0, value=2_000_000, step=50_000)
        funding_rounds = st.number_input("Number of funding rounds", min_value=0, value=1, step=1)
        avg_participants = st.number_input("Average investors per round", min_value=0.0, value=2.0, step=0.5)
        cols = st.columns(3)
        has_VC = cols[0].checkbox("VC backed")
        has_angel = cols[1].checkbox("Angel backed")
        has_roundA = cols[2].checkbox("Raised Round A")
        cols2 = st.columns(3)
        has_roundB = cols2[0].checkbox("Raised Round B")
        has_roundC = cols2[1].checkbox("Raised Round C")
        has_roundD = cols2[2].checkbox("Raised Round D")

        st.markdown("### Timeline")
        age_first_funding_year = st.number_input("Years from founding to first funding", value=0.5, step=0.1)
        age_last_funding_year = st.number_input("Years from founding to most recent funding", value=1.5, step=0.1)
        relationships = st.number_input("Recorded professional relationships", min_value=0, value=5, step=1)

        st.markdown("### Milestones")
        reached_milestone = st.checkbox("This startup has reached at least one milestone")
        if reached_milestone:
            milestones = st.number_input("Number of milestones reached", min_value=1, value=1, step=1)
            age_first_milestone_year = st.number_input("Years to first milestone", value=1.0, step=0.1)
            age_last_milestone_year = st.number_input("Years to most recent milestone", value=1.5, step=0.1)
            no_milestone_reached = 0
        else:
            milestones = 0
            age_first_milestone_year = 0.0
            age_last_milestone_year = 0.0
            no_milestone_reached = 1

        submitted = st.form_submit_button("Estimate outcome")

with right:
    st.markdown("### Result")
    if not submitted:
        st.markdown(
            "<div class='ledger-card'><span class='ledger-label'>Awaiting input</span>"
            "<div class='ledger-verdict' style='color: var(--muted)'>Fill in the form and estimate</div></div>",
            unsafe_allow_html=True,
        )
    else:
        row = {c: 0 for c in FEATURE_ORDER}
        row.update({
            "age_first_funding_year": age_first_funding_year,
            "age_last_funding_year": age_last_funding_year,
            "age_first_milestone_year": age_first_milestone_year,
            "age_last_milestone_year": age_last_milestone_year,
            "relationships": relationships,
            "funding_rounds": funding_rounds,
            "milestones": milestones,
            CATEGORIES[category]: 1,
            STATES[state]: 1,
            "has_VC": int(has_VC), "has_angel": int(has_angel),
            "has_roundA": int(has_roundA), "has_roundB": int(has_roundB),
            "has_roundC": int(has_roundC), "has_roundD": int(has_roundD),
            "avg_participants": avg_participants,
            "is_top500": int(is_top500),
            "no_milestone_reached": no_milestone_reached,
            "funding_total_usd_log": np.log1p(funding_total_usd),
        })
        X = pd.DataFrame([row])[FEATURE_ORDER]

        model = load_model()
        prob_acquired = model.predict_proba(X)[0][1]
        pct = round(prob_acquired * 100, 1)
        verdict = "Leans acquired" if prob_acquired >= 0.5 else "Leans closed"
        verdict_color = "var(--grow)" if prob_acquired >= 0.5 else "var(--risk)"

        st.markdown(f"""
        <div class="ledger-card">
            <span class="ledger-label">Model estimate</span>
            <div class="ledger-verdict" style="color: {verdict_color}">{verdict}</div>
            <div class="ledger-bar">
                <div class="ledger-marker" style="left: calc({pct}% - 1px);"></div>
            </div>
            <div class="ledger-scale">
                <span>Closed</span>
                <span style="font-weight: 500; color: var(--ink)">{pct}% acquired</span>
                <span>Acquired</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<p style='color: var(--muted); font-size: 0.85rem; margin-top: 1rem;'>"
            "Baseline model, first pass. Not investment advice.</p>",
            unsafe_allow_html=True,
        )