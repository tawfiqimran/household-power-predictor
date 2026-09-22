import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide", page_icon="💳")

# --- Load saved artifacts ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)
with open("threshold.pkl", "rb") as f:
    threshold = pickle.load(f)
with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)
with open("feature_importance.pkl", "rb") as f:
    feature_importance = pickle.load(f)

NUM_COLS = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
            "LoanToIncome", "IncomePerCreditLine"]

# --- Light custom styling ---
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Risk Dashboard")
st.caption("Loan default risk prediction, powered by a HistGradientBoosting model")

tab_predict, tab_model, tab_data = st.tabs(["🔮 Predict", "📊 Model Performance", "📁 Data Overview"])

# ============================================================
# TAB 1: PREDICTION
# ============================================================
with tab_predict:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Applicant Details")

        with st.expander("Financials", expanded=True):
            income = st.number_input("Annual Income ($)", 10000, 200000, 60000, step=1000)
            loan_amount = st.number_input("Loan Amount ($)", 1000, 250000, 50000, step=1000)
            interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 10.0)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

        with st.expander("Personal", expanded=True):
            age = st.slider("Age", 18, 70, 35)
            months_employed = st.slider("Months Employed", 0, 120, 24)
            num_credit_lines = st.slider("Number of Credit Lines", 0, 10, 2)
            education = st.selectbox("Education", ["Bachelor's", "High School", "PhD"])
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

        with st.expander("Other", expanded=True):
            c1, c2, c3 = st.columns(3)
            has_mortgage = c1.selectbox("Mortgage", ["Yes", "No"])
            has_dependents = c2.selectbox("Dependents", ["Yes", "No"])
            has_cosigner = c3.selectbox("Co-Signer", ["Yes", "No"])

        predict_clicked = st.button("Predict Default Risk", type="primary", use_container_width=True)

    with right:
        if predict_clicked:
            input_dict = {
                "Age": age, "Income": income, "LoanAmount": loan_amount,
                "CreditScore": credit_score, "MonthsEmployed": months_employed,
                "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
                "LoanTerm": loan_term, "DTIRatio": dti_ratio,
                "LoanToIncome": loan_amount / income,
                "IncomePerCreditLine": income / (num_credit_lines + 1),
                f"Education_{education}": 1, f"EmploymentType_{employment_type}": 1,
                f"MaritalStatus_{marital_status}": 1, f"LoanPurpose_{loan_purpose}": 1,
                f"HasMortgage_{has_mortgage}": 1, f"HasDependents_{has_dependents}": 1,
                f"HasCoSigner_{has_cosigner}": 1,
            }
            input_df = pd.DataFrame([input_dict])
            for col in columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[columns]
            input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

            proba = model.predict_proba(input_df)[0][1]
            is_high_risk = proba >= threshold

            # --- Gauge chart ---
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                title={"text": "Default Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d62728" if is_high_risk else "#2ca02c"},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "rgba(44,160,44,0.2)"},
                        {"range": [threshold * 100, 100], "color": "rgba(214,39,40,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "value": threshold * 100,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            if is_high_risk:
                st.error(f"⚠️ High risk of default (cutoff: {threshold:.0%})")
            else:
                st.success(f"✅ Low risk of default (cutoff: {threshold:.0%})")

            # --- Feature importance (precomputed via permutation importance
            # at training time — this is the model's overall behavior, not
            # specific to this one prediction) ---
            st.subheader("Top factors the model relies on")
            top_imp = feature_importance.head(8)
            fig_imp = go.Figure(go.Bar(
                x=top_imp.values, y=top_imp.index, orientation="h",
                marker_color="#1f77b4"
            ))
            fig_imp.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                                   xaxis_title="Importance (drop in ROC-AUC when shuffled)")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("This shows what the model leans on overall, not what specifically "
                       "drove this one prediction.")
        else:
            st.info("Fill in the applicant details and click **Predict Default Risk** to see results here.")

# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================
with tab_model:
    st.subheader("Model Performance on Held-Out Test Data")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    m2.metric("F1 Score", f"{metrics['f1']:.3f}")
    m3.metric("Precision", f"{metrics['precision']:.3f}")
    m4.metric("Recall", f"{metrics['recall']:.3f}")

    st.divider()
    st.markdown(f"""
    **Model:** HistGradientBoostingClassifier (balanced class weights)
    **Decision threshold:** {metrics['threshold']:.3f} (tuned to maximize F1, not the default 0.5)

    **Honest read of these numbers:** ROC-AUC of {metrics['roc_auc']:.2f} means the model
    is meaningfully better than random guessing, but this is inherently noisy data —
    predicting individual human default behavior from a handful of financial features
    has a real accuracy ceiling. Precision of {metrics['precision']:.0%} means that of the
    loans flagged as high risk, about {metrics['precision']:.0%} actually default. This model
    is useful for *flagging applicants for closer review*, not for fully automated
    accept/reject decisions.
    """)

# ============================================================
# TAB 3: DATA OVERVIEW
# ============================================================
with tab_data:
    st.subheader("Training Data Summary")
    d1, d2 = st.columns(2)
    d1.metric("Total records", f"{metrics['n_rows']:,}")
    d2.metric("Default rate", f"{metrics['default_rate']:.1%}")
    st.caption("The dataset is imbalanced (most loans don't default), which is why "
               "the decision threshold was tuned instead of using a plain 0.5 cutoff.")