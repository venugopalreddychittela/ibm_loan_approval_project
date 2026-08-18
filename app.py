import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Automated Credit Decision Engine", page_icon="💳", layout="wide")

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH, MODEL_INFO_PATH = "model/loan_model.pkl", "model/model_info.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
model_info = joblib.load(MODEL_INFO_PATH)

# ============================================================
# DATABASE
# ============================================================
DB_PATH = "loan_app.db"

APP_COLUMNS = [
    "application_date", "age", "credit_score", "checking_account", "duration",
    "credit_history", "purpose", "credit_amount", "savings", "employment",
    "installment_rate", "personal_status", "other_debtors", "residence_since",
    "property", "other_installment_plans", "housing", "existing_credits", "job",
    "dependents", "telephone", "foreign_worker", "prediction", "probability", "decision",
]


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_database():
    with get_connection() as conn:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {", ".join(c + (" INTEGER" if c in
                    {"age", "credit_score", "duration", "installment_rate",
                     "residence_since", "existing_credits", "dependents"}
                    else " REAL" if c in {"credit_amount", "probability"} else " TEXT")
                    for c in APP_COLUMNS)}
            )
        """)


create_database()

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main-title { font-size: 42px; font-weight: 700; margin-bottom: 5px; }
.subtitle { font-size: 18px; color: #777; margin-bottom: 30px; }
.approved { padding: 25px; border-radius: 12px; border: 2px solid #2e8b57; background-color: #eaf8ef; text-align: center; }
.rejected { padding: 25px; border-radius: 12px; border: 2px solid #c0392b; background-color: #fdecea; text-align: center; }
.metric-card { padding: 20px; border-radius: 12px; border: 1px solid #ddd; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# OPTION DICTIONARIES (code -> label)
# ============================================================
PERSONAL_STATUS = {"A91": "Male - Divorced/Separated", "A92": "Female - Divorced/Separated/Married",
                    "A93": "Male - Single", "A94": "Male - Married/Widowed", "A95": "Female - Single"}
CHECKING_ACCOUNT = {"A11": "Less than 0 DM", "A12": "0 to less than 200 DM",
                     "A13": "200 DM or more", "A14": "No checking account"}
SAVINGS = {"A61": "Less than 100 DM", "A62": "100 to less than 500 DM", "A63": "500 to less than 1000 DM",
           "A64": "1000 DM or more", "A65": "Unknown / No savings"}
CREDIT_HISTORY = {"A30": "No credits / all paid duly", "A31": "All credits at this bank paid duly",
                   "A32": "Existing credits paid duly", "A33": "Delay in paying in the past",
                   "A34": "Critical account / other credits"}
PURPOSE = {"A40": "New Car", "A41": "Used Car", "A42": "Furniture / Equipment", "A43": "Radio / Television",
           "A44": "Domestic Appliances", "A45": "Repairs", "A46": "Education", "A47": "Vacation",
           "A48": "Retraining", "A49": "Business", "A410": "Other"}
EMPLOYMENT = {"A71": "Unemployed", "A72": "Less than 1 year", "A73": "1 to less than 4 years",
              "A74": "4 to less than 7 years", "A75": "7 years or more"}
HOUSING = {"A151": "Rent", "A152": "Own", "A153": "For Free"}
PROPERTY = {"A121": "Real Estate", "A122": "Building Society / Life Insurance",
            "A123": "Car / Other Property", "A124": "Unknown / No Property"}
OTHER_DEBTORS = {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"}
OTHER_INSTALLMENT = {"A141": "Bank", "A142": "Stores", "A143": "None"}
JOB = {"A171": "Unemployed / Unskilled Non-resident", "A172": "Unskilled Resident",
       "A173": "Skilled Employee / Official", "A174": "Management / Self-employed / Highly Qualified"}
TELEPHONE = {"A191": "No", "A192": "Yes"}
FOREIGN_WORKER = {"A201": "Yes", "A202": "No"}


def select(label, options: dict, **kw):
    return st.selectbox(label, list(options.keys()), format_func=lambda x: options[x], **kw)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("💳 Credit Decision Engine")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Dashboard", "📝 New Application", "📊 Application History",
    "📈 Model Performance", "ℹ️ About",
])

# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-title">Automated Financial Credit & Loan Decision Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Machine-learning based credit risk classification using the German Credit Risk Dataset</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", "1,000")
    c2.metric("Features", "20")
    c3.metric("Best Model", model_info["best_model"])

    st.divider()
    st.subheader("How the system works")
    st.markdown("""
### 1️⃣ Applicant enters financial information
The system collects information about the applicant's credit history, loan amount,
employment, savings, housing, age and other financial attributes.

### 2️⃣ Backend validation
The application validates all submitted information. Applicants must be between **18 and 65 years old**.

### 3️⃣ Machine Learning prediction
The trained classification model evaluates the applicant's credit-risk profile.

### 4️⃣ Automated decision
- Good Credit Risk → **APPROVED**
- Bad Credit Risk → **REJECTED**

### 5️⃣ Application record
The decision is stored in the local SQLite database.
""")
    st.info("This is an educational ML-based credit-risk decision prototype, not a real banking system.")

# ============================================================
# NEW APPLICATION
# ============================================================
elif page == "📝 New Application":
    st.title("📝 New Credit Application")
    st.write("Enter the applicant's financial and credit information.")
    st.divider()

    st.subheader("👤 Personal Information")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=65, value=25, step=1)
    with c2:
        personal_status = select("Personal Status", PERSONAL_STATUS)
    dependents = st.number_input("Number of Dependents", min_value=1, max_value=10, value=1, step=1)

    st.subheader("🏦 Account Information")
    c1, c2 = st.columns(2)
    with c1:
        checking_account = select("Checking Account Status", CHECKING_ACCOUNT)
    with c2:
        savings = select("Savings Account", SAVINGS)

    st.subheader("📜 Credit History")
    credit_history = select("Credit History", CREDIT_HISTORY)

    st.subheader("💰 Loan Information")
    c1, c2 = st.columns(2)
    with c1:
        duration = st.number_input("Loan Duration (months)", min_value=4, max_value=72, value=24, step=1)
    with c2:
        credit_amount = st.number_input("Credit Amount", min_value=250, max_value=20000, value=5000, step=250)
    purpose = select("Loan Purpose", PURPOSE)

    st.subheader("💼 Employment")
    employment = select("Present Employment", EMPLOYMENT)

    st.subheader("💳 Credit Score")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700, step=1,
                                    help="Project-defined credit score range: 300–900.")
    rating = next(r for lo, r in [(750, "Excellent"), (700, "Good"), (650, "Fair"), (600, "Weak"), (0, "Very Weak")]
                  if credit_score >= lo)
    st.info(f"Credit Rating: **{rating}**")

    st.subheader("📊 Financial Profile")
    c1, c2 = st.columns(2)
    with c1:
        installment_rate = st.selectbox("Installment Rate (% of disposable income)", [1, 2, 3, 4])
    with c2:
        existing_credits = st.number_input("Existing Credits", min_value=1, max_value=4, value=1, step=1)

    st.subheader("🏠 Housing & Property")
    c1, c2 = st.columns(2)
    with c1:
        housing = select("Housing", HOUSING)
    with c2:
        property_value = select("Property", PROPERTY)
    residence_since = st.number_input("Years at Current Residence", min_value=1, max_value=4, value=2, step=1)

    st.subheader("📋 Other Credit Information")
    c1, c2 = st.columns(2)
    with c1:
        other_debtors = select("Other Debtors / Guarantors", OTHER_DEBTORS)
    with c2:
        other_installment_plans = select("Other Installment Plans", OTHER_INSTALLMENT)

    job = select("Job", JOB)

    st.subheader("📞 Other Details")
    c1, c2 = st.columns(2)
    with c1:
        telephone = select("Telephone", TELEPHONE)
    with c2:
        foreign_worker = select("Foreign Worker", FOREIGN_WORKER)

    st.divider()

    if st.button("🔍 Evaluate Credit Application", type="primary", use_container_width=True):
        if age < 18 or age > 65:
            st.error("Application rejected: applicant age must be between 18 and 65 years.")
            st.stop()

        input_data = pd.DataFrame([{
            "checking_account": checking_account, "duration": duration, "credit_history": credit_history,
            "purpose": purpose, "credit_amount": credit_amount, "savings": savings, "employment": employment,
            "installment_rate": installment_rate, "personal_status": personal_status,
            "other_debtors": other_debtors, "residence_since": residence_since, "property": property_value,
            "age": age, "other_installment_plans": other_installment_plans, "housing": housing,
            "existing_credits": existing_credits, "job": job, "dependents": dependents,
            "telephone": telephone, "foreign_worker": foreign_worker,
        }])

        prediction = model.predict(input_data)[0]
        probability_good = model.predict_proba(input_data)[0][1]
        probability_bad = 1 - probability_good

        # -------------------- rule-based factors --------------------
        positive_factors, risk_factors = [], []

        RULES = [
            (credit_score < 600, risk_factors, "Credit score is below the minimum project threshold."),
            (600 <= credit_score < 650, risk_factors, "Credit score indicates elevated credit risk."),
            (credit_score >= 750, positive_factors, "Excellent credit score."),
            (700 <= credit_score < 750, positive_factors, "Good credit score."),
            (650 <= credit_score < 700, positive_factors, "Acceptable credit score."),

            (credit_history in ("A30", "A31", "A32"), positive_factors, "Credit history indicates satisfactory repayment behavior."),
            (credit_history == "A33", risk_factors, "Credit history indicates previous payment delays."),
            (credit_history == "A34", risk_factors, "Credit history indicates a critical or problematic account."),

            (employment == "A75", positive_factors, "Long-term employment history."),
            (employment == "A74", positive_factors, "Stable employment history."),
            (employment == "A71", risk_factors, "Applicant is currently unemployed."),
            (employment == "A72", risk_factors, "Applicant has less than one year of employment history."),

            (savings == "A64", positive_factors, "High level of savings."),
            (savings == "A63", positive_factors, "Moderate level of savings."),
            (savings == "A61", risk_factors, "Limited savings."),
            (savings == "A65", risk_factors, "Savings information indicates no significant savings."),

            (installment_rate == 1, positive_factors, "Low installment burden."),
            (installment_rate == 4, risk_factors, "High installment burden."),

            (existing_credits >= 3, risk_factors, "Applicant has multiple existing credit obligations."),
            (existing_credits == 1, positive_factors, "Applicant has limited existing credit obligations."),

            (credit_amount >= 10000, risk_factors, "Requested credit amount is relatively high."),
            (credit_amount <= 5000, positive_factors, "Requested credit amount is relatively moderate."),
        ]
        for condition, bucket, message in RULES:
            if condition:
                bucket.append(message)

        # -------------------- final decision --------------------
        if age < 18 or age > 65 or credit_score < 600:
            decision, risk = "REJECTED", "HIGH"
        elif prediction == 1 and credit_score >= 650:
            decision = "APPROVED"
            risk = "LOW" if credit_score >= 700 else "MEDIUM"
        else:
            decision, risk = "REJECTED", "HIGH"

        if decision == "APPROVED":
            decision_reason = (
                "The applicant satisfies the project eligibility rules and the ML model "
                "classified the applicant as Good Credit." if risk == "LOW" else
                "The applicant satisfies the eligibility rules but has some moderate-risk indicators."
            )
        else:
            decision_reason = (
                "The application was rejected because one or more risk indicators exceeded "
                "the project's decision criteria." if risk_factors else
                "The ML model classified the applicant as Bad Credit."
            )

        # -------------------- save --------------------
        application_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction_text = "Good Credit" if prediction == 1 else "Bad Credit"
        values = (application_date, age, credit_score, checking_account, duration, credit_history, purpose,
                  credit_amount, savings, employment, installment_rate, personal_status, other_debtors,
                  residence_since, property_value, other_installment_plans, housing, existing_credits, job,
                  dependents, telephone, foreign_worker, prediction_text, float(probability_good), decision)

        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"INSERT INTO applications ({', '.join(APP_COLUMNS)}) VALUES ({', '.join(['?'] * len(APP_COLUMNS))})",
                values,
            )
            application_id = cursor.lastrowid

        # -------------------- display --------------------
        st.divider()
        css_class = "approved" if decision == "APPROVED" else "rejected"
        title = "✅ APPLICATION APPROVED" if decision == "APPROVED" else "❌ APPLICATION REJECTED"
        st.markdown(f"""
            <div class="{css_class}">
            <h1>{title}</h1>
            <h3>Credit Risk: {risk}</h3>
            <p>Application ID: <b>LN-{application_id:05d}</b></p>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("📋 Decision Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decision", decision)
        c2.metric("Risk Level", risk)
        c3.metric("Credit Score", credit_score)
        c4.metric("Good Credit Probability", f"{probability_good * 100:.2f}%")
        st.info(f"**Decision Reason:** {decision_reason}")

        st.subheader("✅ Positive Factors")
        if positive_factors:
            for factor in positive_factors:
                st.success(factor, icon="✅")
        else:
            st.write("No significant positive indicators identified.")

        st.subheader("⚠️ Risk Factors")
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor, icon="⚠️")
        else:
            st.write("No significant rule-based risk indicators identified.")

        with st.expander("🔍 Model Information"):
            st.write(f"**ML Classification:** {prediction_text}")
            st.write(f"**Good Credit Probability:** {probability_good * 100:.2f}%")
            st.write(f"**Bad Credit Probability:** {probability_bad * 100:.2f}%")
            st.caption("The positive and risk factors shown above are rule-based indicators. "
                       "They are not individual feature explanations generated by the ML model.")

# ============================================================
# APPLICATION HISTORY
# ============================================================
elif page == "📊 Application History":
    st.title("📊 Application History")

    with get_connection() as connection:
        data = pd.read_sql_query("""
            SELECT id AS "Application ID", application_date AS "Date", age AS "Age",
                   credit_amount AS "Credit Amount", prediction AS "Credit Risk",
                   probability AS "Good Credit Probability", decision AS "Decision"
            FROM applications ORDER BY id DESC
        """, connection)

    if data.empty:
        st.info("No applications have been submitted yet.")
    else:
        data["Good Credit Probability"] = (data["Good Credit Probability"] * 100).round(2)
        data = data.rename(columns={"Good Credit Probability": "Good Credit Probability (%)"})
        st.dataframe(data, use_container_width=True, hide_index=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.write("Comparison of classification models trained on the German Credit dataset.")
    st.divider()

    results = model_info["results"]
    performance_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": round(m["accuracy"] * 100, 2),
            "Precision": round(m["precision"] * 100, 2),
            "Recall": round(m["recall"] * 100, 2),
            "F1 Score": round(m["f1_score"] * 100, 2),
            "ROC-AUC": round(m["roc_auc"] * 100, 2),
        }
        for name, m in results.items()
    ])
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    st.success(f"Selected model: {model_info['best_model']}")

    st.divider()
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select Model", list(results.keys()))
    cm = np.array(results[selected_model]["confusion_matrix"])

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad Credit", "Good Credit"], yticklabels=["Bad Credit", "Good Credit"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {selected_model}")
    st.pyplot(fig)

# ============================================================
# ABOUT
# ============================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About the Project")
    st.markdown("""
## Automated Financial Credit & Loan Decision Engine

This project uses machine learning to classify applicants according to their credit risk.

### Dataset
**Statlog German Credit Dataset**
- 1,000 credit applications
- 20 predictive attributes
- Good / Bad credit-risk classification

### Machine Learning Models
- Logistic Regression
- Random Forest
- Support Vector Machine

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix

### Age Restriction
Applicants must be between **18 and 65 years**.

### Technology
Python · Streamlit · Pandas · NumPy · Scikit-learn · SQLite · Joblib

### Disclaimer
This project is an educational machine-learning prototype. It should not be used
as an actual financial lending system.
""")
