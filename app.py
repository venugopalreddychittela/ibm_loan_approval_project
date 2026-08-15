import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
import hashlib
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Automated Credit Decision Engine",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "model/loan_model.pkl"
MODEL_INFO_PATH = "model/model_info.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Please run train_model.py first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
model_info = joblib.load(MODEL_INFO_PATH)


# ============================================================
# DATABASE
# ============================================================

DB_PATH = "loan_app.db"


def get_connection():

    connection = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

    return connection


def create_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            application_date TEXT,

            age INTEGER,
            credit_score INTEGER,

            checking_account TEXT,
            duration INTEGER,
            credit_history TEXT,
            purpose TEXT,
            credit_amount REAL,
            savings TEXT,
            employment TEXT,
            installment_rate INTEGER,
            personal_status TEXT,
            other_debtors TEXT,
            residence_since INTEGER,
            property TEXT,
            other_installment_plans TEXT,
            housing TEXT,
            existing_credits INTEGER,
            job TEXT,
            dependents INTEGER,
            telephone TEXT,
            foreign_worker TEXT,

            prediction TEXT,
            probability REAL,
            decision TEXT

        )
    """)

    connection.commit()

    connection.close()


create_database()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .approved {
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #2e8b57;
        background-color: #eaf8ef;
        text-align: center;
    }

    .rejected {
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #c0392b;
        background-color: #fdecea;
        text-align: center;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💳 Credit Decision Engine")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "📝 New Application",
        "📊 Application History",
        "📈 Model Performance",
        "ℹ️ About"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        'Automated Financial Credit & Loan Decision Engine'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine-learning based credit risk classification '
        'using the German Credit Risk Dataset'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Dataset Records",
            "1,000"
        )

    with col2:

        st.metric(
            "Features",
            "20"
        )

    with col3:

        st.metric(
            "Best Model",
            model_info["best_model"]
        )

    st.divider()

    st.subheader(
        "How the system works"
    )

    st.markdown(
        """
        ### 1️⃣ Applicant enters financial information

        The system collects information about the applicant's
        credit history, loan amount, employment, savings,
        housing, age and other financial attributes.

        ### 2️⃣ Backend validation

        The application validates all submitted information.

        Applicants must be between **18 and 65 years old**.

        ### 3️⃣ Machine Learning prediction

        The trained classification model evaluates the
        applicant's credit-risk profile.

        ### 4️⃣ Automated decision

        The system produces:

        - Good Credit Risk → **APPROVED**
        - Bad Credit Risk → **REJECTED**

        ### 5️⃣ Application record

        The decision is stored in the local SQLite database.
        """
    )

    st.info(
        "This is an educational ML-based credit-risk "
        "decision prototype, not a real banking system."
    )


# ============================================================
# NEW APPLICATION
# ============================================================

elif page == "📝 New Application":

    st.title("📝 New Credit Application")

    st.write(
        "Enter the applicant's financial and credit information."
    )

    st.divider()

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=65,
            value=25,
            step=1
        )

    with col2:

        personal_status = st.selectbox(
            "Personal Status",
            [
                "A91",
                "A92",
                "A93",
                "A94",
                "A95"
            ],
            format_func=lambda x: {
                "A91": "Male - Divorced/Separated",
                "A92": "Female - Divorced/Separated/Married",
                "A93": "Male - Single",
                "A94": "Male - Married/Widowed",
                "A95": "Female - Single"
            }[x]
        )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )


    # --------------------------------------------------------
    # ACCOUNT INFORMATION
    # --------------------------------------------------------

    st.subheader("🏦 Account Information")

    col1, col2 = st.columns(2)

    with col1:

        checking_account = st.selectbox(
            "Checking Account Status",
            [
                "A11",
                "A12",
                "A13",
                "A14"
            ],
            format_func=lambda x: {
                "A11": "Less than 0 DM",
                "A12": "0 to less than 200 DM",
                "A13": "200 DM or more",
                "A14": "No checking account"
            }[x]
        )

    with col2:

        savings = st.selectbox(
            "Savings Account",
            [
                "A61",
                "A62",
                "A63",
                "A64",
                "A65"
            ],
            format_func=lambda x: {
                "A61": "Less than 100 DM",
                "A62": "100 to less than 500 DM",
                "A63": "500 to less than 1000 DM",
                "A64": "1000 DM or more",
                "A65": "Unknown / No savings"
            }[x]
        )


    # --------------------------------------------------------
    # CREDIT HISTORY
    # --------------------------------------------------------

    st.subheader("📜 Credit History")

    credit_history = st.selectbox(
        "Credit History",
        [
            "A30",
            "A31",
            "A32",
            "A33",
            "A34"
        ],
        format_func=lambda x: {
            "A30": "No credits / all paid duly",
            "A31": "All credits at this bank paid duly",
            "A32": "Existing credits paid duly",
            "A33": "Delay in paying in the past",
            "A34": "Critical account / other credits"
        }[x]
    )


    # --------------------------------------------------------
    # LOAN INFORMATION
    # --------------------------------------------------------

    st.subheader("💰 Loan Information")

    col1, col2 = st.columns(2)

    with col1:

        duration = st.number_input(
            "Loan Duration (months)",
            min_value=4,
            max_value=72,
            value=24,
            step=1
        )

    with col2:

        credit_amount = st.number_input(
            "Credit Amount",
            min_value=250,
            max_value=20000,
            value=5000,
            step=250
        )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "A40",
            "A41",
            "A42",
            "A43",
            "A44",
            "A45",
            "A46",
            "A47",
            "A48",
            "A49",
            "A410"
        ],
        format_func=lambda x: {
            "A40": "New Car",
            "A41": "Used Car",
            "A42": "Furniture / Equipment",
            "A43": "Radio / Television",
            "A44": "Domestic Appliances",
            "A45": "Repairs",
            "A46": "Education",
            "A47": "Vacation",
            "A48": "Retraining",
            "A49": "Business",
            "A410": "Other"
        }[x]
    )


    # --------------------------------------------------------
    # EMPLOYMENT
    # --------------------------------------------------------

    st.subheader("💼 Employment")

    employment = st.selectbox(
        "Present Employment",
        [
            "A71",
            "A72",
            "A73",
            "A74",
            "A75"
        ],
        format_func=lambda x: {
            "A71": "Unemployed",
            "A72": "Less than 1 year",
            "A73": "1 to less than 4 years",
            "A74": "4 to less than 7 years",
            "A75": "7 years or more"
        }[x]
    )



    st.subheader("💳 Credit Score")

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700,
        step=1,
        help="Project-defined credit score range: 300–900."
    )

    if credit_score >= 750:
        credit_rating = "Excellent"
    elif credit_score >= 700:
        credit_rating = "Good"
    elif credit_score >= 650:
        credit_rating = "Fair"
    elif credit_score >= 600:
        credit_rating = "Weak"
    else:
        credit_rating = "Very Weak"

    st.info(
        f"Credit Rating: **{credit_rating}**"
    )

    # --------------------------------------------------------
    # FINANCIAL PROFILE
    # --------------------------------------------------------

    st.subheader("📊 Financial Profile")

    col1, col2 = st.columns(2)

    with col1:

        installment_rate = st.selectbox(
            "Installment Rate (% of disposable income)",
            [1, 2, 3, 4]
        )

    with col2:

        existing_credits = st.number_input(
            "Existing Credits",
            min_value=1,
            max_value=4,
            value=1,
            step=1
        )


    # --------------------------------------------------------
    # HOUSING
    # --------------------------------------------------------

    st.subheader("🏠 Housing & Property")

    col1, col2 = st.columns(2)

    with col1:

        housing = st.selectbox(
            "Housing",
            [
                "A151",
                "A152",
                "A153"
            ],
            format_func=lambda x: {
                "A151": "Rent",
                "A152": "Own",
                "A153": "For Free"
            }[x]
        )

    with col2:

        property_value = st.selectbox(
            "Property",
            [
                "A121",
                "A122",
                "A123",
                "A124"
            ],
            format_func=lambda x: {
                "A121": "Real Estate",
                "A122": "Building Society / Life Insurance",
                "A123": "Car / Other Property",
                "A124": "Unknown / No Property"
            }[x]
        )

    residence_since = st.number_input(
        "Years at Current Residence",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )


    # --------------------------------------------------------
    # OTHER CREDIT INFORMATION
    # --------------------------------------------------------

    st.subheader("📋 Other Credit Information")

    col1, col2 = st.columns(2)

    with col1:

        other_debtors = st.selectbox(
            "Other Debtors / Guarantors",
            [
                "A101",
                "A102",
                "A103"
            ],
            format_func=lambda x: {
                "A101": "None",
                "A102": "Co-applicant",
                "A103": "Guarantor"
            }[x]
        )

    with col2:

        other_installment_plans = st.selectbox(
            "Other Installment Plans",
            [
                "A141",
                "A142",
                "A143"
            ],
            format_func=lambda x: {
                "A141": "Bank",
                "A142": "Stores",
                "A143": "None"
            }[x]
        )


    # --------------------------------------------------------
    # JOB
    # --------------------------------------------------------

    job = st.selectbox(
        "Job",
        [
            "A171",
            "A172",
            "A173",
            "A174"
        ],
        format_func=lambda x: {
            "A171": "Unemployed / Unskilled Non-resident",
            "A172": "Unskilled Resident",
            "A173": "Skilled Employee / Official",
            "A174": "Management / Self-employed / Highly Qualified"
        }[x]
    )


    # --------------------------------------------------------
    # OTHER DETAILS
    # --------------------------------------------------------

    st.subheader("📞 Other Details")

    col1, col2 = st.columns(2)

    with col1:

        telephone = st.selectbox(
            "Telephone",
            [
                "A191",
                "A192"
            ],
            format_func=lambda x: {
                "A191": "No",
                "A192": "Yes"
            }[x]
        )

    with col2:

        foreign_worker = st.selectbox(
            "Foreign Worker",
            [
                "A201",
                "A202"
            ],
            format_func=lambda x: {
                "A201": "Yes",
                "A202": "No"
            }[x]
        )


    # ========================================================
    # SUBMIT
    # ========================================================

    st.divider()

    if st.button(
        "🔍 Evaluate Credit Application",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # SERVER-SIDE AGE VALIDATION
        # ----------------------------------------------------

        if age < 18 or age > 65:

            st.error(
                "Application rejected: applicant age must "
                "be between 18 and 65 years."
            )

            st.stop()


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            {
                "checking_account": [checking_account],
                "duration": [duration],
                "credit_history": [credit_history],
                "purpose": [purpose],
                "credit_amount": [credit_amount],
                "savings": [savings],
                "employment": [employment],
                "installment_rate": [installment_rate],
                "personal_status": [personal_status],
                "other_debtors": [other_debtors],
                "residence_since": [residence_since],
                "property": [property_value],
                "age": [age],
                "other_installment_plans": [
                    other_installment_plans
                ],
                "housing": [housing],
                "existing_credits": [existing_credits],
                "job": [job],
                "dependents": [dependents],
                "telephone": [telephone],
                "foreign_worker": [foreign_worker]
            }
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        probability_good = model.predict_proba(
            input_data
        )[0][1]

        probability_bad = 1 - probability_good

        # ----------------------------------------------------
        # DECISION ENGINE
        # ----------------------------------------------------

        decision = "REJECTED"
        risk = "HIGH"

        risk_factors = []
        positive_factors = []


        # ====================================================
        # AGE CHECK
        # ====================================================

        if age < 18 or age > 65:

            decision = "REJECTED"
            risk = "HIGH"

            risk_factors.append(
                "Applicant does not meet the age requirement."
            )


        # ====================================================
        # CREDIT SCORE
        # ====================================================

        elif credit_score < 600:

            decision = "REJECTED"
            risk = "HIGH"

            risk_factors.append(
                "Credit score is below the minimum project threshold."
            )

        elif credit_score < 650:

            risk_factors.append(
                "Credit score indicates elevated credit risk."
            )

        elif credit_score >= 750:

            positive_factors.append(
                "Excellent credit score."
            )

        elif credit_score >= 700:

            positive_factors.append(
                "Good credit score."
            )

        else:

            positive_factors.append(
                "Acceptable credit score."
            )


        # ====================================================
        # CREDIT HISTORY
        # ====================================================

        if credit_history in ["A30", "A31", "A32"]:

            positive_factors.append(
                "Credit history indicates satisfactory repayment behavior."
            )

        elif credit_history == "A33":

            risk_factors.append(
                "Credit history indicates previous payment delays."
            )

        elif credit_history == "A34":

            risk_factors.append(
                "Credit history indicates a critical or problematic account."
            )


        # ====================================================
        # EMPLOYMENT
        # ====================================================

        if employment == "A75":

            positive_factors.append(
                "Long-term employment history."
            )

        elif employment == "A74":

            positive_factors.append(
                "Stable employment history."
            )

        elif employment == "A71":

            risk_factors.append(
                "Applicant is currently unemployed."
            )

        elif employment == "A72":

            risk_factors.append(
                "Applicant has less than one year of employment history."
            )


        # ====================================================
        # SAVINGS
        # ====================================================

        if savings == "A64":

            positive_factors.append(
                "High level of savings."
            )

        elif savings == "A63":

            positive_factors.append(
                "Moderate level of savings."
            )

        elif savings == "A61":

            risk_factors.append(
                "Limited savings."
            )

        elif savings == "A65":

            risk_factors.append(
                "Savings information indicates no significant savings."
            )


        # ====================================================
        # INSTALLMENT RATE
        # ====================================================

        if installment_rate == 1:

            positive_factors.append(
                "Low installment burden."
            )

        elif installment_rate == 4:

            risk_factors.append(
                "High installment burden."
            )


        # ====================================================
        # EXISTING CREDITS
        # ====================================================

        if existing_credits >= 3:

            risk_factors.append(
                "Applicant has multiple existing credit obligations."
            )

        elif existing_credits == 1:

            positive_factors.append(
                "Applicant has limited existing credit obligations."
            )


        # ====================================================
        # LOAN AMOUNT
        # ====================================================

        if credit_amount >= 10000:

            risk_factors.append(
                "Requested credit amount is relatively high."
            )

        elif credit_amount <= 5000:

            positive_factors.append(
                "Requested credit amount is relatively moderate."
            )


        # ====================================================
        # FINAL ML + BUSINESS RULE DECISION
        # ====================================================

        # Age and very low credit score are hard rejection rules.

        if age < 18 or age > 65:

            decision = "REJECTED"
            risk = "HIGH"

        elif credit_score < 600:

            decision = "REJECTED"
            risk = "HIGH"

        else:

            # ML model + credit score

            if prediction == 1 and credit_score >= 650:

                decision = "APPROVED"

                if credit_score >= 700:

                    risk = "LOW"

                else:

                    risk = "MEDIUM"

            else:

                decision = "REJECTED"
                risk = "HIGH"


        # ====================================================
        # DECISION REASON
        # ====================================================

        if decision == "APPROVED":

            if risk == "LOW":

                decision_reason = (
                    "The applicant satisfies the project eligibility "
                    "rules and the ML model classified the applicant "
                    "as Good Credit."
                )

            else:

                decision_reason = (
                    "The applicant satisfies the eligibility rules "
                    "but has some moderate-risk indicators."
                )

        else:

            if risk_factors:

                decision_reason = (
                    "The application was rejected because one or more "
                    "risk indicators exceeded the project's decision criteria."
                )

            else:

                decision_reason = (
                    "The ML model classified the applicant as Bad Credit."
                )


        # ----------------------------------------------------
        # SAVE APPLICATION
        # ----------------------------------------------------

        application_date = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        prediction_text = (
            "Good Credit"
            if prediction == 1
            else "Bad Credit"
        )

        application_columns = [
            "application_date",
            "age",
            "credit_score",
            "checking_account",
            "duration",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings",
            "employment",
            "installment_rate",
            "personal_status",
            "other_debtors",
            "residence_since",
            "property",
            "other_installment_plans",
            "housing",
            "existing_credits",
            "job",
            "dependents",
            "telephone",
            "foreign_worker",
            "prediction",
            "probability",
            "decision"
        ]

        application_values = (
            application_date,
            age,
            credit_score,
            checking_account,
            duration,
            credit_history,
            purpose,
            credit_amount,
            savings,
            employment,
            installment_rate,
            personal_status,
            other_debtors,
            residence_since,
            property_value,
            other_installment_plans,
            housing,
            existing_credits,
            job,
            dependents,
            telephone,
            foreign_worker,
            prediction_text,
            float(probability_good),
            decision
        )

        placeholders = ", ".join(
            ["?"] * len(application_columns)
        )

        columns = ", ".join(application_columns)

        query = f"""
            INSERT INTO applications ({columns})
            VALUES ({placeholders})
        """

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            query,
            application_values
        )

        # Commit BEFORE closing the connection
        connection.commit()

        application_id = cursor.lastrowid

        # Close only after commit
        connection.close()


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()


        # ====================================================
        # DECISION HEADER
        # ====================================================

        if decision == "APPROVED":

            st.markdown(
                f"""
                <div class="approved">

                <h1>✅ APPLICATION APPROVED</h1>

                <h3>Credit Risk: {risk}</h3>

                <p>
                Application ID:
                <b>LN-{application_id:05d}</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="rejected">

                <h1>❌ APPLICATION REJECTED</h1>

                <h3>Credit Risk: {risk}</h3>

                <p>
                Application ID:
                <b>LN-{application_id:05d}</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # DECISION SUMMARY
        # ====================================================

        st.subheader("📋 Decision Summary")


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Decision",
                decision
            )


        with col2:

            st.metric(
                "Risk Level",
                risk
            )


        with col3:

            st.metric(
                "Credit Score",
                credit_score
            )


        with col4:

            st.metric(
                "Good Credit Probability",
                f"{probability_good * 100:.2f}%"
            )


        st.info(
            f"**Decision Reason:** {decision_reason}"
        )


        # ====================================================
        # POSITIVE FACTORS
        # ====================================================

        st.subheader("✅ Positive Factors")


        if positive_factors:

            for factor in positive_factors:

                st.success(
                    factor,
                    icon="✅"
                )

        else:

            st.write(
                "No significant positive indicators identified."
            )


        # ====================================================
        # RISK FACTORS
        # ====================================================

        st.subheader("⚠️ Risk Factors")


        if risk_factors:

            for factor in risk_factors:

                st.warning(
                    factor,
                    icon="⚠️"
                )

        else:

            st.write(
                "No significant rule-based risk indicators identified."
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with st.expander("🔍 Model Information"):

            st.write(
                f"**ML Classification:** "
                f"{'Good Credit' if prediction == 1 else 'Bad Credit'}"
            )

            st.write(
                f"**Good Credit Probability:** "
                f"{probability_good * 100:.2f}%"
            )

            st.write(
                f"**Bad Credit Probability:** "
                f"{probability_bad * 100:.2f}%"
            )

            st.caption(
                "The positive and risk factors shown above are "
                "rule-based indicators. They are not individual "
                "feature explanations generated by the ML model."
            )


# ============================================================
# APPLICATION HISTORY
# ============================================================

elif page == "📊 Application History":

    st.title("📊 Application History")

    connection = get_connection()

    data = pd.read_sql_query(
        """
        SELECT
            id AS "Application ID",
            application_date AS "Date",
            age AS "Age",
            credit_amount AS "Credit Amount",
            prediction AS "Credit Risk",
            probability AS "Good Credit Probability",
            decision AS "Decision"
        FROM applications
        ORDER BY id DESC
        """,
        connection
    )

    connection.close()

    if data.empty:

        st.info(
            "No applications have been submitted yet."
        )

    else:

        data["Good Credit Probability"] = (
            data["Good Credit Probability"] * 100
        ).round(2)

        data = data.rename(
            columns={
                "Good Credit Probability":
                "Good Credit Probability (%)"
            }
        )

        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "Comparison of classification models trained on "
        "the German Credit dataset."
    )

    st.divider()

    results = model_info["results"]

    performance_data = []

    for model_name, metrics in results.items():

        performance_data.append(
            {
                "Model": model_name,
                "Accuracy": round(
                    metrics["accuracy"] * 100,
                    2
                ),
                "Precision": round(
                    metrics["precision"] * 100,
                    2
                ),
                "Recall": round(
                    metrics["recall"] * 100,
                    2
                ),
                "F1 Score": round(
                    metrics["f1_score"] * 100,
                    2
                ),
                "ROC-AUC": round(
                    metrics["roc_auc"] * 100,
                    2
                )
            }
        )

    performance_df = pd.DataFrame(
        performance_data
    )

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Selected model: {model_info['best_model']}"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Select Model",
        list(results.keys())
    )

    cm = np.array(
        results[selected_model]["confusion_matrix"]
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Bad Credit",
            "Good Credit"
        ],
        yticklabels=[
            "Bad Credit",
            "Good Credit"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    st.pyplot(fig)


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.markdown(
        """
        ## Automated Financial Credit & Loan Decision Engine

        This project uses machine learning to classify
        applicants according to their credit risk.

        ### Dataset

        **Statlog German Credit Dataset**

        The dataset contains:

        - 1,000 credit applications
        - 20 predictive attributes
        - Good / Bad credit-risk classification

        ### Machine Learning Models

        The system compares:

        - Logistic Regression
        - Random Forest
        - Support Vector Machine

        ### Evaluation Metrics

        Models are evaluated using:

        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC
        - Confusion Matrix

        ### Age Restriction

        Applicants must be between:

        **18 and 65 years**

        ### Technology

        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - SQLite
        - Joblib

        ### Disclaimer

        This project is an educational machine-learning
        prototype. It should not be used as an actual
        financial lending system.
        """
    )