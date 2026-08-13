# Automated Credit Decision Engine

An AI/ML-powered web application that evaluates credit applications using machine learning and rule-based business logic.

The system uses the **Statlog German Credit Dataset** to classify applicants as **Good Credit** or **Bad Credit**, compares multiple machine-learning models, selects the best model based on F1 score, and combines the ML prediction with project-defined credit-risk rules to produce an automated approval or rejection decision.

> **Disclaimer:** This is an educational machine-learning prototype and is not intended for use as an actual banking, lending, or financial decision-making system.

---

## 📌 Project Overview

Traditional loan evaluation can involve manual assessment of financial and credit information.

This project demonstrates how machine learning can be used to automate part of the credit-risk assessment process.

The application allows a user to:

- Enter applicant and financial information
- Evaluate the applicant using a trained ML model
- Calculate Good Credit probability
- Apply additional business rules
- Generate an APPROVED or REJECTED decision
- Display risk factors and positive factors
- Store applications in a local SQLite database
- View previous applications
- Compare ML model performance
- Visualize confusion matrices

---

## 🚀 Key Features

### 1. Automated Credit Risk Prediction

The system predicts whether an applicant belongs to:

- **Good Credit**
- **Bad Credit**

using a trained machine-learning classification model.

### 2. Multiple ML Models

The training pipeline compares:

- Logistic Regression
- Random Forest
- Support Vector Machine

### 3. Automatic Model Selection

The best model is selected based on **F1 Score**.

For the current training run:

**Random Forest** achieved the highest F1 Score of **79.70%**.

### 4. Business Rule Decision Engine

The ML prediction is combined with additional project-defined rules.

Examples include:

- Applicant age must be between 18 and 65
- Credit score below 600 results in rejection
- Credit score of 650 or above is required for approval
- ML prediction must indicate Good Credit
- Additional financial indicators are used to identify positive and risk factors

### 5. Explainable Decision Summary

After evaluation, the application displays:

- Final decision
- Risk level
- Credit score
- Good Credit probability
- Decision reason
- Positive factors
- Risk factors
- ML classification

### 6. Application History

Every evaluated application is stored in a local SQLite database.

The application history page displays:

- Application ID
- Date
- Age
- Credit amount
- Credit risk classification
- Good Credit probability
- Final decision

### 7. Model Performance Dashboard

The application provides:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

for each trained model.

---

## 🧠 Machine Learning Workflow

```text
German Credit Dataset
        ↓
Data Loading
        ↓
Feature Renaming
        ↓
Target Transformation
        ↓
Train/Test Split
        ↓
Preprocessing
 ┌───────────────────────┐
 │ Numerical Features    │ → StandardScaler
 │ Categorical Features  │ → OneHotEncoder
 └───────────────────────┘
        ↓
Model Training
 ┌───────────────────────┐
 │ Logistic Regression   │
 │ Random Forest         │
 │ Support Vector Machine│
 └───────────────────────┘
        ↓
Model Evaluation
        ↓
Select Best Model by F1 Score
        ↓
Save Trained Model
        ↓
Streamlit Application
        ↓
ML Prediction
        ↓
Business Rules
        ↓
APPROVED / REJECTED