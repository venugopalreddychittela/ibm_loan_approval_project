import os
import joblib
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)


# ============================================================
# 2. LOAD GERMAN CREDIT DATASET
# ============================================================

print("Downloading German Credit Dataset...")

dataset = fetch_ucirepo(id=144)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()


# ============================================================
# 3. RENAME FEATURES
# ============================================================

column_names = [
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
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker"
]

X.columns = column_names


# ============================================================
# 4. CLEAN TARGET
# ============================================================

target_column = y.columns[0]

y = y[target_column]

# German Credit:
# 1 = Good
# 2 = Bad
#
# Convert to:
# 0 = Bad
# 1 = Good

y = y.astype(int)

y = y.map({
    1: 1,
    2: 0
})


# ============================================================
# 5. SAVE DATASET LOCALLY
# ============================================================

full_data = X.copy()
full_data["credit_risk"] = y

full_data.to_csv(
    "data/german_credit.csv",
    index=False
)

print("Dataset saved to data/german_credit.csv")


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "dependents"
]

categorical_features = [
    "checking_account",
    "credit_history",
    "purpose",
    "savings",
    "employment",
    "personal_status",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]


# ============================================================
# 7. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 9. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=8
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = {}

best_model_name = None
best_model = None
best_f1 = -1

print("\n================ MODEL RESULTS ================\n")

for model_name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

    print(model_name)
    print("-----------------------------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print()

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Bad Credit", "Good Credit"],
            zero_division=0
        )
    )

    # Select best model based on F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name
        best_model = pipeline


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "model/loan_model.pkl"
)


# ============================================================
# 12. SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "best_model": best_model_name,
    "best_f1_score": best_f1,
    "results": results,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "age_min": 18,
    "age_max": 65
}

joblib.dump(
    model_info,
    "model/model_info.pkl"
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("MODEL TRAINING COMPLETED")
print("==============================================")
print(f"Best Model : {best_model_name}")
print(f"F1 Score   : {best_f1:.4f}")
print()
print("Saved files:")
print("  model/loan_model.pkl")
print("  model/model_info.pkl")
print("  data/german_credit.csv")
print("==============================================")