# =====================================================
# LOAN DEFAULT PREDICTION USING LOGISTIC REGRESSION
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(r"C:\Users\manthan barot\OneDrive\Desktop\WELTECH\MACHINE LEARNING\ML TEST - 13.6.26\ml_datasets\04_logistic_regression_loan_default.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop("Defaulted", axis=1)
y = df["Defaulted"]

# =====================================================
# COLUMN TYPES
# =====================================================

numerical_features = [
    "Age",
    "Annual_Income_INR",
    "Loan_Amount_INR",
    "Credit_Score",
    "Employment_Years",
    "Existing_Loans"
]

categorical_features = [
    "Loan_Type"
]

# =====================================================
# PREPROCESSING
# =====================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# =====================================================
# MODEL PIPELINE
# =====================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# =====================================================
# EVALUATION METRICS
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL RESULTS ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# =====================================================
# CONFUSION MATRIX
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Default", "Default"]
)

disp.plot(ax=ax)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# ROC CURVE
# =====================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# ACTUAL CLASS DISTRIBUTION
# =====================================================

class_counts = y.value_counts()

plt.figure(figsize=(6,5))

plt.bar(
    ["No Default", "Default"],
    class_counts.values
)

plt.title("Loan Default Distribution")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# SAMPLE PREDICTION
# =====================================================

sample = pd.DataFrame({
    "Age":[35],
    "Annual_Income_INR":[1200000],
    "Loan_Amount_INR":[800000],
    "Credit_Score":[720],
    "Employment_Years":[8],
    "Existing_Loans":[1],
    "Loan_Type":["Home"]
})

prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0][1]

print("\nSample Prediction")

print("Default Prediction :", prediction)

print(f"Default Probability : {probability:.2%}")