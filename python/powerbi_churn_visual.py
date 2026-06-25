"""
Power BI Python Visual Script — Churn Prediction Model
======================================================

Instructions:
1. In Power BI, add a Python visual to your report.
2. Paste this entire script into the Python script editor.
3. Drag the following fields into the visual's Values well:
   - tenure
   - MonthlyCharges
   - TotalCharges
   - gender
   - SeniorCitizen
   - Partner
   - Dependents
   - PhoneService
   - MultipleLines
   - InternetService
   - OnlineSecurity
   - OnlineBackup
   - DeviceProtection
   - TechSupport
   - StreamingTV
   - StreamingMovies
   - Contract
   - PaperlessBilling
   - PaymentMethod
   - Churn

Note: This script reprocesses data inside Power BI for modeling.
It will render a 2x2 grid: ROC curves, confusion matrices, and a
performance comparison bar chart.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

dataset.columns = dataset.columns.str.strip()

data = dataset.copy()

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.dropna(subset=["TotalCharges"], inplace=True)

data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

cols_with_no_internet = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
for col in cols_with_no_internet:
    if col in data.columns:
        data[col] = data[col].replace("No internet service", "No")

if "MultipleLines" in data.columns:
    data["MultipleLines"] = data["MultipleLines"].map(
        {"No phone service": "No", "No": "No", "Yes": "Yes"}
    )

binary_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling", "SeniorCitizen"
]
for col in binary_cols:
    if col in data.columns:
        data[col] = data[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

cat_cols = [c for c in ["InternetService", "Contract", "PaymentMethod"]
            if c in data.columns]
data = pd.get_dummies(data, columns=cat_cols, drop_first=True, dtype=int)

feature_cols = [c for c in data.columns if c not in ["customerID", "Churn"]]
X = data[feature_cols]
y = data["Churn"]

num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"]
            if c in X.columns]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(
    class_weight="balanced", random_state=42,
    n_estimators=200, max_depth=10, min_samples_leaf=5
)
rf.fit(X_train, y_train)


def get_results(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc, "cm": cm, "fpr": fpr, "tpr": tpr,
        "y_pred": y_pred, "y_proba": y_proba
    }


lr_res = get_results(lr, X_test, y_test)
rf_res = get_results(rf, X_test, y_test)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Customer Churn — Predictive Model Results",
             fontsize=16, fontweight="bold", y=1.01)

for ax, res, name, color in [
    (axes[0, 0], lr_res, "Logistic Regression", "#3498db"),
    (axes[0, 1], rf_res, "Random Forest", "#2ecc71")
]:
    ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
            label=f"ROC (AUC = {res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} — ROC Curve", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")

for ax, res, name in [
    (axes[1, 0], lr_res, "Logistic Regression"),
    (axes[1, 1], rf_res, "Random Forest")
]:
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Active", "Churned"],
                yticklabels=["Active", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} — Confusion Matrix", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()
