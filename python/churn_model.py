import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)


import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    return df


def preprocess(df):
    data = df.copy()

    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    data["SeniorCitizen"] = data["SeniorCitizen"].map({1: "Yes", 0: "No"})

    cols_with_no_internet = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in cols_with_no_internet:
        data[col] = data[col].replace("No internet service", "No")

    multiple_lines_map = {"No phone service": "No", "No": "No", "Yes": "Yes"}
    data["MultipleLines"] = data["MultipleLines"].map(multiple_lines_map)

    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling", "SeniorCitizen"
    ]
    for col in binary_cols:
        data[col] = data[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    data = pd.get_dummies(
        data,
        columns=["InternetService", "Contract", "PaymentMethod"],
        drop_first=True,
        dtype=int
    )

    feature_cols = [c for c in data.columns if c not in [
        "customerID", "Churn"
    ]]
    X = data[feature_cols]
    y = data["Churn"]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, feature_cols, scaler


def train_models(X_train, y_train):
    lr = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE,
        n_estimators=200, max_depth=10, min_samples_leaf=5
    )
    rf.fit(X_train, y_train)

    return lr, rf


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": roc_auc, "cm": cm, "fpr": fpr, "tpr": tpr,
            "y_pred": y_pred, "y_proba": y_proba}


def plot_feature_importance(model, feature_names, model_name="Model", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if hasattr(model, "coef_"):
        importances = model.coef_[0]
        title = f"{model_name} — Coefficient Importance"
    else:
        importances = model.feature_importances_
        title = f"{model_name} — Feature Importance"

    indices = np.argsort(np.abs(importances))[-15:]
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in importances[indices]]

    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Coefficient Value" if hasattr(model, "coef_") else "Importance")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.5)
    return ax


def plot_confusion_matrix(cm, model_name="Model", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Active", "Churned"],
                yticklabels=["Active", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=11, fontweight="bold")
    return ax


def plot_roc_curve(fpr, tpr, roc_auc, model_name="Model", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color="#3498db", lw=2,
            label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    return ax


def generate_summary_plots(lr_results, rf_results, feature_names):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Telco Customer Churn — Predictive Model Results",
                 fontsize=16, fontweight="bold", y=1.02)

    plot_roc_curve(lr_results["fpr"], lr_results["tpr"],
                   lr_results["auc"], "Logistic Regression", ax=axes[0, 0])
    plot_roc_curve(rf_results["fpr"], rf_results["tpr"],
                   rf_results["auc"], "Random Forest", ax=axes[0, 1])

    axes[0, 2].axis("off")

    plot_confusion_matrix(lr_results["cm"], "Logistic Regression", ax=axes[1, 0])
    plot_confusion_matrix(rf_results["cm"], "Random Forest", ax=axes[1, 1])

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
        "Logistic Regression": [
            lr_results["accuracy"], lr_results["precision"],
            lr_results["recall"], lr_results["f1"], lr_results["auc"]
        ],
        "Random Forest": [
            rf_results["accuracy"], rf_results["precision"],
            rf_results["recall"], rf_results["f1"], rf_results["auc"]
        ]
    })
    metrics_df = metrics_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Model",
                palette=["#3498db", "#2ecc71"], ax=axes[1, 2])
    axes[1, 2].set_title("Model Performance Comparison", fontsize=11, fontweight="bold")
    axes[1, 2].set_ylim(0, 1.05)
    for container in axes[1, 2].containers:
        axes[1, 2].bar_label(container, fmt="%.3f", fontsize=7)

    plt.tight_layout()
    return fig


def generate_feature_importance_plots(lr, rf, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Top Churn Drivers — Feature Importance",
                 fontsize=14, fontweight="bold")

    plot_feature_importance(lr, feature_names, "Logistic Regression", ax=axes[0])
    plot_feature_importance(rf, feature_names, "Random Forest", ax=axes[1])

    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")

    print("Preprocessing...")
    X, y, feature_names, scaler = preprocess(df)
    print(f"Features: {len(feature_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    lr, rf = train_models(X_train, y_train)

    lr_results = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")

    print("\nGenerating plots...")
    fig1 = generate_summary_plots(lr_results, rf_results, feature_names)
    fig1.savefig("model_performance.png", dpi=150, bbox_inches="tight")
    print("Saved: model_performance.png")

    fig2 = generate_feature_importance_plots(lr, rf, feature_names)
    fig2.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    print("Saved: feature_importance.png")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
