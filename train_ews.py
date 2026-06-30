"""
=============================================================================
AI-DRIVEN EARLY WARNING SYSTEM FOR PREDICTING STUDENT DROPOUT
IN NIGERIAN UNIVERSITIES — TRAINING PIPELINE
=============================================================================
Author  : Senior Data Scientist & ML Engineer (EDM Specialisation)
Methodology : Chapter 3
Dataset : Nigerian-contextualised Student Dropout Dataset (4,424 records)
Target  : Dropout_Status (1 = Dropout, 0 = Non-Dropout)
"""

import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ML Pipeline & Metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Class Imbalance
from imblearn.over_sampling import SMOTE

# Visualisations
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("=" * 75)
print("  EWS PIPELINE TRAINING: MODEL EXECUTION & COMPARISON")
print("=" * 75)

# ─────────────────────────────────────────────────────────────────────────────
# §1  DATA LOADING & INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[§1] Loading dataset...")
DATA_PATH = "data/Actual_nigerian_student_dropout_dataset.csv"
df = pd.read_csv(DATA_PATH)

print(f"      Records loaded  : {len(df):,}")
print(f"      Feature columns : {df.shape[1] - 1}")
print(f"      Missing values  : {df.isnull().sum().sum()}")

dropout_counts = df["Dropout_Status"].value_counts()
dropout_pct = dropout_counts[1] / len(df) * 100
print(f"\n      Class distribution:")
print(f"        Non-Dropout (0) = {dropout_counts[0]:,}  ({100 - dropout_pct:.1f}%)")
print(f"        Dropout     (1) = {dropout_counts[1]:,}  ({dropout_pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# §2  FEATURE PREPROCESSING & ENCODING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[§2] Feature preprocessing...")

# ── 2a. Drop variables with no meaningful Nigerian analogue (if present) ────
DROP_COLS = ["Unemployment rate", "Inflation rate", "GDP", "Nationality", "International"]
existing_drop = [c for c in DROP_COLS if c in df.columns]
if existing_drop:
    df.drop(columns=existing_drop, inplace=True)
    print(f"      Dropped {len(existing_drop)} macro columns.")
else:
    print("      No macro columns to drop (already absent).")

# ── 2b. Separate features and target ─────────────────────────────────────────
TARGET = "Dropout_Status"
y = df[TARGET].copy()
X = df.drop(columns=[TARGET]).copy()

# ── 2c. Categorical Encoding ─────────────────────────────────────────────────
# One-hot encode nominal categoricals (Faculty). 
# Gender and Study_Mode are already binary (0/1) in this dataset.
OHE_COLS = ["Faculty"]
X = pd.get_dummies(X, columns=OHE_COLS, drop_first=True, dtype=int)

FEATURE_NAMES_ENCODED = X.columns.tolist()
print(f"      Post-encoding feature count : {len(FEATURE_NAMES_ENCODED)}")

# ── 2d. Feature Scaling List ─────────────────────────────────────────────────
# Scale continuous numeric features and discrete counts as per methodology
NUMERIC_COLS = [
    "CGPA_5point_Scale",
    "GPA_Semester_1_5pt",
    "GPA_Semester_2_5pt",
    "GPA_Change",
    "UTME_PostUME_Score",
    "Secondary_School_Exit_Grade",
    "Age_at_Matriculation",
    "Units_Registered_Semester_1", "Units_Passed_Semester_1",
    "Assessments_Sat_Semester_1",  "Units_No_Assessment_Semester_1",
    "Pass_Rate_Semester_1",
    "Units_Registered_Semester_2", "Units_Passed_Semester_2",
    "Assessments_Sat_Semester_2",  "Units_No_Assessment_Semester_2",
    "Pass_Rate_Semester_2",
]

# ─────────────────────────────────────────────────────────────────────────────
# §3  STRATIFIED TRAIN / TEST SPLIT (80/20, seed=42)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[§3] Stratified 80/20 train-test split (random_state=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y  # Maintains target class balance in splits
)

print(f"      Training set  : {X_train.shape[0]:,} records | Dropout rate = {y_train.mean()*100:.1f}%")
print(f"      Test set      : {X_test.shape[0]:,} records | Dropout rate = {y_test.mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# §4  FEATURE SCALING (Fit on Train, Transform Both to Prevent Data Leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[§4] StandardScaler fitting and transforming...")
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

present_numeric = [c for c in NUMERIC_COLS if c in X_train.columns]
X_train_scaled[present_numeric] = scaler.fit_transform(X_train[present_numeric])
X_test_scaled[present_numeric] = scaler.transform(X_test[present_numeric])

print(f"      Scaled {len(present_numeric)} continuous/discrete numeric features.")

# ─────────────────────────────────────────────────────────────────────────────
# §5  SMOTE CLASS IMBALANCE CORRECTION (Training Set Only!)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[§5] SMOTE oversampling (training set only)...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

pre_counts = pd.Series(y_train).value_counts()
post_counts = pd.Series(y_train_bal).value_counts()
print(f"      Before SMOTE : Non-Dropout={pre_counts[0]:,}, Dropout={pre_counts[1]:,} (ratio ~ {pre_counts[0]/pre_counts[1]:.1f}:1)")
print(f"      After  SMOTE : Non-Dropout={post_counts[0]:,}, Dropout={post_counts[1]:,} (ratio = 1:1)")

X_train_arr = np.array(X_train_bal)
y_train_arr = np.array(y_train_bal)
X_test_arr = np.array(X_test_scaled)
y_test_arr = np.array(y_test)

# ─────────────────────────────────────────────────────────────────────────────
# §6  MODEL DEFINITIONS & STRATIFIED 10-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[§6] Executing Stratified 10-Fold CV on training set...")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        max_features="sqrt",
        min_samples_leaf=5,
        bootstrap=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pre_counts[0] / pre_counts[1],
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ),
    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    )
}

cv_results = {}
trained_models = {}

for name, model in models.items():
    print(f"  -> {name}")
    
    cv_f1 = cross_val_score(model, X_train_arr, y_train_arr, cv=skf, scoring="f1", n_jobs=-1)
    cv_recall = cross_val_score(model, X_train_arr, y_train_arr, cv=skf, scoring="recall", n_jobs=-1)
    
    print(f"     10-Fold CV F1-Score : {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"     10-Fold CV Recall   : {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
    
    # Fit model on the full balanced training set
    model.fit(X_train_arr, y_train_arr)
    trained_models[name] = model
    
    cv_results[name] = {
        "cv_f1_mean": cv_f1.mean(),
        "cv_f1_std": cv_f1.std(),
        "cv_recall_mean": cv_recall.mean(),
        "cv_recall_std": cv_recall.std()
    }
    print()

# -----------------------------------------------------------------------------
# Section 7  TEST SET EVALUATION
# -----------------------------------------------------------------------------
print("=" * 75)
print("  Section 7 HELD-OUT TEST SET EVALUATION")
print("=" * 75)

eval_results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test_arr)
    y_proba = model.predict_proba(X_test_arr)[:, 1]
    
    acc = accuracy_score(y_test_arr, y_pred)
    prec = precision_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    roc = roc_auc_score(y_test_arr, y_proba)
    cm = confusion_matrix(y_test_arr, y_pred)
    
    eval_results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": cm.tolist(),  # Serialisable
        "y_proba": y_proba
    }
    
    print(f"\nModel: {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (Dropout class)")
    print(f"  Recall    : {rec:.4f}  (Dropout class) [Primary Metric]")
    print(f"  F1-Score  : {f1:.4f}  (Dropout class)")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:,}, FP={cm[0,1]:,}, FN={cm[1,0]:,}, TP={cm[1,1]:,}")

# -----------------------------------------------------------------------------
# Section 8  MODEL SELECTION (Recall Priority, F1 Tiebreaker)
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print("  Section 8 MODEL SELECTION SUMMARY")
print("=" * 75)

# Primary criterion: Recall (reduce missed at-risk students)
# Secondary criterion: F1-score (balance precision)
best_model_name = max(
    eval_results,
    key=lambda m: (eval_results[m]["recall"], eval_results[m]["f1"])
)

print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for name, res in eval_results.items():
    marker = " * BEST" if name == best_model_name else ""
    print(f"  {name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1']:>10.4f} {res['roc_auc']:>10.4f}{marker}")

print(f"\n  [OK] Selected Best Model: '{best_model_name}' (Optimising Recall)")
best_model = trained_models[best_model_name]

# Extract feature importances
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
else:
    importances = None

if importances is not None:
    feat_imp = pd.Series(importances, index=FEATURE_NAMES_ENCODED)
    top15 = feat_imp.nlargest(15)
    print("\n  Top Feature Importances:")
    for feat, val in top15.items():
        bar = "#" * int(val / top15.max() * 30)
        print(f"    {feat:<35} {val:.4f}  {bar}")
    GLOBAL_TOP_FEATURES = feat_imp.nlargest(20).index.tolist()
    FEATURE_IMPORTANCE_DICT = feat_imp.to_dict()
else:
    # MLP neural network fallback
    GLOBAL_TOP_FEATURES = [
        "CGPA_5point_Scale", "GPA_Semester_2_5pt", "GPA_Semester_1_5pt",
        "Fee_Arrears_Status", "School_Fees_Payment_Status",
        "Units_Passed_Semester_2", "Units_Passed_Semester_1",
        "GPA_Change", "Pass_Rate_Semester_2"
    ]
    FEATURE_IMPORTANCE_DICT = {f: 0.1 for f in GLOBAL_TOP_FEATURES}
    print("\n  [MLP Model - using methodology priority features for drivers]")

# -----------------------------------------------------------------------------
# Section 9  SERIALISE ARTEFACTS (Local Workspace Models Directory)
# -----------------------------------------------------------------------------
print("\n[Section 9] Serialising model artefacts to models/ directory...")
joblib.dump(best_model, "models/ews_best_model.pkl")
joblib.dump(scaler, "models/ews_scaler.pkl")
joblib.dump(FEATURE_NAMES_ENCODED, "models/ews_feature_names.pkl")
joblib.dump(GLOBAL_TOP_FEATURES, "models/ews_top_features.pkl")
joblib.dump(FEATURE_IMPORTANCE_DICT, "models/ews_feature_importances.pkl")

# Serialize all performance metrics for the API dashboard
metrics_summary = {}
for name, res in eval_results.items():
    metrics_summary[name] = {
        "accuracy": float(res["accuracy"]),
        "precision": float(res["precision"]),
        "recall": float(res["recall"]),
        "f1": float(res["f1"]),
        "roc_auc": float(res["roc_auc"]),
        "confusion_matrix": res["confusion_matrix"]
    }
with open("models/metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("      Saved: ews_best_model.pkl, ews_scaler.pkl, ews_feature_names.pkl")
print("      Saved: ews_top_features.pkl, ews_feature_importances.pkl, metrics_summary.json")

# -----------------------------------------------------------------------------
# Section 12  EVALUATION VISUALISATION REPORT
# -----------------------------------------------------------------------------
print("\n[Section 12] Plotting evaluation dashboard to reports/ews_evaluation_report.png...")
fig = plt.figure(figsize=(18, 16))
fig.suptitle(
    "AI-Driven Early Warning System (EWS) for student Dropout Prediction\n"
    "Nigerian Universities MVP — Model Evaluation Report",
    fontsize=16, fontweight="bold", y=0.98
)
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

palette = ["#2C3E50", "#E74C3C", "#27AE60", "#3498DB", "#9B59B6"]
names = list(eval_results.keys())
short_names = ["LR", "DT", "RF", "XGB", "MLP"]

# Plot 1: Recall Comparison
ax1 = fig.add_subplot(gs[0, 0])
recalls = [eval_results[n]["recall"] for n in names]
bars = ax1.bar(short_names, recalls, color=palette, edgecolor="white", linewidth=1.2)
ax1.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
ax1.set_title("Recall (Dropout Class)", fontweight="bold")
ax1.set_ylabel("Recall")
ax1.set_ylim(0, 1.15)
ax1.axhline(y=max(recalls), color="red", linestyle="--", alpha=0.5, linewidth=1)

# Plot 2: F1-Score Comparison
ax2 = fig.add_subplot(gs[0, 1])
f1s = [eval_results[n]["f1"] for n in names]
bars = ax2.bar(short_names, f1s, color=palette, edgecolor="white", linewidth=1.2)
ax2.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
ax2.set_title("F1-Score (Dropout Class)", fontweight="bold")
ax2.set_ylabel("F1-Score")
ax2.set_ylim(0, 1.15)

# Plot 3: ROC-AUC Comparison
ax3 = fig.add_subplot(gs[0, 2])
rocs = [eval_results[n]["roc_auc"] for n in names]
bars = ax3.bar(short_names, rocs, color=palette, edgecolor="white", linewidth=1.2)
ax3.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
ax3.set_title("ROC-AUC Score", fontweight="bold")
ax3.set_ylabel("AUC")
ax3.set_ylim(0.5, 1.15)
ax3.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5)

# Plots 4-8: Confusion Matrices
positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
for pos, name, short in zip(positions, names, short_names):
    ax = fig.add_subplot(gs[pos[0], pos[1]])
    cm = np.array(eval_results[name]["confusion_matrix"])
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Pred: Non-Drop", "Pred: Dropout"],
        yticklabels=["Act: Non-Drop", "Act: Dropout"],
        ax=ax, cbar=False, linewidths=0.5, linecolor="grey"
    )
    ax.set_title(f"{short} — Confusion Matrix", fontweight="bold", fontsize=10)

# Plot 9: Multi-Metric Comparison
ax9 = fig.add_subplot(gs[2, 2])
metrics_data = {
    "Accuracy": [eval_results[n]["accuracy"] for n in names],
    "Precision": [eval_results[n]["precision"] for n in names],
    "Recall": [eval_results[n]["recall"] for n in names],
    "F1-Score": [eval_results[n]["f1"] for n in names]
}
x = np.arange(len(names))
width = 0.18
for idx, (metric, vals) in enumerate(metrics_data.items()):
    offset = (idx - 1.5) * width
    ax9.bar(x + offset, vals, width, label=metric)
ax9.set_xticks(x)
ax9.set_xticklabels(short_names)
ax9.set_title("Metrics Summary", fontweight="bold")
ax9.set_ylabel("Score")
ax9.set_ylim(0, 1.2)
ax9.legend(fontsize=8, loc="lower right")

plt.savefig("reports/ews_evaluation_report.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("      Saved: reports/ews_evaluation_report.png")
print("\n[OK] PIPELINE COMPLETE. Best model trained: " + best_model_name)
print("=" * 75)
