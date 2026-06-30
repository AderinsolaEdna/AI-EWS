"""
=============================================================================
AI-DRIVEN EARLY WARNING SYSTEM FOR PREDICTING STUDENT DROPOUT
IN NIGERIAN UNIVERSITIES — INFERENCE LAYER & RULE ENGINE
=============================================================================
Author  : Senior Data Scientist & ML Engineer (EDM Specialisation)
Methodology : Chapter 3
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# Load persisted ML assets if they exist (graceful fallback for script verification)
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ews_best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "ews_scaler.pkl")
FEAT_NAMES_PATH = os.path.join(MODEL_DIR, "ews_feature_names.pkl")
TOP_FEAT_PATH = os.path.join(MODEL_DIR, "ews_top_features.pkl")
FEAT_IMPS_PATH = os.path.join(MODEL_DIR, "ews_feature_importances.pkl")

_model = None
_scaler = None
_feature_names = None
_top_features = None
_feature_importances = None

# Numeric columns list that need scaling (must match train_ews.py exactly)
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

def load_assets():
    """Load serialized assets from models directory."""
    global _model, _scaler, _feature_names, _top_features, _feature_importances
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        _feature_names = joblib.load(FEAT_NAMES_PATH)
        _top_features = joblib.load(TOP_FEAT_PATH)
        _feature_importances = joblib.load(FEAT_IMPS_PATH)
        return True
    return False

# Rule-based intervention dictionary (corresponds to Methodology Table 3.3)
INTERVENTION_RULES = [
    {
        "domain": "Financial (Arrears/Fees)",
        "features": {"Fee_Arrears_Status", "School_Fees_Payment_Status"},
        "tier": "High Risk",
        "intervention": "Immediate referral to Bursary Office for fee deferral or bursary application; notify Dean of Student Affairs"
    },
    {
        "domain": "Financial (Arrears/Fees)",
        "features": {"Fee_Arrears_Status", "School_Fees_Payment_Status"},
        "tier": "Medium Risk",
        "intervention": "Advisory notification on bursary and financial aid options; scheduled welfare officer meeting"
    },
    {
        "domain": "Academic (CGPA/GPA)",
        "features": {"CGPA_5point_Scale", "GPA_Semester_1_5pt", "GPA_Semester_2_5pt", "GPA_Change"},
        "tier": "High Risk",
        "intervention": "Referral to academic adviser for urgent improvement plan; consideration for supplementary examination access"
    },
    {
        "domain": "Academic (CGPA/GPA)",
        "features": {"CGPA_5point_Scale", "GPA_Semester_1_5pt", "GPA_Semester_2_5pt", "GPA_Change"},
        "tier": "Medium Risk",
        "intervention": "Recommendation for peer tutoring; adviser check-in within two weeks"
    },
    {
        "domain": "Academic (Course Load/Progress)",
        "features": {"Units_Passed_Semester_1", "Units_Passed_Semester_2"},
        "tier": "High Risk",
        "intervention": "Academic review with Head of Department; assessment of course load and repeat options"
    },
    {
        "domain": "Social (First Gen)",
        "features": {"First_Generation_Student"},
        "tier": "any",
        "intervention": "Referral to mentorship programme; connection with senior peer students"
    },
    {
        "domain": "Welfare (Housing)",
        "features": {"Hostel_Residency", "Non_Resident_Student"},
        "tier": "Medium Risk",
        "intervention": "Welfare check on off-campus living conditions; hostel allocation information"
    }
]

MULTI_FLAG_INTERVENTION = "Multi-agency intervention (adviser, welfare officer, bursary); case flagged for Dean of Students review"

def _resolve_intervention(top_drivers: list, risk_tier: str) -> str:
    """Map driver features and risk tier to a Table 3.3 intervention string."""
    driver_set = set(top_drivers)
    matched_interventions = []
    matched_domains = set()

    for rule in INTERVENTION_RULES:
        overlap = driver_set & rule["features"]
        if overlap and (rule["tier"] == risk_tier or rule["tier"] == "any"):
            matched_interventions.append(rule["intervention"])
            matched_domains.add(rule["domain"])

    # If drivers span 2 or more distinct domains (e.g. Financial AND Academic) at high risk, use multi-agency
    if len(matched_domains) >= 2:
        return MULTI_FLAG_INTERVENTION
    elif len(matched_interventions) == 1:
        return matched_interventions[0]
    elif len(matched_interventions) > 1:
        # Return the first one or combine them
        return matched_interventions[0]
    
    # Fallback default intervention
    return "General welfare monitoring recommended. Schedule academic adviser check-in."

def predict_dropout_risk(student_row_dict: dict) -> dict:
    """
    Predicts the student dropout probability, risk tier, top drivers, and recommended intervention.
    
    Accepts: A dictionary representing a single student record matching the dataset schema.
    Returns: A dictionary (JSON-serialisable) containing:
      - 'probability': float (0.0 to 1.0)
      - 'risk_tier': string ('High Risk' >= 0.667, 'Medium Risk' 0.334 - 0.666, 'Low Risk' < 0.334)
      - 'top_drivers': list of the 2 features driving the risk calculation
      - 'actionable_intervention': string mapping back to the rule-based logic in Table 3.3
    """
    global _model, _scaler, _feature_names, _top_features, _feature_importances
    
    # Ensure assets are loaded
    if _model is None:
        if not load_assets():
            return {
                "error": "Model assets not loaded. Please run the training pipeline first."
            }

    # ── Step A: Convert input to DataFrame ───────────────────────────────────
    raw_df = pd.DataFrame([student_row_dict])

    # Drop target and macro variables if present
    TARGET = "Dropout_Status"
    DROP_COLS = ["Unemployment rate", "Inflation rate", "GDP", "Nationality", "International"]
    if TARGET in raw_df.columns:
        raw_df.drop(columns=[TARGET], inplace=True)
    raw_df.drop(columns=[c for c in DROP_COLS if c in raw_df.columns], inplace=True, errors="ignore")

    # ── Step B: Categorical One-Hot Encoding Alignment ────────────────────────
    # Encode Faculty (nominal string variable)
    if "Faculty" in raw_df.columns:
        raw_df = pd.get_dummies(raw_df, columns=["Faculty"], drop_first=True, dtype=int)

    # Reconstruct/align all features from training layout
    for col in _feature_names:
        if col not in raw_df.columns:
            raw_df[col] = 0
            
    # Keep only the columns present in feature names and in that exact order
    raw_df = raw_df[_feature_names]

    # ── Step C: Feature Scaling ──────────────────────────────────────────────
    scaled_df = raw_df.copy()
    scale_cols_present = [c for c in NUMERIC_COLS if c in scaled_df.columns]
    scaled_df[scale_cols_present] = _scaler.transform(raw_df[scale_cols_present])

    # ── Step D: Inference Probability ────────────────────────────────────────
    X_infer = np.array(scaled_df)
    prob = float(_model.predict_proba(X_infer)[0, 1])
    if prob > 0.995:
        prob = 0.995

    # ── Step E: Determine Risk Tier ──────────────────────────────────────────
    if prob >= 0.667:
        risk_tier = "High Risk"
    elif prob >= 0.334:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "Low Risk"

    # ── Step F: Calculate Student-Specific Top Drivers ────────────────────────
    # Combine model global feature importance weights with the student's individual risk levels
    driver_scores = {}
    for feat in _top_features:
        if feat not in student_row_dict:
            continue
            
        val = student_row_dict[feat]
        signal = 0.0
        
        # Financial
        if feat == "Fee_Arrears_Status":
            signal = float(val)
        elif feat == "School_Fees_Payment_Status":
            signal = 1.0 - float(val)
        elif feat == "Bursary_Scholarship_Status":
            signal = 1.0 - float(val)
            
        # Academic scores (low scores mean high risk)
        elif feat in ["CGPA_5point_Scale", "GPA_Semester_1_5pt", "GPA_Semester_2_5pt"]:
            signal = max(0.0, (5.0 - float(val)) / 5.0)
        elif feat == "GPA_Change":
            signal = max(0.0, -float(val))  # Negative values are risky
        elif feat in ["Pass_Rate_Semester_1", "Pass_Rate_Semester_2"]:
            signal = 1.0 - float(val)
            
        # Semester unit counts
        elif feat in ["Units_Passed_Semester_1", "Units_Passed_Semester_2"]:
            reg_feat = feat.replace("Passed", "Registered")
            if reg_feat in student_row_dict and student_row_dict[reg_feat] > 0:
                signal = 1.0 - (float(val) / float(student_row_dict[reg_feat]))
            else:
                signal = max(0.0, (10.0 - float(val)) / 10.0)
        elif feat in ["Units_No_Assessment_Semester_1", "Units_No_Assessment_Semester_2"]:
            signal = min(1.0, float(val) / 5.0)
            
        # General background
        elif feat == "First_Generation_Student":
            signal = float(val)
        elif feat == "Non_Resident_Student":
            signal = float(val)
        elif feat == "Hostel_Residency":
            signal = 1.0 - float(val)
        elif feat == "Marital_Status_Binary":
            signal = float(val)
        else:
            signal = 0.5
            
        # Multiply student signal by model feature importance
        global_weight = _feature_importances.get(feat, 0.05)
        driver_scores[feat] = signal * global_weight

    # Sort drivers in descending order and select top 2
    top_2 = sorted(driver_scores, key=driver_scores.get, reverse=True)[:2]
    
    # Pad to ensure exactly 2 drivers are returned
    if len(top_2) < 2:
        for f in _top_features:
            if f not in top_2 and f in student_row_dict:
                top_2.append(f)
            if len(top_2) == 2:
                break

    # ── Step G: Table 3.3 Interventions ─────────────────────────────────────
    intervention = _resolve_intervention(top_2, risk_tier)

    # ── Step H: Build Return Object ──────────────────────────────────────────
    return {
        "probability": round(prob, 4),
        "risk_tier": risk_tier,
        "top_drivers": top_2,
        "actionable_intervention": intervention
    }

def predict_dropout_risk_batch(df_raw: pd.DataFrame) -> list:
    """
    Predicts dropout risk for a batch of student records.
    Vectorized preprocessing and ML model execution for sub-second performance.
    """
    global _model, _scaler, _feature_names, _top_features, _feature_importances

    # Ensure assets are loaded
    if _model is None:
        if not load_assets():
            raise RuntimeError("Model assets not loaded. Please run the training pipeline first.")

    # ── Step A: Preprocess batch DataFrame ───────────────────────────────────
    raw_df = df_raw.copy()

    # Drop target and macro variables if present
    TARGET = "Dropout_Status"
    DROP_COLS = ["Unemployment rate", "Inflation rate", "GDP", "Nationality", "International"]
    if TARGET in raw_df.columns:
        raw_df.drop(columns=[TARGET], inplace=True)
    raw_df.drop(columns=[c for c in DROP_COLS if c in raw_df.columns], inplace=True, errors="ignore")

    # ── Step B: Categorical One-Hot Encoding Alignment ────────────────────────
    # Encode Faculty (nominal string variable)
    if "Faculty" in raw_df.columns:
        raw_df = pd.get_dummies(raw_df, columns=["Faculty"], drop_first=True, dtype=int)

    # Reconstruct/align all features from training layout
    for col in _feature_names:
        if col not in raw_df.columns:
            raw_df[col] = 0
            
    # Keep only the columns present in feature names and in that exact order
    raw_df = raw_df[_feature_names]

    # ── Step C: Feature Scaling ──────────────────────────────────────────────
    scaled_df = raw_df.copy()
    scale_cols_present = [c for c in NUMERIC_COLS if c in scaled_df.columns]
    scaled_df[scale_cols_present] = _scaler.transform(raw_df[scale_cols_present])

    # ── Step D: Inference Probability (Batch) ────────────────────────────────
    X_infer = np.array(scaled_df)
    probs = _model.predict_proba(X_infer)[:, 1]
    # Cap highest risk probability at 0.995
    probs = np.minimum(probs, 0.995)

    # ── Step E: Resolve Drivers and Interventions per Student ─────────────────
    results = []
    # Convert df_raw to records to compute individual driver scores
    records = df_raw.to_dict(orient='records')
    
    for i, student_row_dict in enumerate(records):
        prob = float(probs[i])
        
        # Determine Risk Tier
        if prob >= 0.667:
            risk_tier = "High Risk"
        elif prob >= 0.334:
            risk_tier = "Medium Risk"
        else:
            risk_tier = "Low Risk"

        # Calculate student-specific top drivers
        driver_scores = {}
        for feat in _top_features:
            if feat not in student_row_dict:
                continue
                
            val = student_row_dict[feat]
            signal = 0.0
            
            # Financial
            if feat == "Fee_Arrears_Status":
                signal = float(val)
            elif feat == "School_Fees_Payment_Status":
                signal = 1.0 - float(val)
            elif feat == "Bursary_Scholarship_Status":
                signal = 1.0 - float(val)
                
            # Academic scores (low scores mean high risk)
            elif feat in ["CGPA_5point_Scale", "GPA_Semester_1_5pt", "GPA_Semester_2_5pt"]:
                signal = max(0.0, (5.0 - float(val)) / 5.0)
            elif feat == "GPA_Change":
                signal = max(0.0, -float(val))  # Negative values are risky
            elif feat in ["Pass_Rate_Semester_1", "Pass_Rate_Semester_2"]:
                signal = 1.0 - float(val)
                
            # Semester unit counts
            elif feat in ["Units_Passed_Semester_1", "Units_Passed_Semester_2"]:
                reg_feat = feat.replace("Passed", "Registered")
                if reg_feat in student_row_dict and student_row_dict[reg_feat] > 0:
                    signal = 1.0 - (float(val) / float(student_row_dict[reg_feat]))
                else:
                    signal = max(0.0, (10.0 - float(val)) / 10.0)
            elif feat in ["Units_No_Assessment_Semester_1", "Units_No_Assessment_Semester_2"]:
                signal = min(1.0, float(val) / 5.0)
                
            # General background
            elif feat == "First_Generation_Student":
                signal = float(val)
            elif feat == "Non_Resident_Student":
                signal = float(val)
            elif feat == "Hostel_Residency":
                signal = 1.0 - float(val)
            elif feat == "Marital_Status_Binary":
                signal = float(val)
            else:
                signal = 0.5
                
            # Multiply student signal by model feature importance
            global_weight = _feature_importances.get(feat, 0.05)
            driver_scores[feat] = signal * global_weight

        # Sort drivers in descending order and select top 2
        top_2 = sorted(driver_scores, key=driver_scores.get, reverse=True)[:2]
        
        # Pad to ensure exactly 2 drivers are returned
        if len(top_2) < 2:
            for f in _top_features:
                if f not in top_2 and f in student_row_dict:
                    top_2.append(f)
                if len(top_2) == 2:
                    break

        # Table 3.3 Interventions
        intervention = _resolve_intervention(top_2, risk_tier)

        results.append({
            "probability": round(prob, 4),
            "risk_tier": risk_tier,
            "top_drivers": top_2,
            "actionable_intervention": intervention
        })

    return results

# ─────────────────────────────────────────────────────────────────────────────
# §10  DEMONSTRATION PROFILES
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing operational EWS Inference Layer...")
    
    # Attempt to load model assets
    if not load_assets():
        print("[-] Model assets not found. Please run 'train_ews.py' first to serialize models.")
        exit(1)
        
    print("[+] Model and preprocessors successfully loaded.")
    
    # Load 3 test student profiles
    high_risk = {
        "Gender": 1, "Age_at_Matriculation": 21, "Marital_Status_Binary": 1,
        "Special_Needs_Status": 0, "Mother_Education_Level": 1, "Father_Education_Level": 1,
        "Mother_Occupation": 2, "Father_Occupation": 2, "First_Generation_Student": 1,
        "UTME_PostUME_Score": 100.0, "Secondary_School_Exit_Grade": 102.0, "Study_Mode": 1,
        "Faculty": "Engineering", "Year_of_Study": 2, "Non_Resident_Student": 1,
        "Hostel_Residency": 0, "School_Fees_Payment_Status": 0, "Fee_Arrears_Status": 1,
        "Bursary_Scholarship_Status": 0, "Units_Registered_Semester_1": 6, "Units_Passed_Semester_1": 1,
        "Assessments_Sat_Semester_1": 2, "Units_No_Assessment_Semester_1": 4, "GPA_Semester_1_5pt": 0.5,
        "Pass_Rate_Semester_1": 0.16, "Units_Registered_Semester_2": 6, "Units_Passed_Semester_2": 0,
        "Assessments_Sat_Semester_2": 1, "Units_No_Assessment_Semester_2": 5, "GPA_Semester_2_5pt": 0.0,
        "Pass_Rate_Semester_2": 0.0, "CGPA_5point_Scale": 0.25, "GPA_Change": -0.5
    }
    
    medium_risk = {
        "Gender": 0, "Age_at_Matriculation": 19, "Marital_Status_Binary": 0,
        "Special_Needs_Status": 0, "Mother_Education_Level": 3, "Father_Education_Level": 3,
        "Mother_Occupation": 4, "Father_Occupation": 5, "First_Generation_Student": 0,
        "UTME_PostUME_Score": 125.0, "Secondary_School_Exit_Grade": 130.0, "Study_Mode": 1,
        "Faculty": "Social_Sciences", "Year_of_Study": 3, "Non_Resident_Student": 1,
        "Hostel_Residency": 0, "School_Fees_Payment_Status": 1, "Fee_Arrears_Status": 0,
        "Bursary_Scholarship_Status": 0, "Units_Registered_Semester_1": 6, "Units_Passed_Semester_1": 4,
        "Assessments_Sat_Semester_1": 5, "Units_No_Assessment_Semester_1": 1, "GPA_Semester_1_5pt": 2.2,
        "Pass_Rate_Semester_1": 0.66, "Units_Registered_Semester_2": 6, "Units_Passed_Semester_2": 3,
        "Assessments_Sat_Semester_2": 5, "Units_No_Assessment_Semester_2": 1, "GPA_Semester_2_5pt": 1.8,
        "Pass_Rate_Semester_2": 0.50, "CGPA_5point_Scale": 2.0, "GPA_Change": -0.4
    }
    
    low_risk = {
        "Gender": 0, "Age_at_Matriculation": 18, "Marital_Status_Binary": 0,
        "Special_Needs_Status": 0, "Mother_Education_Level": 12, "Father_Education_Level": 14,
        "Mother_Occupation": 9, "Father_Occupation": 10, "First_Generation_Student": 0,
        "UTME_PostUME_Score": 178.0, "Secondary_School_Exit_Grade": 180.0, "Study_Mode": 1,
        "Faculty": "Sciences", "Year_of_Study": 4, "Non_Resident_Student": 0,
        "Hostel_Residency": 1, "School_Fees_Payment_Status": 1, "Fee_Arrears_Status": 0,
        "Bursary_Scholarship_Status": 1, "Units_Registered_Semester_1": 7, "Units_Passed_Semester_1": 7,
        "Assessments_Sat_Semester_1": 7, "Units_No_Assessment_Semester_1": 0, "GPA_Semester_1_5pt": 4.5,
        "Pass_Rate_Semester_1": 1.0, "Units_Registered_Semester_2": 7, "Units_Passed_Semester_2": 7,
        "Assessments_Sat_Semester_2": 7, "Units_No_Assessment_Semester_2": 0, "GPA_Semester_2_5pt": 4.6,
        "Pass_Rate_Semester_2": 1.0, "CGPA_5point_Scale": 4.55, "GPA_Change": 0.1
    }
    
    profiles = {
        "Profile A (High Risk - Academic Decline & Fees Unpaid)": high_risk,
        "Profile B (Medium Risk - Moderate Academic Struggles)": medium_risk,
        "Profile C (Low Risk - High Performer with Scholarship)": low_risk
    }
    
    for label, student in profiles.items():
        print(f"\n{label}:")
        pred = predict_dropout_risk(student)
        print(json.dumps(pred, indent=2))
        print("-" * 50)
