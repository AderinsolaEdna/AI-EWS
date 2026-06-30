# AI-Driven Early Warning System (EWS) for Predicting Student Dropout in Nigerian Universities

An operational, end-to-end Machine Learning pipeline and interactive dark-mode web dashboard designed to predict student dropout risk in Nigerian Universities. The system leverages machine learning classifiers (Logistic Regression, Decision Trees, Random Forests, XGBoost, and MLP Neural Networks) trained on a contextualized dataset of **4,424 student records**, applies Class Imbalance correction (SMOTE), and resolves individual student risk drivers to suggest targeted, localized institutional interventions.

---

## 🌟 Key Features

*   **Multi-Model ML Pipeline (`train_ews.py`):** Trains and evaluates five classification algorithms using Stratified 10-Fold Cross-Validation, handles class imbalance via SMOTE (Synthetic Minority Over-sampling Technique), scales features using `StandardScaler` to prevent leakage, and exports the best-performing model based on Recall priority.
*   **Contextual Explainable Inference Layer (`ews_inference.py`):** Combines model feature importances with individual student signals to isolate the top two primary risk drivers. Includes a vectorized batch prediction engine.
*   **Rule-Based Intervention Engine:** Integrates a rule-based lookup mapping student risk tiers (High, Medium, Low) and driver domains (Financial, Academic, Welfare, Social) to contextualized, actionable intervention protocols (e.g., Bursary referrals, peer tutoring, academic adviser plans, and Dean of Students reviews).
*   **Lightweight Web Server Backend (`server.py`):** Serves static files and exposes a local JSON API using Python's standard `http.server` module:
    *   `POST /api/predict` - Real-time risk prediction for individual student inputs.
    *   `GET /api/metrics` - Exposes model evaluation metrics and confusion matrices.
    *   `GET /api/students` - Paginated querying, search, sorting, and filtering of precomputed student files.
    *   `GET /api/student-detail` - Fetches detailed features and prediction outputs for a specific student ID.
*   **Interactive Dark-Mode Dashboard (`dashboard.html`):** A premium web dashboard designed with modern CSS (glassmorphism, CSS variables, Google Fonts: *Inter* and *Outfit*). Includes:
    *   **Student Records Database View:** High-performance paginated table with sorting (by ID, CGPA, and Risk Probability) and filtering (by Faculty and Risk Tier).
    *   **Individual Predictor Form View:** Interactive form broken into logical categories (Academic Performance, Background & Admission, Financial & Support, Socio-Demographic) with quick-load buttons for High, Medium, and Low-risk student templates.
    *   **Dynamic Risk Metrics View:** Renders risk probabilities, gauge charts, student-specific driver breakdowns, and actionable intervention recommendation cards.

---

## 📁 Directory Structure

```text
AI-EWS system/
├── data/
│   └── Actual_nigerian_student_dropout_dataset.csv  # Dataset (4,424 records)
├── models/
│   ├── ews_best_model.pkl                          # Serialized best ML classifier (XGBoost)
│   ├── ews_scaler.pkl                              # Fit StandardScaler instance
│   ├── ews_feature_names.pkl                       # Alignment feature columns list
│   ├── ews_top_features.pkl                        # Sorted list of key predictive features
│   ├── ews_feature_importances.pkl                 # Feature importance mapping dict
│   └── metrics_summary.json                        # Pre-calculated test metrics for all models
├── reports/
│   └── ews_evaluation_report.png                   # Evaluation charts (ROC curves, confusion matrices)
├── dashboard.html                                  # Premium frontend web interface
├── ews_inference.py                                # Inference pipeline & rule-based engine
├── server.py                                       # Web server & REST API backend
├── train_ews.py                                    # Model training & comparison pipeline
└── requirements.txt                                # Python dependencies list
```

---

## ⚙️ Installation & Requirements

Ensure you have Python 3.9+ installed.

1. Clone or navigate to the project directory:
   ```bash
   cd "AI-EWS system"
   ```

2. Install all required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

*Key dependencies include:* `numpy`, `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `joblib`, `matplotlib`, and `seaborn`.

---

## 🚀 How to Run the System

### Step 1: Execute the Training Pipeline
Train the classifiers, calculate metrics, and save serialized assets to `models/` by running:
```bash
python train_ews.py
```
This script runs cross-validation, evaluates held-out test data, saves the best model, and outputs a visual summary report to `reports/ews_evaluation_report.png`.

### Step 2: Start the Web Backend
Launch the lightweight web server and API layer:
```bash
python server.py
```
On startup, the server loads model assets and precomputes predictions for the entire student dataset to speed up database queries.

### Step 3: Open the Dashboard
Open your web browser and navigate to:
```text
http://localhost:8000/
```
From here, you can search and explore the student database or input arbitrary student attributes to test the Early Warning System manually.

---

## 📊 Model Evaluation & Metrics

The training pipeline evaluates models on a **held-out 20% test set**. The models are optimized for **Recall** (minimizing false negatives to ensure no at-risk student is missed) with **F1-Score** acting as the tiebreaker.

Below is the performance summary generated from `models/metrics_summary.json`:

| Model | Accuracy | Precision | Recall (Primary) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Selected Best)** | **88.36%** | **81.10%** | **83.10%** | **82.09%** | **92.39%** |
| **Random Forest** | 88.59% | 81.88% | 82.75% | 82.31% | 93.20% |
| **Decision Tree** | 84.52% | 73.94% | 79.93% | 76.82% | 90.68% |
| **MLP Neural Network** | 85.54% | 77.46% | 77.46% | 77.46% | 90.04% |
| **Logistic Regression** | 86.10% | 79.93% | 75.70% | 77.76% | 90.67% |

---

## 📋 Dataset Schema & Features

The model ingests 36 primary columns contextualized for Nigerian tertiary institutions:

### 🎓 Academic Performance
*   `CGPA_5point_Scale`: Cumulative GPA (typically on a 1.0 - 5.0 scale in Nigeria).
*   `GPA_Semester_1_5pt` / `GPA_Semester_2_5pt`: First and second-semester grade point averages.
*   `GPA_Change`: Rate of academic improvement or decline (`Sem 2 GPA - Sem 1 GPA`).
*   `Units_Registered_Semester_1` / `Units_Registered_Semester_2`: Total credit units registered.
*   `Units_Passed_Semester_1` / `Units_Passed_Semester_2`: Total credit units successfully completed.
*   `Pass_Rate_Semester_1` / `Pass_Rate_Semester_2`: Ratio of passed units to registered units.
*   `Assessments_Sat_Semester_1` / `Assessments_Sat_Semester_2`: Count of exams, tests, and practicals completed.
*   `Units_No_Assessment_Semester_1` / `Units_No_Assessment_Semester_2`: Registered units without completed assessments.

### 🏛️ Admission & Demographics
*   `Faculty`: Faculty affiliation (Engineering, Sciences, Arts, Social Sciences, Medicine, Law).
*   `Year_of_Study`: Current year (1 to 5).
*   `UTME_PostUME_Score`: Consolidated score from university entrance examinations.
*   `Secondary_School_Exit_Grade`: Final high school exit examination score (e.g., WAEC/NECO grade aggregation).
*   `Study_Mode`: Full-time (1) vs. Part-time (0).
*   `Gender`: Female (0) vs. Male (1).
*   `Age_at_Matriculation`: Age of entry.

### 💰 Financial & Welfare Support
*   `School_Fees_Payment_Status`: Paid (1) vs. Unpaid (0).
*   `Fee_Arrears_Status`: Has outstanding arrears (1) vs. No arrears (0).
*   `Bursary_Scholarship_Status`: Recipient of financial aid/scholarship (1) vs. None (0).
*   `Non_Resident_Student`: Off-campus commuter (1) vs. On-campus resident (0).
*   `Hostel_Residency`: Successfully allocated a bedspace in university hostels (1) vs. Not allocated (0).

### 👥 Socio-Demographic & Parental Background
*   `First_Generation_Student`: True (1) if parents did not attend tertiary education.
*   `Marital_Status_Binary`: Single (0) vs. Married/Other (1).
*   `Special_Needs_Status`: Presence of documented physical or learning disabilities (1) vs. None (0).
*   `Mother_Education_Level` / `Father_Education_Level`: Coded levels of parental educational attainment.
*   `Mother_Occupation` / `Father_Occupation`: Coded levels of parental occupation categories.

---

## 🛡️ Rule-Based Intervention Framework

Based on the isolated top risk drivers, the EWS recommends targeted institutional support:

| Domain | Linked Features | Risk Level | Contextual Intervention Protocol |
| :--- | :--- | :---: | :--- |
| **Financial** | `Fee_Arrears_Status`, `School_Fees_Payment_Status` | **High** | Immediate referral to Bursary Office for fee deferral or bursary application; notify Dean of Student Affairs. |
| **Financial** | `Fee_Arrears_Status`, `School_Fees_Payment_Status` | **Medium** | Advisory notification on bursary and financial aid options; scheduled welfare officer meeting. |
| **Academic** | `CGPA_5point_Scale`, `GPA_Semester_1_5pt`, `GPA_Change`, etc. | **High** | Referral to academic adviser for urgent improvement plan; consideration for supplementary examination access. |
| **Academic** | `CGPA_5point_Scale`, `GPA_Semester_1_5pt`, `GPA_Change`, etc. | **Medium** | Recommendation for peer tutoring; adviser check-in within two weeks. |
| **Course Load** | `Units_Passed_Semester_1`, `Units_Passed_Semester_2` | **High** | Academic review with Head of Department; assessment of course load and repeat options. |
| **Social** | `First_Generation_Student` | **Any** | Referral to mentorship programme; connection with senior peer students. |
| **Welfare** | `Hostel_Residency`, `Non_Resident_Student` | **Medium** | Welfare check on off-campus living conditions; hostel allocation information. |
| **Multi-Agency**| *Combination of 2 or more domains at high risk* | **High** | Multi-agency intervention (adviser, welfare officer, bursary); case flagged for Dean of Students review. |

---

## 📄 License & Authorship

*   **Author:** Senior Data Scientist & ML Engineer (EDM Specialisation)
*   **Methodology Context:** Prepared in accordance with Academic EWS development standards (Chapter 3).
