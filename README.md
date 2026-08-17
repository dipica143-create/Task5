# Telco Customer Churn Prediction Pipeline

An end-to-end Machine Learning pipeline built with Python and `scikit-learn` to predict customer churn for telecommunications data. This project covers data ingestion, rigorous exploratory checks, automated data preprocessing with `ColumnTransformer` and `Pipeline`, model training (Logistic Regression, Random Forest, Gradient Boosting), cross-validation, and extensive evaluation metrics (ROC-AUC, Confusion Matrix, Classification Reports).

---

## Table of Contents
1. [Dataset](#dataset)
2. [Project Workflow](#project-workflow)
3. [Installation & Requirements](#installation--requirements)
4. [Usage](#usage)
5. [Model Evaluation & Results](#model-evaluation--results)
6. [License](#license)

---

## Dataset
* **Source:** Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
* **Size:** 7,043 rows and 21 columns.
* **Target Variable:** `Churn` (Binary: Yes/No encoded as 1/0).
* **Key Features:** Customer demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account info (`tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`), and subscribed services (`PhoneService`, `InternetService`, `TechSupport`, `StreamingTV`, etc.).

---

## Project Workflow

1. **Data Ingestion & Inspection:** 
   * Loads the CSV file using `pandas`.
   * Inspects data shapes, data types, and missing values.
2. **Data Preprocessing & Feature Engineering:**
   * Encodes target labels and cleans numeric fields like `TotalCharges` (coercing errors and imputing missing values with the median).
   * Drops unique identifiers (`customerID`) that offer no predictive power.
   * Implements a robust `ColumnTransformer` pipeline:
     * **Numerical Pipeline:** Median imputation + Feature scaling (`StandardScaler`).
     * **Categorical Pipeline:** Most-frequent imputation + One-Hot Encoding (`OneHotEncoder`)[cite: 1].
   * Performs stratified train-test splitting (80% train, 20% test) to preserve class distributions[cite: 1].
3. **Model Training & Cross-Validation:**
   * Utilizes `StratifiedKFold` (CV = 5) to reliably validate performance across metrics like Accuracy and ROC-AUC[cite: 1].
   * Trains baseline and advanced classifiers (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`)[cite: 1].
4. **Evaluation & Visualization:**
   * Generates classification reports (Precision, Recall, F1-Score)[cite: 1].
   * Plots **Confusion Matrices** and **ROC Curves (with AUC scores)** for model comparison[cite: 1].

---

## Installation & Requirements

Ensure you have Python 3.8+ installed. You will need the following data science libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
