# Telecom_Customer_Churn
# Telecom Customer Churn Prediction

This project aims to build a machine learning model to predict whether a telecom customer will churn or not, based on service usage patterns and contract features. Churn prediction is vital for telecom companies to reduce customer loss and improve service retention.

---

##  Objective

Design and evaluate a machine learning pipeline that:
- Predicts customer churn with high precision and recall
- Applies both label encoding and one-hot encoding strategies
- Tunes hyperparameters using Grid Search and Cross-Validation
- Uses key classification metrics (accuracy, F1-score, ROC AUC) for evaluation

---

##  Dataset Overview

- **File Used**: `Churn.csv`
- **Rows**: 3333
- **Columns**: 21
- **Target Variable**: `Churn` (0 = No churn, 1 = Churn)

---

##  Project Workflow

### 1. Data Loading
- Loaded dataset using pandas.
- Displayed top entries, checked structure and missing values.

### 2. Data Preprocessing
- Dropped uninformative columns: `State`, `Area_Code`, `Phone`
- Replaced `yes`/`no` strings in `Churn`, `Intl_Plan`, `Vmail_Plan` with binary values
- Applied:
  - **Label Encoding**
  - **One-Hot Encoding**
- Applied **Normalization** using `StandardScaler` and `MinMaxScaler`

### 3.  Feature Engineering
- Explored correlation and unique value counts
- Filtered non-impactful features before modeling

---

## Machine Learning Model

### Base Model: `DecisionTreeClassifier`
- Used `train_test_split` with 70% training and 30% testing
- Performed prediction and comparison with actual churn labels
- Visualized tree structure using `plot_tree()`

---

## Evaluation Metrics

| Metric                | Value      |
|-----------------------|------------|
| **Accuracy**          | 91.9%      |
| **Precision**         | 74%        |
| **Recall**            | 68%        |
| **F1-Score**          | 0.71       |
| **True Positive Rate**| 0.68       |
| **False Positive Rate**| 0.04      |
| **ROC AUC (Best Model)** | 0.85   |

- Visualizations: Confusion Matrix and ROC Curve
- Classification Reports generated for both encoding strategies

---

##  Grid Search & Cross-Validation

- Applied `GridSearchCV` with:
  - `criterion`: `gini`, `entropy`
  - `max_depth`: 3 to 7
- **Best parameters found**:  
  `{'criterion': 'entropy', 'max_depth': 6}`

- Evaluated mean test scores across all CV folds
- ROC Curve area improved to **0.85** with tuned model

---

## Tech Stack

- **Languages**: Python 3
- **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
- **Tools**: Jupyter Notebook

---

##  How to Run

1. Clone this repository:
   ```bash
  git clone https://github.com/Joy2004114/Telecom_Customer_Churn.git
  cd Telecom_Customer_Churn

