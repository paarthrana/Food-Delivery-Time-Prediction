# Food Delivery Time Prediction

An end-to-end machine learning study auditing, benchmarking, and evaluating continuous regression and binary classification models for food delivery time estimation.

---

## 1. Project Overview

Accurate food delivery time estimation is a foundational challenge in on-demand logistics, directly influencing customer satisfaction, driver dispatch efficiency, and kitchen scheduling.

This repository provides an audited, reproducible evaluation of standard baseline machine learning workflows applied to delivery order metadata:
1. **Continuous Regression**: Predicting exact `Delivery_Time` (in minutes) using Ordinary Least Squares (OLS) Linear Regression.
2. **Binary Classification**: Classifying deliveries into relative operational cohorts (**Fast** vs. **Delayed**) using Logistic Regression with standardized features.

### Operational Definition of "Delayed"
Within this dataset, **Delayed** is defined relative to the dataset median delivery duration ($72.78$ minutes), categorizing the slower 50% of completed deliveries. **This label does not represent a breached customer Service Level Agreement (SLA) or missed promised delivery deadline**, but rather an internal median split for relative speed classification.

---

## 2. Repository Structure

```
Food-Delivery-Time-Prediction/
├── data/
│   ├── processed/          # Placeholder directory for processed artifacts
│   └── raw/
│       └── foodtime.csv    # Source tabular delivery dataset (200 rows, 15 columns)
├── notebooks/
│   └── main.ipynb          # End-to-end Jupyter notebook structured into 9 sections
├── .gitignore              # Git ignore rules for checkpoints, caches, and environments
├── requirements.txt        # Minimal pinned dependencies actually used
└── README.md               # Project documentation and empirical audit report
```

---

## 3. Dataset Description

The dataset (`data/raw/foodtime.csv`) contains **200 completed delivery orders** across 15 attributes:

| Feature Name | Type | Description | Handling |
| :--- | :--- | :--- | :--- |
| `Order_ID` | String (Identifier) | Unique order identifier (`ORD0001` - `ORD0200`) | Dropped (non-predictive primary key) |
| `Customer_Location` | String (Coordinates) | Latitude/longitude coordinate tuple string | Dropped (unique per row; distance provided) |
| `Restaurant_Location` | String (Coordinates) | Latitude/longitude coordinate tuple string | Dropped (unique per row; distance provided) |
| `Distance` | Float (km) | Linear distance between restaurant and customer ($0.52 - 24.90$ km) | Retained as continuous feature |
| `Weather_Conditions` | Categorical (Nominal) | `Rainy`, `Cloudy`, `Snowy`, `Sunny` | One-hot encoded (`drop_first=True`) |
| `Traffic_Conditions` | Categorical (Ordinal) | `Low`, `Medium`, `High` | Ordinal encoded ($0, 1, 2$) |
| `Delivery_Person_Experience` | Integer (Years) | Courier experience ($1 - 10$ years) | Retained as continuous feature |
| `Order_Priority` | Categorical (Ordinal) | `Low`, `Medium`, `High` | Ordinal encoded ($0, 1, 2$) |
| `Order_Time` | Categorical (Nominal) | `Morning`, `Afternoon`, `Evening`, `Night` | One-hot encoded (`drop_first=True`) |
| `Vehicle_Type` | Categorical (Nominal) | `Car`, `Bike`, `Bicycle` | One-hot encoded (`drop_first=True`) |
| `Restaurant_Rating` | Float ($1.0 - 5.0$) | Rating of the food establishment ($2.5 - 5.0$) | Retained as continuous feature |
| `Customer_Rating` | Float ($1.0 - 5.0$) | Customer satisfaction rating ($2.6 - 5.0$) | Retained as continuous feature |
| `Order_Cost` | Float | Financial total of the order | Retained as continuous feature |
| `Tip_Amount` | Float | Gratuity provided by the customer | Retained as continuous feature |
| **`Delivery_Time`** | Float (Minutes) | Total trip duration ($15.23 - 119.67$ min) | **Target Variable** (Mean: $70.49$, Median: $72.78$) |

---

## 4. Methodology & Key Code Fixes

### Data Preprocessing & Hygiene
- **Missing Values & Duplicates**: Audited with zero missing cells ($0$ `null`) and zero duplicate rows across all 200 records.
- **Identifier Removal**: `Order_ID`, `Customer_Location`, and `Restaurant_Location` were removed to prevent arbitrary memorization and noise, as linear spatial distance is already captured numerically in `Distance`.

### Feature Engineering & Bug Fixes
- **Ordinal Encoding Bug Fix**:
  - *Identified Defect*: In the initial implementation, `OrdinalEncoder()` was initialized without arguments, and a subsequent parameterized instantiation was discarded, falling back to lexicographical order (`High`=0, `Low`=1, `Medium`=2).
  - *Correction*: Explicitly initialized `OrdinalEncoder(categories=[['Low', 'Medium', 'High'], ['Low', 'Medium', 'High']])` for `Traffic_Conditions` and `Order_Priority`, ensuring a verified monotonic mapping:
    $$\text{Low} \rightarrow 0, \quad \text{Medium} \rightarrow 1, \quad \text{High} \rightarrow 2$$
- **Nominal One-Hot Encoding**: Applied `pd.get_dummies(..., drop_first=True, dtype=int)` on `Weather_Conditions`, `Order_Time`, and `Vehicle_Type` to prevent multicollinearity (dummy variable trap).

### Leakage Controls & Splitting
- **Linear Regression**: 80/20 train/test split ($N_{train}=160$, $N_{test}=40$, `random_state=42`).
- **Logistic Regression**: 
  - Stratified 80/20 split on `Delivery_Status` to preserve exact 50/50 class balance in train and test splits.
  - Strict exclusion of both continuous `Delivery_Time` and target label `Delivery_Status` from predictor matrices.
  - `StandardScaler` fitted **strictly on training data** and transformed on test data to eliminate distribution snooping.

---

## 5. Verified Empirical Results

### Continuous Regression (Delivery Time in Minutes)
Evaluated on the held-out test split ($N=40$):

| Metric | Linear Regression (OLS) | Naive Training Mean Baseline | Naive Training Median Baseline |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **$27.18$ min** | **$25.53$ min** | **$25.38$ min** |
| **Root Mean Squared Error (RMSE)** | **$31.91$ min** | **$29.83$ min** | **$29.87$ min** |
| **Mean Squared Error (MSE)** | $1018.38$ | $889.87$ | $892.48$ |
| **Coefficient of Determination ($R^2$)** | **$-0.1011$** | **$-0.0012$** | — |

*Note on Negative $R^2$*: In out-of-sample evaluation, $R^2$ measures relative variance explained compared to the training mean. An $R^2$ of $-0.1011$ indicates the fitted linear model has higher residual error than simply predicting the training mean ($71.39$ minutes).

### Binary Classification (Fast vs. Delayed)
Evaluated on the held-out test split ($N=40$, 20 Fast, 20 Delayed; Positive Class = `Delayed`):

| Evaluation Metric | Logistic Regression | Random Guessing Baseline |
| :--- | :---: | :---: |
| **Accuracy** | **$40.00\%$** ($16 / 40$) | $50.00\%$ ($20 / 40$) |
| **Precision (`Delayed`)** | **$38.89\%$** ($7 / 18$) | $50.00\%$ |
| **Recall (`Delayed`)** | **$35.00\%$** ($7 / 20$) | $50.00\%$ |
| **F1-Score (`Delayed`)** | **$0.3684$** | $0.5000$ |
| **ROC-AUC Score** | **$0.4075$** | $0.5000$ |

#### Test Set Confusion Matrix
```
                 Predicted Fast    Predicted Delayed
Actual Fast            9                  11
Actual Delayed        13                   7
```

---

## 6. Diagnostic Investigation: Understanding the Sub-0.5 ROC-AUC

In previous iterations, the observation of an ROC-AUC score below $0.5$ ($0.4075$) prompted questions regarding whether probabilities were reversed or labels inverted (since $1 - 0.4075 = 0.5925$).

Our investigation verified:
1. **Class Order & Probability Selection**:
   - `classifier.classes_` is ordered `['Delayed', 'Fast']`.
   - Index `0` corresponds to `Delayed`. The probability of delay is extracted via `predict_proba(X)[:, 0]`.
   - Ground truth binary labels map `Delayed` to `1` and `Fast` to `0`.
   - The metric calculation is mathematically and implementationally verified.
2. **Why Reversing Predictions is Methodologically Unsound**:
   - Inverting probabilities (`1 - y_prob`) post-hoc because the test set score dipped below $0.5$ represents **data snooping**.
   - There is no theoretical or operational basis to claim that indicators of delay predict faster deliveries. If applied to real-world out-of-sample data, an inverted noise model would continue to fail.
3. **Statistical Explanation**:
   - Tabular features in this dataset share near-zero correlation with delivery duration ($|r| < 0.10$).
   - On a small training partition ($N=160$) with $16$ predictors, the model fits sample noise.
   - When evaluated on a small test split ($N=40$), sample variance and negative generalization of spurious correlations cause empirical test performance to dip below chance ($0.4075$ AUC). This is an authentic empirical demonstration of model limitations on uninformative data.

---

## 7. Limitations & Recommendations

### Limitations
- **Small Sample Size**: With only 200 records ($160$ train, $40$ test), statistical power is low, making metrics sensitive to individual sample splits.
- **Uncorrelated Features**: Standard operational attributes (distance, weather, vehicle, traffic) exhibit near-zero correlation with delivery time, suggesting synthetic generation or missing critical operational dynamics.
- **Missing Operational Latencies**: Real delivery durations are dominated by kitchen preparation delay, courier transit to restaurant, order pickup queue time, and real-time street traffic, none of which are recorded in this dataset.

### Actionable Recommendations & Hypotheses
- **Data Collection (Hypothesis)**: *We hypothesize that recording kitchen cooking duration and courier pickup wait time would capture the majority of delivery variance, enabling effective predictive modeling.*
- **Geographic Routing (Hypothesis)**: *We hypothesize that replacing straight-line distance with street-level routing API duration estimates would establish a strong linear predictor.*
- **Production Readiness Decision**: **Neither model is production-ready.** Deploying these models would produce worse predictions than simple historical median heuristics ($72.78$ minutes). Operational teams should rely on naive baseline heuristics until predictive telemetry is integrated.

---

## 8. Setup and Execution

### Prerequisites
- Python 3.10+
- Recommended: Virtual environment (`venv` or `conda`)

### Installation
```bash
# Clone the repository
git clone https://github.com/paarthrana/Food-Delivery-Time-Prediction.git
cd Food-Delivery-Time-Prediction

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Running the Notebook
```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/main.ipynb
```

Or execute headlessly from the command line:
```bash
jupyter nbconvert --to notebook --execute notebooks/main.ipynb --inplace
```

---

## 9. Suggested Resume Bullet

> *Audited and refactored a food delivery time ML pipeline in scikit-learn; resolved ordinal encoding defects and leakage vectors, established naive baseline benchmarks, and conducted rigorous diagnostic evaluations of Linear and Logistic Regression workflows across regression and classification tasks.*

