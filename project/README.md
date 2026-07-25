# E-Commerce Customer Analytics Platform

Production-ready customer intelligence system built on the Online Retail II dataset.

This project combines:
1. Unsupervised learning for customer segmentation (RFM + K-Means)
2. Supervised regression for Customer Lifetime Value (CLV)
3. Supervised classification for churn prediction
4. Rule-based business recommendations
5. SHAP-powered explainability for model transparency
6. Interactive Streamlit dashboard for business users

---

## 1. Problem Statement

E-commerce teams need one unified platform that answers four practical questions:

1. Which customer segment does a user belong to?
2. What future value (CLV) can we expect from this customer?
3. What is the probability this customer churns?
4. What action should the business take now?

This repository solves all four in one pipeline and exposes outputs through a dashboard.

---

## 2. Dataset
- Download from Kaggle:
- https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- Source: Online Retail II (UCI / Kaggle format)
- File location: project/data/online_retail_II.csv
- Core columns used:
  - Invoice
  - StockCode
  - Description
  - Quantity
  - InvoiceDate
  - Price
  - Customer ID
  - Country


---

## 3. End-to-End Architecture

```text
Raw CSV (project/data/online_retail_II.csv)
      |
      v
Data Preprocessing
  - schema validation
  - type conversion
  - remove invalid rows
  - create TotalAmount
      |
      v
Processed Dataset (project/data/processed_online_retail_II.csv)
      |
      +-----------------------------+
      |                             |
      v                             v
RFM Clustering                   CLV Regression
(K-Means + Elbow)               (Linear/RF/XGB)
      |                             |
      +-------------+---------------+
               |
               v
          Churn Classification
         (Logistic/RF/XGB)
               |
               v
      Recommendation Rules Engine
               |
               v
            SHAP Explainability
          (global + per-customer)
               |
               v
        Streamlit Dashboard (5 pages)
```

---

## 4. Project Structure

```text
project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── online_retail_II.csv
│   └── processed_online_retail_II.csv
├── models/
│   ├── rfm_kmeans_artifacts.joblib
│   ├── clv_model_artifacts.joblib
│   ├── churn_model_artifacts.joblib
│   ├── customer_segments.csv
│   ├── customer_predictions.csv
│   ├── clv_feature_importance.csv
│   ├── churn_feature_importance.csv
│   ├── clv_shap_importance.png
│   ├── churn_shap_importance.png
│   ├── elbow_plot.png
│   ├── rfm_clusters_2d.png
│   └── training_report.json
├── notebooks/
│   ├── eda.ipynb
│   ├── rfm_clustering.ipynb
│   ├── clv_regression.ipynb
│   └── churn_classification.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── recommendation_engine.py
│   ├── explainability.py
│   ├── train_models.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## 5. Technology Stack

### Core Language
- Python 3.11 (pinned in `runtime.txt`; also verified on 3.11–3.14)

> macOS note: XGBoost needs the OpenMP runtime. If `import xgboost` fails with
> `Library not loaded: @rpath/libomp.dylib`, run `brew install libomp`.

### Data and ML Libraries
- pandas, numpy
- scikit-learn
- xgboost
- joblib
- shap (installed for explainability extensions)

### Visualization and UI
- matplotlib, seaborn
- plotly
- streamlit

### Notebook Environment
- jupyter

---

## 6. How the System Works (Detailed)

### 6.1 Data Preprocessing
Implemented in src/data_preprocessing.py

Pipeline logic:
1. Load CSV with tolerant parsing for malformed lines
2. Validate expected schema
3. Convert date and numeric fields
4. Remove null-critical rows
5. Remove non-positive Quantity and Price rows
6. Create engineered columns:
  - CustomerID (integer normalized from Customer ID)
  - TotalAmount = Quantity × Price
7. Save cleaned dataset

Output artifact:
- project/data/processed_online_retail_II.csv

### 6.2 Feature Engineering
Implemented in src/feature_engineering.py

Shared customer-level features:
- Recency
- Frequency
- Monetary
- AverageBasketSize
- PurchaseFrequency
- Country

Additional feature datasets:
- CLV dataset with FutureRevenue target (next 90 days window)
- Churn dataset with ChurnLabel based on recency threshold

### 6.3 Module 1: RFM Clustering
1. Build customer RFM
2. Standardize with StandardScaler
3. Compute Elbow inertias for K = 2..10
4. Select K using elbow heuristic
5. Train K-Means
6. Map numeric clusters to business labels:
  - Champions
  - Loyal Customers
  - At Risk
  - Lost Customers (if present by K)

Saved outputs:
- rfm_kmeans_artifacts.joblib
- customer_segments.csv
- elbow_plot.png
- rfm_clusters_2d.png

### 6.4 Module 2: CLV Regression
Target:
- FutureRevenue over next 90 days

Features:
- Recency, Frequency, Monetary
- AverageBasketSize, PurchaseFrequency
- Country (one-hot encoded)

Models trained:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor (if available)

Metrics:
- RMSE
- R2

Best model selection criterion:
- Minimum RMSE

Saved outputs:
- clv_model_artifacts.joblib
- clv_feature_importance.csv

### 6.5 Module 3: Churn Classification
Label definition:
- ChurnLabel = 1 if Recency > threshold (default 90 days)

Features:
- Recency, Frequency, Monetary
- PredictedCLV
- ClusterLabel

Models trained:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier (if available)

Metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Best model selection criterion:
- Maximum ROC-AUC

Saved outputs:
- churn_model_artifacts.joblib
- churn_feature_importance.csv

### 6.6 Business Recommendation Layer (Feature 1)
Implemented in src/recommendation_engine.py and used in src/predict.py

Rules:
1. If churn_probability > 0.7 → Offer Discount
2. If predicted CLV is high → Mark as VIP
3. If cluster label = At Risk → Send Retention Campaign
4. Else fallback → Maintain Engagement

Output fields:
- CustomerSegment
- PriorityLevel
- RecommendedAction

### 6.7 Explainability (Feature 2)
Implemented in src/explainability.py and integrated in both training and prediction flow.

Capabilities:
1. Global explainability artifacts
  - SHAP importance summary for CLV model
  - SHAP importance summary for churn model
2. Per-customer explanations at inference time
  - Top SHAP contributing features for CLV
  - Top SHAP contributing features for churn
  - Human-readable explanation strings for dashboard users

Saved outputs:
- clv_feature_importance.csv
- churn_feature_importance.csv
- clv_shap_importance.png
- churn_shap_importance.png

---

## 7. Streamlit Dashboard (Feature 3 Upgrade)

Implemented in app/streamlit_app.py with 5 pages:

1. Overview Dashboard
  - KPI cards for sales, customers, average order value
  - Monthly trend
  - High churn risk percentage and cluster mix

2. Customer Segmentation
  - Cluster distribution
  - RFM scatter views

3. CLV Prediction
  - Input customer ID
  - Predicted CLV card
  - Why this prediction? panel with SHAP top features

4. Churn Prediction
  - Input customer ID
  - Churn probability card
  - Churn risk gauge
  - Why this prediction? panel with SHAP top features

5. Recommendations
  - Input customer ID
  - Unified decision output:
    - Cluster Label
    - Predicted CLV
    - Churn Probability
    - Priority level
    - Recommended action + channel
  - Integrated explanation context from SHAP signals

### 7.1 Output Contract from predict.py
For a given Customer ID, the prediction response now includes:
- ClusterLabel
- PredictedCLV
- ChurnProbability
- Decision (Segment, PriorityLevel, RecommendedAction, RecommendedChannel)
- Explanations (human-readable CLV/churn reasons)
- ShapTopFeatures (top contributors for CLV and churn)

---

## 8. How to Run Locally

Run commands from repository root.

### Step 1: Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r project/requirements.txt
```

### Step 3: Preprocess data
```bash
python project/src/data_preprocessing.py \
  --input_csv project/data/online_retail_II.csv \
  --output_csv project/data/processed_online_retail_II.csv
```

### Step 4: Train all models
```bash
python project/src/train_models.py \
  --processed_csv project/data/processed_online_retail_II.csv \
  --models_dir project/models \
  --horizon_days 90 \
  --churn_days 90
```

### Step 5: Launch dashboard
```bash
streamlit run project/app/streamlit_app.py
```

Then open the URL printed in terminal (typically http://localhost:8501).

---

## 9. One-Line Run (after repository is cloned)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r project/requirements.txt && python project/src/data_preprocessing.py --input_csv project/data/online_retail_II.csv --output_csv project/data/processed_online_retail_II.csv && python project/src/train_models.py --processed_csv project/data/processed_online_retail_II.csv --models_dir project/models --horizon_days 90 --churn_days 90 && streamlit run project/app/streamlit_app.py
```

Alternative launcher:
```bash
./run.sh
```

---

## 10. Notebooks

- notebooks/eda.ipynb: data understanding and quality checks
- notebooks/rfm_clustering.ipynb: clustering workflow
- notebooks/clv_regression.ipynb: CLV modeling experiments
- notebooks/churn_classification.ipynb: churn modeling experiments

---

## 11. Deployment (Streamlit Cloud)

1. Push repository to GitHub
2. Go to Streamlit Community Cloud
3. Create a new app and select:
  - Repo: your repository
  - Branch: main
  - App file: project/app/streamlit_app.py
4. Add project/data/online_retail_II.csv to repo or configure remote data access
5. Deploy

Deployment link:
- https://ecommerce-customer-analytics-ml-iw5v4ndz7tnjtf3zxweari.streamlit.app

---

## 12. Troubleshooting

1. Dashboard opens but shows warning about missing artifacts
  - Run the training step first to generate files in project/models

2. Module not found errors
  - Ensure virtual environment is active
  - Reinstall dependencies from project/requirements.txt

3. Streamlit port issue
  - Use a custom port:
    streamlit run project/app/streamlit_app.py --server.port 8503

4. Dataset not found
  - Confirm project/data/online_retail_II.csv exists

---

## 12.5 Results

Numbers below come from `project/models/training_report.json`, regenerated by
`train_models.py`. Holdout is a 20% split; CV is 5-fold on the training split.

### Churn classification (best: XGBoostClassifier)

| Metric | Value |
|---|---|
| ROC-AUC (holdout) | 0.879 |
| ROC-AUC (5-fold CV) | 0.877 ± 0.005 |
| F1 | 0.838 |
| Precision / Recall | 0.811 / 0.868 |
| Brier score | 0.138 |

Features are built strictly from history up to a cutoff date; the label is
whether the customer purchased in the 90-day window *after* that cutoff. No
feature is computed from the label window.

### CLV regression (best: XGBoost on log1p target)

| Metric | Value |
|---|---|
| Spearman rank correlation | 0.588 |
| Top-decile lift | 3.63× |
| MAE | 565 |
| RMSE | 5633 |
| R² (holdout) | 0.032 |

**Why R² is reported but not used for model selection.** Future 90-day revenue
is extreme-tailed: the single largest customer in the holdout set contributes
**81% of its total sum of squares**, and is larger than anything in training. R²
and RMSE therefore rank models mostly by how they happened to fit one
unlearnable point, and they barely move across very different models. Since
every downstream consumer of CLV here (VIP flagging, priority ordering, campaign
targeting) uses the *ranking*, models are selected on Spearman correlation and
reported with top-decile lift.

Read the lift figure as: **the top 10% of customers this model flags go on to
spend 3.6× the average customer.** That is the number worth acting on.

**Honest limitation.** All four candidate models land within a narrow band
(Spearman 0.53–0.59, lift 3.63–3.74). The ceiling here is the feature set — five
RFM-style aggregates plus country — not the algorithm. Meaningful gains need
richer features (product-category mix, inter-purchase-time distribution,
seasonality, tenure curves), not more model tuning.

---

## 13. Current Status

Implemented and connected end-to-end:
1. Data preprocessing
2. RFM clustering
3. CLV regression
4. Churn classification
5. Recommendation engine
6. SHAP explainability (global + per-customer)
7. Streamlit dashboard (Feature 3 upgraded UI)

This means you can now enter a Customer ID in the dashboard and directly get:
- Cluster label
- Predicted CLV
- Churn probability
- Recommended action

## Branch
 - feature/ml-enhancements-v2
 
