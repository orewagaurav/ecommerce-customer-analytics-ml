# E-Commerce Customer Analytics Platform

Production-ready customer intelligence system built on the Online Retail II dataset.

This project combines:
1. Unsupervised learning for customer segmentation (RFM + K-Means)
2. Supervised regression for Customer Lifetime Value (CLV)
3. Supervised classification for churn prediction
4. Rule-based business recommendations
5. Interactive Streamlit dashboard for business users

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
        Streamlit Dashboard (5 pages)
```

---

## 4. Project Structure

```text
project/
├── app/
│   └── streamlit_app.py
├── data/
│   └── processed_online_retail_II.csv
├── models/
│   ├── rfm_kmeans_artifacts.joblib
│   ├── clv_model_artifacts.joblib
│   ├── churn_model_artifacts.joblib
│   ├── customer_segments.csv
│   ├── customer_predictions.csv
│   ├── clv_feature_importance.csv
│   ├── churn_feature_importance.csv
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
│   ├── train_models.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## 5. Technology Stack

### Core Language
- Python 3.14+

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

### 6.6 Business Recommendation Layer
Implemented in src/feature_engineering.py and used in src/predict.py

Rules:
1. If churn_probability > 0.7 → Offer Discount
2. If predicted CLV is high → Mark as VIP
3. If cluster label = At Risk → Send Retention Campaign
4. Else fallback → Maintain Engagement

### 6.7 Explainability
Current explainability uses model-native feature importance:
- Tree models: feature_importances_
- Linear models: absolute coefficient magnitude

For single-customer inference:
- Top important feature categories are converted into plain-language reasons
  (for example: recency days, monetary spend, CLV contribution)

---

## 7. Streamlit Dashboard

Implemented in app/streamlit_app.py with 5 pages:

1. Overview Dashboard
  - Revenue KPIs
  - Monthly trend
  - High churn risk percentage

2. Customer Segmentation
  - Cluster distribution
  - RFM scatter views

3. CLV Prediction
  - Input customer ID
  - Predicted CLV
  - CLV feature importance + explanation

4. Churn Prediction
  - Input customer ID
  - Churn probability
  - Churn feature importance + explanation

5. Recommendations
  - Input customer ID
  - Unified output:
    - Cluster Label
    - Predicted CLV
    - Churn Probability
    - Recommended Actions

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
- Add your live URL here after deployment

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

## 13. Current Status

Implemented and connected end-to-end:
1. Data preprocessing
2. RFM clustering
3. CLV regression
4. Churn classification
5. Recommendation engine
6. Streamlit dashboard

This means you can now enter a Customer ID in the dashboard and directly get:
- Cluster label
- Predicted CLV
- Churn probability
- Recommended action

## Branch
 - feature/ml-enhancements-v2
 
