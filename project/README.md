# E-Commerce Customer Analytics Platform

End-to-end customer intelligence pipeline built on the Online Retail II dataset.

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

Training and feature materialisation run offline. The request path does no data
scanning: it is a keyed lookup plus a model call.

```text
              OFFLINE (batch)                          ONLINE (request path)
 ┌────────────────────────────────────────┐     ┌──────────────────────────────┐
 │ raw CSV (90 MB)                        │     │ Streamlit dashboard          │
 │     ↓ data_preprocessing.py            │     │     │ HTTP                   │
 │ processed transactions                 │     │     ↓                        │
 │     ↓ feature_engineering.py           │     │ FastAPI /v1/predict/{id}     │
 │ RFM + behavioural features             │     │         /v1/health           │
 │     ↓ train_models.py                  │     │         /v1/model-info       │
 │ ├── KMeans segmentation                │     │         /v1/metrics          │
 │ ├── CLV regression   (log1p / 2-stage) │     │     ↓                        │
 │ └── Churn classifier (OOF CLV feature) │     │ PredictionService            │
 │     ↓ build_feature_store.py           │     │     ↓                        │
 │ features.parquet (357 KB) ─────────────┼────▶│ FeatureStore   O(1) lookup   │
 │ dashboard aggregates                   │     │ model artifacts (cached)     │
 │ model_registry.json       ─────────────┼────▶│ SHAP explainer               │
 └────────────────────────────────────────┘     │     ↓                        │
                                                │ prediction history (audit)   │
                                                └──────────────────────────────┘
```

### 3.1 Layering

| Layer | Modules | Responsibility |
|---|---|---|
| API | `src/api/` | HTTP contract, validation, DI, error translation |
| Service | `src/services/` | Scoring orchestration, prediction audit log |
| Data access | `feature_store.py`, `analytics_store.py`, `registry.py` | Materialised reads |
| Domain | `feature_engineering.py`, `recommendation_engine.py`, `explainability.py` | Business logic |
| Training | `train_models.py`, `build_feature_store.py` | Offline batch |

Dependencies point inward: API depends on services, services on data access and
domain, and the domain layer imports nothing from the API. Collaborators are
injected through a container on `app.state`, so no module reaches for a
singleton and every layer is testable in isolation.

---

## 4. Project Structure

```text
project/
├── app/
│   ├── streamlit_app.py        # dashboard (HTTP client only, no model code)
│   └── api_client.py           # typed wrapper over the scoring API
├── src/
│   ├── api/
│   │   ├── main.py             # application factory, middleware, lifespan
│   │   ├── routes.py           # versioned endpoints
│   │   ├── schemas.py          # Pydantic request/response contract
│   │   ├── dependencies.py     # DI container
│   │   └── errors.py           # domain exception -> HTTP translation
│   ├── services/
│   │   ├── prediction_service.py
│   │   └── history_service.py
│   ├── config.py               # Pydantic Settings
│   ├── logging_config.py       # structured JSON logging
│   ├── feature_store.py
│   ├── analytics_store.py
│   ├── registry.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── explainability.py
│   ├── recommendation_engine.py
│   ├── train_models.py
│   ├── build_feature_store.py
│   └── predict.py              # retained CLI entry point
├── tests/                      # 93 tests
├── benchmarks/
├── models/                     # artifacts + training_report + registry
├── feature_store/              # generated parquet
└── notebooks/
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
- Recency, Frequency, Monetary
- AverageBasketSize, PurchaseFrequency
- Tenure (days since first purchase)
- AvgInterPurchaseDays (mean gap between purchase occasions)
- DistinctProducts (unique stock codes bought)
- AvgItemsPerInvoice
- RecentRevenueShare (share of spend in the trailing 90 days of history)
- Country

Every feature is computed from history up to the cutoff date only.

Additional feature datasets:
- CLV dataset with FutureRevenue target (next 90 days window)
- Churn dataset with ChurnLabel set from the post-cutoff purchase window

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
- The full behavioural feature set from 6.2
- Country (one-hot encoded)

Models trained:
- Linear Regression on the raw target (retained as the original baseline)
- Random Forest and XGBoost on a log1p target
- Two-stage zero-inflated model: P(purchase) x E[spend | purchase]

Metrics:
- Spearman rank correlation, top-decile lift (selection)
- MAE, RMSE, R2, 5-fold CV R2 (reported)

Best model selection criterion:
- Maximum Spearman correlation (see 12.5 for why not RMSE)

Saved outputs:
- clv_model_artifacts.joblib
- clv_feature_importance.csv

### 6.5 Module 3: Churn Classification
Label definition:
- ChurnLabel = 1 if the customer made no purchase in the `threshold_days`
  window after the cutoff date (default 90 days)

Features:
- The full behavioural feature set from 6.2
- PredictedCLV, generated out-of-fold via `cross_val_predict` so it cannot
  carry a memorised copy of the churn label (see 12.5)
- ClusterLabel

Models trained:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier (if available)

Metrics:
- Accuracy, Precision, Recall, F1
- ROC-AUC, 5-fold CV ROC-AUC
- Brier score (calibration)

Best model selection criterion:
- Maximum ROC-AUC, returned by the trainer so the recorded name cannot diverge
  from the pipeline actually saved

Saved outputs:
- churn_model_artifacts.joblib
- churn_feature_importance.csv

### 6.6 Business Recommendation Layer
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

### 6.7 Explainability
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

## 7. Streamlit Dashboard

Nine pages. The dashboard holds no model code: it reads precomputed aggregates
and calls the API over HTTP.

| Page | Contents |
|---|---|
| Executive Dashboard | Revenue and customer KPIs, revenue trend, revenue by country, segment mix, churn distribution, top customers, recommended focus, revenue at risk |
| Overview | Headline KPIs and monthly revenue |
| Customer 360 | Profile, monthly spend and orders, top products, segment, CLV, churn, SHAP explanation, recommended action |
| Segmentation | Cluster distribution and RFM scatter |
| CLV Prediction | Per-customer CLV with SHAP panel |
| Churn Prediction | Per-customer churn with risk gauge and SHAP panel |
| Recommendations | Unified decision output |
| What-if Simulator | Adjust Recency / Frequency / Monetary and rescore against a baseline |
| Prediction History | Filterable audit log with CSV export |

The sidebar shows API connection state and the deployed model version, so a
broken backend is visible immediately rather than as a failed page.

**What-if is read-only.** Simulations are hypotheticals, not predictions the
system made, so they are never written to the audit log — a test pins this.

### 7.1 Output Contract

The dashboard consumes the API response, not a Python dict. The authoritative
contract is `src/api/schemas.py`, rendered at `/docs`:

| Field | Meaning |
|---|---|
| `customer_id` | Scored customer |
| `cluster_label` | RFM segment |
| `predicted_clv` | Expected revenue, next 90 days |
| `churn_probability` | 0-1 |
| `decision` | `customer_segment`, `priority_level`, `recommended_action` |
| `recommendation_actions` | Rule-engine action list |
| `explanations` | Human-readable CLV and churn reasons |
| `shap_top_features` | Top contributors per task, with sign |
| `model_version` | Registry version that produced this |
| `latency_ms` | Server-side scoring time |

---

## 8. How to Run Locally

Run from the repository root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

### Step 1: Preprocess

```bash
python project/src/data_preprocessing.py \
  --input_csv project/data/online_retail_II.csv \
  --output_csv project/data/processed_online_retail_II.csv
```

### Step 2: Train

```bash
python project/src/train_models.py \
  --processed_csv project/data/processed_online_retail_II.csv \
  --models_dir project/models \
  --horizon_days 90 --churn_days 90
```

### Step 3: Materialise the feature store and registry

Required before the API will serve. Builds `features.parquet`, the dashboard
aggregates, and `model_registry.json`.

```bash
python project/src/build_feature_store.py
```

### Step 4: Run the service and dashboard

```bash
# Terminal 1
uvicorn src.api.main:app --app-dir project --port 8000

# Terminal 2
streamlit run project/app/streamlit_app.py
```

API docs at http://localhost:8000/docs, dashboard at http://localhost:8501.

---

## 9. Docker

Both services run from one image: the FastAPI scoring service and the Streamlit
dashboard that consumes it over HTTP.

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API docs (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/v1/health |
| Dashboard | http://localhost:8501 |

The dashboard waits on the API healthcheck before starting, and prediction
history is persisted to a named volume so the audit log survives restarts.

**Verified on Docker 29.6.2 / Compose v5.3.1 (arm64):** image builds, both
containers reach healthy, predictions serve end to end, XGBoost imports
(`libgomp1` present), the container runs as non-root `appuser` (uid 1000), and
history survives `docker compose restart`.

The image deliberately excludes `project/data`. Inference reads the feature
store, so the 80 MB CSV never ships.

---

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
  - Branch: main (or your default branch)
  - App file: project/app/streamlit_app.py
4. Add project/data/online_retail_II.csv to repo or configure remote data access
5. Deploy

Deployment link:
- https://ecommerce-customer-analytics-ml-iw5v4ndz7tnjtf3zxweari.streamlit.app

---

## 12. Performance

Measured with `python project/benchmarks/benchmark.py`, 60 runs per path.

| Path | Mean | Median | p95 |
|---|---|---|---|
| Legacy (re-read CSV per call) | 1586.5 ms | 1574.4 ms | 1624.9 ms |
| Feature store + SHAP | 426.6 ms | 421.0 ms | 482.7 ms |
| Feature store, no SHAP | 57.8 ms | 57.9 ms | 60.5 ms |

**3.7x faster end to end, or 27x with `include_explanations: false`.**

| | Before | After |
|---|---|---|
| Inference data file | 79.9 MB CSV | 357 KB parquet |
| Feature lookup | full re-aggregation of ~800k rows | O(1) dict lookup |

Inside Docker on macOS, steady state is ~550 ms with SHAP and ~75 ms without —
roughly 25% above native, which is Docker Desktop VM overhead. The first request
after start is slower (~1 s) while caches warm.

**Honest reading of these numbers.** The feature store removed data loading from
the request path entirely; what remains is SHAP, which is now ~85% of a fully
explained request. The 3.7x figure is the honest end-to-end number, and the 27x
figure is what the same service does when a caller does not need explanations.

---

## 13. Tests

```bash
pip install -r requirements-dev.txt
pytest                          # 60 tests
pytest --cov=src                # with coverage
```

Run from the repository root. CI runs the same command on Python 3.11 and 3.12
(`.github/workflows/ci.yml`).

| File | What it protects |
|---|---|
| `test_leakage.py` | The temporal boundary: features come from history before the cutoff, labels from the window after it |
| `test_explainability.py` | SHAP contributions are non-zero for single-customer scoring |
| `test_train_models.py` | Selection metrics and the zero-inflated two-stage estimator |
| `test_feature_engineering.py` | RFM arithmetic, elbow selection, segment labelling |
| `test_data_preprocessing.py` | Returns, nulls and unparseable dates are removed |
| `test_recommendation_engine.py` | Rule precedence, and that no rule is unreachable |
| `test_predict_integration.py` | End-to-end scoring against the committed artifacts |

Two tests exist because the corresponding bugs actually shipped, and neither
would have been caught by an accuracy metric:

- **Per-customer SHAP returned all zeros.** Single-row scoring passed that row
  as its own SHAP background, so every contribution cancelled to zero while the
  dashboard kept rendering a confident-looking explanation panel.
- **Churn features could silently absorb the label window.** A one-line change
  from `history` to the full frame reintroduces it; `test_leakage.py` fails with
  the observed and expected spend rather than a generic assertion error.

---

## 14. Results

Numbers below come from `project/models/training_report.json`, regenerated by
`train_models.py`. Holdout is a 20% split; CV is 5-fold on the training split.

### Churn classification (best: RandomForestClassifier)

| Metric | Value |
|---|---|
| ROC-AUC (holdout) | 0.813 |
| ROC-AUC (5-fold CV) | 0.788 |
| Brier score | 0.173 |

**These numbers are lower than an earlier revision of this README claimed, and
the earlier ones were wrong.** The churn model takes `PredictedCLV` as a
feature. The CLV model is trained to predict revenue in the same window the
churn label is derived from, and `FutureRevenue > 0` is exactly
`ChurnLabel == 0`. Scoring the CLV training rows in-sample therefore fed the
churn model a memorised copy of its own label.

Measured directly: `PredictedCLV` **alone** scored churn at **AUC 0.96**, while
a model built from history features without it reached 0.80. The reported
0.879, and a later 0.969, were both leakage, not skill.

`PredictedCLV` is now generated with `cross_val_predict`, so every customer is
scored by a CLV model that never saw them. Ablation under that honest setup,
varying only the feature set:

| Churn feature set | ROC-AUC |
|---|---|
| Original (Frequency, Monetary, PredictedCLV, Cluster) | 0.797 |
| + Recency | 0.798 |
| + behavioural features | **0.810** |

Two things follow. Adding `Recency` contributed almost nothing once the leak was
removed — the earlier claim that it drove the gain was an artifact of the leak
growing as the CLV model improved. The behavioural features are a real but
modest gain of roughly +0.013 AUC.

### CLV regression (best: RandomForest on log1p target)

| Metric | Value |
|---|---|
| Spearman rank correlation | 0.589 |
| Top-decile lift | 3.53x |
| MAE | 549 |
| RMSE | 5622 |
| R2 (holdout) | 0.035 |

**Why R2 is reported but not used for model selection.** Future 90-day revenue
is extreme-tailed: the single largest customer in the holdout set contributes
**81% of its total sum of squares**, and is larger than anything in training. R2
and RMSE therefore rank models mostly by how they happened to fit one
unlearnable point, and they barely move across very different models. Since
every downstream consumer of CLV here (VIP flagging, priority ordering, campaign
targeting) uses the *ranking*, models are selected on Spearman correlation and
reported with top-decile lift.

Read the lift figure as: **the top 10% of customers this model flags go on to
spend 3.5x the average customer.**

**Honest limitation.** Adding tenure, inter-purchase timing, product diversity
and revenue momentum did **not** improve CLV ranking (Spearman 0.588 before,
0.589 after). All candidate models still land within Spearman 0.48-0.59. The
ceiling is not the algorithm and, on this evidence, not these features either.
What is left to try is information this dataset barely carries: product-category
affinity, price sensitivity, and seasonality across more than two years of
history.

---

## 15. Troubleshooting

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

## 16. Current Status

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
