# E-Commerce Customer Analytics Platform

[![CI](https://github.com/orewagaurav/ecommerce-customer-analytics-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/orewagaurav/ecommerce-customer-analytics-ml/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-93%20passing-brightgreen)

A production-style customer intelligence system built on the **Online Retail II**
dataset (1M+ transactions). It segments customers, forecasts 90-day value,
predicts churn, explains every prediction with SHAP, and recommends an action —
served through a **FastAPI** service with a **Streamlit** dashboard on top.

**[Live dashboard](https://ecommerce-customer-analytics-ml-iw5v4ndz7tnjtf3zxweari.streamlit.app)** ·
[Full technical write-up](project/README.md)

---

## Quickstart

```bash
docker compose up --build
```

| | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Health | http://localhost:8000/v1/health |

<details>
<summary>Run without Docker</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# One-time: train, then materialise the feature store + registry
python project/src/train_models.py
python project/src/build_feature_store.py

# Terminal 1 - API
uvicorn src.api.main:app --app-dir project --port 8000

# Terminal 2 - dashboard
streamlit run project/app/streamlit_app.py
```

macOS note: XGBoost needs OpenMP. If `import xgboost` fails with
`Library not loaded: @rpath/libomp.dylib`, run `brew install libomp`.
</details>

---

## Architecture

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

**The core idea:** everything expensive happens offline. A request is a
dictionary lookup plus a model call — never a data scan.

---

## Results

Honest, leakage-free numbers from `project/models/training_report.json`.

| Model | Metric | Value |
|---|---|---|
| **Churn** | ROC-AUC (holdout / 5-fold CV) | **0.813 / 0.788** |
| | Brier score | 0.173 |
| **CLV** | Spearman rank correlation | **0.589** |
| | Top-decile lift | **3.53×** |
| | MAE | 549 |
| **Segmentation** | K (elbow-selected) | 3 |

**Top-decile lift 3.53× means:** the top 10% of customers the model flags go on
to spend 3.5× the average customer. That is the number a marketing team acts on.

R² is reported (0.035) but **not** used for model selection — see
[why](#2-the-metric-was-the-problem-not-just-the-model).

## Performance

| Path | Mean | p95 |
|---|---|---|
| Legacy (re-read 80 MB CSV per call) | 1586.5 ms | 1624.9 ms |
| Feature store + SHAP | **426.6 ms** | 482.7 ms |
| Feature store, no SHAP | **57.8 ms** | 60.5 ms |

**3.7× faster end to end; 27× when explanations aren't needed.**
Inference data: 79.9 MB CSV → 357 KB parquet.

Reproduce with `python project/benchmarks/benchmark.py`.

---

## What to explain in an interview

The models here are ordinary. **The engineering judgement is what's worth
talking about.** Four stories, in descending order of impact.

### 1. I found and fixed target leakage in a stacked model

**The setup.** The churn model takes `PredictedCLV` as a feature. The CLV model
is trained to predict `FutureRevenue` over the same 90-day window the churn
label is derived from. And `FutureRevenue > 0` is *exactly* `ChurnLabel == 0`.

**What went wrong.** The CLV model scored its own training rows *in-sample*. For
those rows it had effectively memorised the answer, so `PredictedCLV` handed the
churn model a copy of its own label.

**How I caught it.** Adding behavioural features pushed churn ROC-AUC to 0.969.
That was implausibly good, so instead of shipping it I measured the suspect
feature directly:

| | AUC |
|---|---|
| `PredictedCLV` **alone** as a churn score | **0.9606** |
| Churn model with `PredictedCLV` removed | **0.8011** |

One feature was carrying essentially the whole model.

**The fix.** Generate `PredictedCLV` with `cross_val_predict`, so every customer
is scored by a CLV model that never saw them. Honest ablation afterwards:

| Churn feature set | ROC-AUC |
|---|---|
| Original | 0.797 |
| + Recency | 0.798 |
| + behavioural features | **0.810** |

**The uncomfortable part — say it out loud.** This leak predated the change. It
had inflated the original 0.795 baseline too, and grew worse as the CLV model
improved. I had earlier reported 0.879 as an improvement. It wasn't. I corrected
it in the README and the PR rather than quietly restating it.

> **Why this lands:** most candidates present their best number. Saying "my
> number was wrong, here's how I proved it and what it actually is" signals
> seniority far more than 0.97 ever would.

### 2. The metric was the problem, not just the model

CLV showed R² = 0.036. The obvious read is "the model is bad." I checked the
data instead:

- **56.6%** of customers have zero future revenue (zero-inflated target)
- Skew **27.96**, dropping to **0.42** under `log1p`
- The single largest holdout customer contributes **81% of the total sum of
  squares**, and is larger than anything in training

So R² and RMSE were mostly measuring how well one unlearnable outlier was fit.
Different models barely moved the number because the number wasn't about them.

**What I changed:** selection moved to **Spearman rank correlation**, reported
with **top-decile lift**. Every downstream consumer of CLV here — VIP flagging,
priority ordering, campaign targeting — uses the *ranking*, not the absolute
currency value. So measure the ranking.

**Expect the pushback:** *"Isn't switching metrics moving the goalposts?"* The
metric must match the decision the model feeds. If we billed customers on
predicted value, RMSE would be correct and the honest conclusion would be that
the model isn't usable. We rank them for outreach, so rank quality is the target.

### 3. A bug no accuracy metric could ever catch

Every per-customer SHAP contribution was **exactly zero**. The dashboard's "Why
this prediction?" panel rendered confidently, showing arbitrary country dummies
as top features.

**Cause:** single-row scoring passed that one row to `shap.Explainer` as both the
sample *and* the background distribution. Explaining a point against itself gives
zero contribution for every feature.

**Why it matters:** accuracy, AUC and Brier were all unaffected. No metric,
dashboard, or test would have flagged it — it was only visible by reading the
output. The fix persists a 200-row background sample per model.

**The lesson:** model quality metrics don't test model *plumbing*. That is why
this project has a test suite and not just a metrics table.

### 4. Debugging discipline: XGBoost that never ran

`requirements.txt` pinned XGBoost, the README advertised it, and
`training_report.json` contained no XGBoost run at all. A
`try/except Exception` around the import was swallowing the failure silently.

The real cause was **not** a version incompatibility — it was a missing macOS
OpenMP runtime (`libomp.dylib`). One `brew install libomp` fixed it.

**The engineering response, not just the fix:** CI now installs `libgomp1` and
runs `python -c "import xgboost"` as an explicit step, so a swallowed
`ImportError` can never silently return.

> **Transferable point:** a bare `except Exception` that degrades silently is
> worse than a crash. The system looked like it was working for months.

### Design decisions worth defending

| Decision | Why |
|---|---|
| Feature store (parquet) instead of live aggregation | Inference re-read 80 MB and re-aggregated 800k rows per request, then used one row. Latency scaled with dataset size — unacceptable on a serving path. |
| `/predict` is `def`, not `async def` | Scoring is CPU-bound (sklearn + SHAP). FastAPI runs sync handlers in a threadpool; `async` would block the event loop for the whole request. |
| JSON registry, not MLflow | No server to run, version-controls with the code, honest about project scale. MLflow when experiment volume justifies it. |
| Split `process` vs `lifetime` metrics | A container restart made `/metrics` report 0 predictions next to a latency average computed from 15 of them. Mixing process counters with persisted aggregates produces self-contradicting payloads. |
| Streamlit talks HTTP, imports no model code | Makes the API/UI split real rather than cosmetic. The dashboard works unchanged if the API moves hosts. |
| Kept the raw-target linear model as a candidate | It's the original baseline. Keeping it in the report anchors every claimed improvement against what shipped first. |

### Questions you should expect

**"Your churn AUC is only 0.81 — why so low?"**
Because it's honest; it was 0.97 with leakage. Predicting whether a retail
customer returns within 90 days from RFM aggregates has a real ceiling, and 0.81
with a tight CV band (±0.005) is a defensible estimate of it.

**"Why didn't the new features help CLV?"**
They didn't — Spearman went 0.588 → 0.589, and I report that plainly. Two rounds
of feature work suggest the RFM-aggregate framing is exhausted; the remaining
signal is product-category affinity and seasonality, which this dataset barely
carries. Knowing when to stop tuning is part of the job.

**"Walk me through a request."**
`POST /v1/predict/{id}` → middleware assigns a request ID and starts a timer →
DI resolves `PredictionService` from the container on `app.state` → O(1) feature
lookup from the parquet store → KMeans assigns a segment → CLV pipeline predicts
→ churn pipeline consumes CLV + segment → SHAP explains both against a persisted
background → rule engine maps outputs to an action → result is appended to the
audit parquet → JSON response with latency and model version.

**"How do you know your tests actually work?"**
Both regression suites were mutation-tested: I reintroduced each bug and
confirmed the test fails. Removing the SHAP background reproduces the original
bug at exactly `0.000000`; reverting `build_churn_dataset` to the full frame
fails with `got 5100.0, expected 100.0`. A test that has never failed hasn't
been shown to work.

**"What would you do next?"**
Product-category and seasonality features for CLV; an uplift model for the
recommendation engine so we target persuadables instead of lost causes; and
moving SHAP off the hot path with a cache, since it's ~85% of an explained
request.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/predict/{customer_id}` | Segment, CLV, churn, SHAP, recommendation |
| `GET /v1/health` | Liveness + readiness (models and store actually loadable) |
| `GET /v1/model-info` | Registry: versions, algorithms, metrics, feature lists |
| `GET /v1/metrics` | Process and lifetime operational counters |
| `POST /v1/simulate/{customer_id}` | What-if rescoring against overridden features |
| `GET /v1/history` | Prediction audit log, filterable |
| `GET /v1/customers` | Scoreable customer IDs |
| `GET /v1/customers/{id}/profile` | Stored features behind a prediction |
| `GET /v1/reports/customer/{id}/pdf` | One-customer PDF briefing |
| `GET /v1/reports/customers/excel` | Multi-sheet Excel workbook |
| `GET /v1/reports/history/excel` | Audit log workbook |
| `GET /docs` | Swagger UI |

```bash
curl -X POST http://localhost:8000/v1/predict/12748 \
     -H 'Content-Type: application/json' \
     -d '{"include_explanations": true}'
```

Pass `{"include_explanations": false}` for a 58 ms response instead of 427 ms.

Errors use one envelope (`error`, `detail`, `request_id`, `context`), and every
response carries `X-Request-ID` and `X-Response-Time-Ms`.

---

## Dashboard

Nine pages. The dashboard holds no model code — it reads precomputed aggregates
and calls the API over HTTP.

| Page | Contents |
|---|---|
| Executive | Revenue and customer KPIs, revenue trend, country split, segment mix, churn distribution, top customers, recommended focus, revenue at risk |
| Overview | Headline KPIs and monthly revenue |
| Customer 360 | Profile, monthly spend and orders, top products, segment, CLV, churn, SHAP, recommended action, PDF/CSV export |
| Segmentation | Cluster distribution and RFM scatter |
| CLV / Churn / Recommendations | Per-customer scoring with SHAP panels |
| What-if Simulator | Adjust Recency / Frequency / Monetary and rescore against a baseline |
| Prediction History | Filterable audit log with CSV and Excel export |

## Reports

Generated server-side and streamed, so any consumer — the dashboard, a scheduled
job, a CRM integration — gets the same artefact from the same endpoint.

- **PDF** per-customer briefing: model output, behavioural profile, SHAP drivers
  for both tasks, top products
- **Excel** workbook: summary, customer features, monthly and country revenue,
  top products; plus an audit-log workbook with a per-segment sheet
- **CSV** from the dashboard for profiles, scored customers and history

## Testing

```bash
pytest                 # 93 tests
pytest --cov=src       # with coverage
```

| Suite | Protects |
|---|---|
| `test_leakage.py` | Features come from history before the cutoff; labels from after |
| `test_out_of_fold.py` | `PredictedCLV` cannot memorise the churn label |
| `test_explainability.py` | SHAP contributions are non-zero for single-customer scoring |
| `test_api.py` | Full API contract through `TestClient` |
| `test_train_models.py` | Selection metrics + two-stage estimator |
| `test_feature_engineering.py` | RFM arithmetic, elbow, segment labelling |
| `test_behavioural_features.py` | Tenure, inter-purchase timing, momentum |
| `test_data_preprocessing.py` | Returns, nulls, unparseable dates removed |
| `test_recommendation_engine.py` | Rule precedence and reachability |
| `test_predict_integration.py` | End-to-end scoring against committed artifacts |

CI runs lint, the feature-store build, and the full suite on Python 3.11 and 3.12.

---

## Project structure

```text
project/
├── app/
│   ├── streamlit_app.py        # dashboard (HTTP client only, no model code)
│   └── api_client.py           # typed wrapper over the scoring API
├── src/
│   ├── api/                    # FastAPI: routes, schemas, DI, error handlers
│   ├── services/               # PredictionService, PredictionHistoryService
│   ├── config.py               # Pydantic Settings (env-driven)
│   ├── logging_config.py       # structured JSON logging
│   ├── feature_store.py        # offline build + lazy O(1) lookup
│   ├── analytics_store.py      # precomputed dashboard aggregates
│   ├── registry.py             # model registry
│   ├── feature_engineering.py  # RFM + behavioural features
│   ├── train_models.py         # training orchestration
│   └── explainability.py       # SHAP
├── tests/                      # 93 tests
├── benchmarks/                 # latency + memory harness
└── feature_store/              # generated parquet artifacts
```

## Stack

Python 3.11 · scikit-learn · XGBoost · SHAP · pandas · FastAPI · Pydantic ·
Streamlit · Plotly · Docker · pytest · GitHub Actions

---

## Honest limitations

- CLV ranking (Spearman 0.589) is near this feature set's ceiling; more tuning
  won't move it.
- SHAP is ~85% of an explained request. The feature store moved the bottleneck
  rather than removing it.
- The 80 MB processed CSV is still committed, which makes CI checkout slow. It
  should move to a download step.
- Returns and cancellations are dropped in preprocessing, so `Monetary` is gross
  rather than net revenue.

See [project/README.md](project/README.md) for the full technical write-up.
