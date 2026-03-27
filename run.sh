#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

VENV_PY="$ROOT_DIR/.venv/bin/python"
VENV_PIP="$ROOT_DIR/.venv/bin/pip"
VENV_STREAMLIT="$ROOT_DIR/.venv/bin/streamlit"

RAW_CSV="$ROOT_DIR/project/data/online_retail_II.csv"
PROCESSED_CSV="$ROOT_DIR/project/data/processed_online_retail_II.csv"
MODELS_DIR="$ROOT_DIR/project/models"

RFM_MODEL="$MODELS_DIR/rfm_kmeans_artifacts.joblib"
CLV_MODEL="$MODELS_DIR/clv_model_artifacts.joblib"
CHURN_MODEL="$MODELS_DIR/churn_model_artifacts.joblib"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[INFO] Creating virtual environment..."
  /opt/homebrew/bin/python3 -m venv .venv
fi

echo "[INFO] Installing/updating dependencies..."
"$VENV_PIP" install -r "$ROOT_DIR/project/requirements.txt"

if [[ ! -f "$RAW_CSV" ]]; then
  echo "[ERROR] Raw dataset not found at: $RAW_CSV"
  echo "        Please place Online Retail II CSV at project/data/online_retail_II.csv"
  exit 1
fi

if [[ ! -f "$PROCESSED_CSV" ]]; then
  echo "[INFO] Processed dataset missing. Running preprocessing..."
  "$VENV_PY" "$ROOT_DIR/project/src/data_preprocessing.py" \
    --input_csv "$RAW_CSV" \
    --output_csv "$PROCESSED_CSV"
fi

if [[ ! -f "$RFM_MODEL" || ! -f "$CLV_MODEL" || ! -f "$CHURN_MODEL" ]]; then
  echo "[INFO] Model artifacts missing. Training models..."
  "$VENV_PY" "$ROOT_DIR/project/src/train_models.py" \
    --processed_csv "$PROCESSED_CSV" \
    --models_dir "$MODELS_DIR" \
    --horizon_days 90 \
    --churn_days 90
fi

echo "[INFO] Launching Streamlit app..."
exec "$VENV_STREAMLIT" run "$ROOT_DIR/project/app/streamlit_app.py"
