#!/bin/sh
set -eu

ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/latest}"
SOURCE_MODE="${SOURCE_MODE:-demo}"
MODEL_NAME="${MODEL_NAME:-lda}"
NUM_TOPICS="${NUM_TOPICS:-6}"
PORT="${STREAMLIT_SERVER_PORT:-8501}"
DASHBOARD_STORAGE_MODE="${DASHBOARD_STORAGE_MODE:-artifacts}"
CSV_PATH="${CSV_PATH:-data/raw/news_articles_extended.csv}"

mkdir -p "${ARTIFACT_DIR}"
mkdir -p "artifacts/mplconfig"

if [ ! -f "${ARTIFACT_DIR}/articles.csv" ]; then
  set -- \
    --source "${SOURCE_MODE}" \
    --model "${MODEL_NAME}" \
    --num-topics "${NUM_TOPICS}" \
    --artifact-dir "${ARTIFACT_DIR}"

  if [ "${SOURCE_MODE}" = "csv" ]; then
    set -- "$@" --csv-path "${CSV_PATH}"
  fi

  if [ -n "${NEWS_TOPIC_DB_BACKEND:-}" ] && [ -n "${NEWS_TOPIC_DB_URI:-}" ]; then
    set -- "$@" \
      --db-backend "${NEWS_TOPIC_DB_BACKEND}" \
      --db-uri "${NEWS_TOPIC_DB_URI}" \
      --db-name "${NEWS_TOPIC_DB_NAME:-news_topic_analysis}" \
      --db-prefix "${NEWS_TOPIC_DB_PREFIX:-news_topic_analysis}"

    python -m news_topic_analysis run "$@"
  else
    python -m news_topic_analysis run "$@"
  fi
fi

export DASHBOARD_STORAGE_MODE
export ARTIFACT_DIR

exec streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT}" \
  -- \
  --artifact-dir "${ARTIFACT_DIR}"
