# AI-Based Topic Modeling and Trend Analysis for News Articles

This repository turns the final-year-project brief into a runnable Python application. It ingests news articles, preprocesses text, extracts topics with multiple algorithms, tracks topic momentum over time, recommends related stories, and exposes the results through a Streamlit dashboard.

## What the project includes

- News collection through demo data, RSS feeds, or a generic HTML scraper
- Bundled offline CSV datasets for demo, full analysis, and evaluation
- NLP preprocessing with stopword removal and optional lemmatization via spaCy or NLTK
- Topic modeling with LDA, NMF, and an optional BERTopic backend
- Trend detection for topic popularity and emerging themes
- Rule-based domain categorization for major news domains
- Similar-article recommendations based on content similarity
- Automatic evaluation metrics using `expected_domain` labels from bundled datasets
- Automatically generated dashboard images for project overview and technology stack
- Optional persistence to MongoDB or MySQL
- Collector diagnostics for live-ingestion success and failure visibility
- Interactive dashboard built with Streamlit and Plotly

## Architecture

```text
Input Sources
    -> News Collection
    -> Text Preprocessing
    -> Topic Modeling
    -> Trend Analysis
    -> Recommendation Engine
    -> Visualization Dashboard
```

## Project structure

```text
src/news_topic_analysis/
    analytics.py
    categorization.py
    cli.py
    collectors.py
    database.py
    dashboard.py
    models.py
    pipeline.py
    presentation_assets.py
    preprocessing.py
    sample_data.py
    storage.py
    topic_modeling.py
streamlit_app.py
data/
    dataset_manifest.json
    raw/
        news_articles_demo.csv
        news_articles_extended.csv
        news_articles_evaluation.csv
tests/test_pipeline.py
```

## Quick start

1. Create a virtual environment.
2. Install the package.
3. Run the pipeline.
4. Open the dashboard.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
news-topics run --source demo --model lda --artifact-dir artifacts\latest
streamlit run streamlit_app.py -- --artifact-dir artifacts\latest
```

Install database support when you want MongoDB or MySQL persistence:

```powershell
pip install -e .[database]
```

## Example commands

Run with NMF:

```powershell
news-topics run --source demo --model nmf --num-topics 6 --artifact-dir artifacts\nmf_demo
```

Run with the bundled extended local dataset:

```powershell
news-topics run --source csv --csv-path data/raw/news_articles_extended.csv --model lda --artifact-dir artifacts\extended_local
```

Try live collection with RSS and HTML scraping:

```powershell
news-topics run --source mixed --model lda --limit-per-source 15 --artifact-dir artifacts\live_run
```

Persist a run to MySQL:

```powershell
news-topics run --source demo --model lda --artifact-dir artifacts\mysql_run `
  --db-backend mysql `
  --db-uri "mysql+pymysql://newsapp:newsapp123@127.0.0.1:3306/news_topic_analysis" `
  --db-prefix news_topic_analysis
```

Persist a run to MongoDB:

```powershell
news-topics run --source demo --model lda --artifact-dir artifacts\mongo_run `
  --db-backend mongodb `
  --db-uri "mongodb://admin:admin123@127.0.0.1:27017/news_topic_analysis?authSource=admin" `
  --db-name news_topic_analysis `
  --db-prefix news_topic_analysis
```

Use optional NLP extras:

```powershell
pip install -e .[spacy,dev]
python -m spacy download en_core_web_sm
news-topics run --source demo --model lda --spacy
```

Use optional BERTopic:

```powershell
pip install -e .[bertopic]
news-topics run --source demo --model bertopic --artifact-dir artifacts\bertopic_demo
```

Load a summary directly from SQL or MongoDB:

```powershell
news-topics summary --storage-mode database `
  --db-backend mysql `
  --db-uri "mysql+pymysql://newsapp:newsapp123@127.0.0.1:3306/news_topic_analysis"
```

Run the evaluation dataset:

```powershell
news-topics run --source csv --csv-path data/raw/news_articles_evaluation.csv --artifact-dir artifacts\evaluation
```

## Outputs

Every pipeline run writes:

- `articles.csv`: cleaned articles with assigned topics and cluster coordinates
- `topic_info.csv`: topic labels, keywords, and sizes
- `trends.csv`: topic counts by date
- `emerging_topics.csv`: momentum analysis
- `topic_relationships.csv`: topic similarity scores
- `recommendations.csv`: top related articles
- `metadata.json`: run configuration and summary

When labeled datasets are used, the run also writes:

- `evaluation_summary.csv`: accuracy, macro F1, weighted F1, and labeled sample counts
- `domain_performance.csv`: precision, recall, F1, and support per domain
- `domain_confusion.csv`: expected-vs-predicted domain counts
- `split_performance.csv`: performance by dataset split
- `presentation_metrics.csv`: presentation-ready headline metrics
- `presentation_report.md`: final-year-project presentation summary
- `images/*_project_overview.png`: auto-generated infographic for project purpose and workflow
- `images/*_technology_stack.png`: auto-generated infographic for the technology stack

## Bundled datasets

The repository now includes all core offline datasets needed to run the project without external APIs:

- `data/raw/news_articles_demo.csv`
  - 24 articles
  - Default offline demo dataset
- `data/raw/news_articles_extended.csv`
  - 96 articles
  - Main local dataset for topic modeling, trend analysis, clustering, and recommendations
- `data/raw/news_articles_evaluation.csv`
  - 12 articles
  - Holdout dataset for manual evaluation and presentation use

Dataset details are tracked in `data/README.md` and `data/dataset_manifest.json`.

When database persistence is enabled, the pipeline also writes the same run into:

- MongoDB collections prefixed with `news_topic_analysis_`
- MySQL tables prefixed with `news_topic_analysis_`

## Deployment

This repository includes:

- `Dockerfile` for the dashboard container
- `docker-compose.yml` with Streamlit, MongoDB, and MySQL services
- `.streamlit/config.toml` for container-friendly Streamlit defaults
- `.env.example` with environment variables for live ingestion and database connections

Quick Docker workflow:

```powershell
copy .env.example .env
docker compose up --build
```

The dashboard container auto-generates artifacts on first boot using the configured `SOURCE_MODE`, can optionally persist the run to MongoDB or MySQL, and then serves Streamlit on port `8501`.

To make the dashboard read directly from the database instead of CSV artifacts, set:

```text
DASHBOARD_STORAGE_MODE=database
NEWS_TOPIC_DB_BACKEND=mysql
NEWS_TOPIC_DB_URI=mysql+pymysql://newsapp:newsapp123@mysql:3306/news_topic_analysis
```

To run the bundled offline dataset inside Docker:

```text
SOURCE_MODE=csv
CSV_PATH=data/raw/news_articles_extended.csv
```

## Notes

- `demo` mode is fully offline and intended for project demos, viva presentations, and testing.
- `mixed` mode combines RSS ingestion with a generic HTML scraper. Some sites may need selector tuning because publisher layouts change.
- RSS collection works well for BBC and CNN in live mode using HTTPS feeds.
- Reuters HTML scraping currently returns `401 Forbidden` in this environment, so `mixed` mode falls back to the RSS collectors unless you provide a Reuters-compatible source or licensed API.
- Run metadata now includes per-source collector diagnostics, so partial ingestion failures are visible in the dashboard and saved summaries.
- The dashboard now includes an evaluation section with confusion charts, split performance, and presentation-ready reporting when labeled datasets are used.
- The dashboard now includes a built-in project overview section that explains the project use cases, workflow, technology stack, and shows the generated infographic images automatically.
