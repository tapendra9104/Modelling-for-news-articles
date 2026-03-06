# Project Datasets

This folder contains the bundled datasets required to run the project offline.

## Files

- `raw/news_articles_demo.csv`
  - 24 curated articles
  - Small offline dataset used by the default `demo` source
- `raw/news_articles_extended.csv`
  - 96 curated articles
  - Main bundled dataset for full local topic-modeling and trend-analysis runs
- `raw/news_articles_evaluation.csv`
  - 12 holdout articles
  - Useful for manual evaluation, demo comparisons, and domain-label checks
- `dataset_manifest.json`
  - Dataset inventory with names, paths, row counts, and purpose

## Schema

Each CSV contains:

- `article_id`
- `title`
- `content`
- `source`
- `published_at`
- `url`
- `language`
- `expected_domain`
- `split`

## Recommended usage

Run the larger offline dataset:

```powershell
python -m news_topic_analysis run --source csv --csv-path data/raw/news_articles_extended.csv --artifact-dir artifacts/local_extended
```

Use the small demo dataset:

```powershell
python -m news_topic_analysis run --source demo --artifact-dir artifacts/local_demo
```
