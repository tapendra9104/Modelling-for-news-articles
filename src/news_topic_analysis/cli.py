from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .database import DatabaseArtifactStore, DatabaseConfig
from .pipeline import NewsTopicPipeline, PipelineConfig
from .storage import ArtifactStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-topics",
        description="Run AI-based topic modeling and trend analysis over news articles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Collect articles and generate analysis artifacts.")
    run_parser.add_argument("--source", choices=["demo", "csv", "rss", "html", "mixed"], default="demo")
    run_parser.add_argument("--model", choices=["lda", "nmf", "bertopic"], default="lda")
    run_parser.add_argument("--num-topics", type=int, default=6)
    run_parser.add_argument("--top-words", type=int, default=8)
    run_parser.add_argument("--max-features", type=int, default=3000)
    run_parser.add_argument("--limit-per-source", type=int, default=20)
    run_parser.add_argument("--top-k", type=int, default=3)
    run_parser.add_argument("--artifact-dir", default="artifacts/latest")
    run_parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to a local CSV dataset. Used when --source csv.",
    )
    run_parser.add_argument("--spacy", action="store_true", help="Use spaCy lemmatization if available.")
    run_parser.add_argument("--nltk", action="store_true", help="Use NLTK lemmatization if spaCy is disabled.")
    run_parser.add_argument(
        "--db-backend",
        choices=["mongodb", "mysql"],
        default=None,
        help="Optional database backend for artifact persistence.",
    )
    run_parser.add_argument(
        "--db-uri",
        default=None,
        help="Database connection URI. If omitted, NEWS_TOPIC_DB_URI is used when present.",
    )
    run_parser.add_argument(
        "--db-name",
        default=None,
        help="Database name for MongoDB or fallback metadata label when the URI does not include one.",
    )
    run_parser.add_argument(
        "--db-prefix",
        default=None,
        help="Collection/table prefix for persisted artifacts.",
    )

    summary_parser = subparsers.add_parser("summary", help="Show a saved run summary.")
    summary_parser.add_argument("--artifact-dir", default="artifacts/latest")
    summary_parser.add_argument(
        "--storage-mode",
        choices=["artifacts", "database"],
        default="artifacts",
        help="Load the summary from local artifacts or from a configured database.",
    )
    summary_parser.add_argument("--db-backend", choices=["mongodb", "mysql"], default=None)
    summary_parser.add_argument("--db-uri", default=None)
    summary_parser.add_argument("--db-name", default=None)
    summary_parser.add_argument("--db-prefix", default=None)
    summary_parser.add_argument("--run-id", default=None)
    return parser


def _run_pipeline(arguments: argparse.Namespace) -> int:
    db_backend = arguments.db_backend or os.getenv("NEWS_TOPIC_DB_BACKEND")
    db_uri = arguments.db_uri or os.getenv("NEWS_TOPIC_DB_URI")
    db_name = arguments.db_name or os.getenv("NEWS_TOPIC_DB_NAME", "news_topic_analysis")
    db_prefix = arguments.db_prefix or os.getenv("NEWS_TOPIC_DB_PREFIX", "news_topic_analysis")

    config = PipelineConfig(
        data_source=arguments.source,
        model_name=arguments.model,
        num_topics=arguments.num_topics,
        top_words=arguments.top_words,
        max_features=arguments.max_features,
        limit_per_source=arguments.limit_per_source,
        top_k_recommendations=arguments.top_k,
        enable_spacy=arguments.spacy,
        enable_nltk=arguments.nltk,
        csv_path=arguments.csv_path,
        db_backend=db_backend,
        db_uri=db_uri,
        db_name=db_name,
        db_prefix=db_prefix,
        artifact_dir=Path(arguments.artifact_dir),
    )

    result = NewsTopicPipeline(config).run()
    print(f"Artifacts written to: {result.artifact_dir.resolve()}")
    print()
    print("Top topics")
    print(result.topic_info.to_string(index=False))
    print()
    collection_summary = result.metadata.get("collection_summary", {})
    if collection_summary:
        print("Collection summary")
        print(json.dumps(collection_summary, indent=2))
        print()
    evaluation_summary = result.metadata.get("evaluation_summary", {})
    if evaluation_summary:
        print("Evaluation summary")
        print(json.dumps(evaluation_summary, indent=2))
        print()
    if result.metadata.get("database", {}).get("enabled"):
        print("Database persistence")
        print(json.dumps(result.metadata["database"], indent=2))
        print()
    print(f"Dashboard: streamlit run streamlit_app.py -- --artifact-dir {result.artifact_dir}")
    return 0


def _show_summary(arguments: argparse.Namespace) -> int:
    if arguments.storage_mode == "database":
        db_backend = arguments.db_backend or os.getenv("NEWS_TOPIC_DB_BACKEND")
        db_uri = arguments.db_uri or os.getenv("NEWS_TOPIC_DB_URI")
        db_name = arguments.db_name or os.getenv("NEWS_TOPIC_DB_NAME", "news_topic_analysis")
        db_prefix = arguments.db_prefix or os.getenv("NEWS_TOPIC_DB_PREFIX", "news_topic_analysis")
        if not db_backend or not db_uri:
            raise SystemExit("Database summary mode requires --db-backend and --db-uri, or matching env vars.")
        payload = DatabaseArtifactStore(
            DatabaseConfig(
                backend=db_backend,
                uri=db_uri,
                name=db_name,
                prefix=db_prefix,
            )
        ).load(run_id=arguments.run_id)
    else:
        payload = ArtifactStore.load(arguments.artifact_dir)
    print(json.dumps(payload["metadata"], indent=2))
    if not payload["topic_info"].empty:
        print()
        print("Saved topics")
        print(payload["topic_info"].to_string(index=False))
    evaluation_summary = payload.get("evaluation_summary", pd.DataFrame())
    if isinstance(evaluation_summary, pd.DataFrame) and not evaluation_summary.empty:
        print()
        print("Saved evaluation summary")
        print(evaluation_summary.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "run":
        return _run_pipeline(arguments)
    if arguments.command == "summary":
        return _show_summary(arguments)
    parser.error(f"Unsupported command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
