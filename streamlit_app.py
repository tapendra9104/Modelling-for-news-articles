from __future__ import annotations

import argparse
import os

from news_topic_analysis.dashboard import run_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifact-dir", default=os.getenv("ARTIFACT_DIR", "artifacts/latest"))
    parser.add_argument(
        "--storage-mode",
        choices=["artifacts", "database"],
        default=os.getenv("DASHBOARD_STORAGE_MODE", "artifacts"),
    )
    parser.add_argument("--db-backend", default=os.getenv("NEWS_TOPIC_DB_BACKEND"))
    parser.add_argument("--db-uri", default=os.getenv("NEWS_TOPIC_DB_URI"))
    parser.add_argument("--db-name", default=os.getenv("NEWS_TOPIC_DB_NAME", "news_topic_analysis"))
    parser.add_argument("--db-prefix", default=os.getenv("NEWS_TOPIC_DB_PREFIX", "news_topic_analysis"))
    parser.add_argument("--run-id", default=os.getenv("DASHBOARD_RUN_ID"))
    arguments, _ = parser.parse_known_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    run_dashboard(
        default_artifact_dir=arguments.artifact_dir,
        default_storage_mode=arguments.storage_mode,
        default_db_backend=arguments.db_backend,
        default_db_uri=arguments.db_uri,
        default_db_name=arguments.db_name,
        default_db_prefix=arguments.db_prefix,
        default_run_id=arguments.run_id,
    )
