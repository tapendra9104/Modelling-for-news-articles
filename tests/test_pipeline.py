from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("sqlalchemy")

from news_topic_analysis.dashboard import load_dashboard_payload
from news_topic_analysis.pipeline import NewsTopicPipeline, PipelineConfig
from news_topic_analysis.presentation_assets import ensure_presentation_assets
from news_topic_analysis.sample_data import default_csv_dataset_path


def test_demo_pipeline_creates_artifacts() -> None:
    artifact_dir = Path("artifacts/test_pipeline_run")
    shutil.rmtree(artifact_dir, ignore_errors=True)

    config = PipelineConfig(
        data_source="demo",
        model_name="lda",
        num_topics=6,
        artifact_dir=artifact_dir,
    )

    result = NewsTopicPipeline(config).run()

    assert not result.articles.empty
    assert not result.topic_info.empty
    assert {"topic_id", "topic_label", "predicted_domain"} <= set(result.articles.columns)
    assert result.metadata["collection_summary"]["total_articles"] == len(result.articles)
    assert result.metadata["collection_summary"]["status_counts"]["success"] >= 1
    assert (artifact_dir / "articles.csv").exists()
    assert (artifact_dir / "metadata.json").exists()
    asset_paths = ensure_presentation_assets(
        artifact_dir=artifact_dir,
        metadata=result.metadata,
        articles=result.articles,
        evaluation_summary=result.evaluation_summary,
    )
    assert asset_paths["project_overview"].exists()
    assert asset_paths["technology_stack"].exists()


def test_database_round_trip_with_sql_backend() -> None:
    artifact_dir = Path("artifacts/test_pipeline_db")
    database_path = Path("artifacts/test_pipeline.db")
    shutil.rmtree(artifact_dir, ignore_errors=True)
    if database_path.exists():
        database_path.unlink()

    config = PipelineConfig(
        data_source="demo",
        model_name="lda",
        num_topics=6,
        artifact_dir=artifact_dir,
        db_backend="mysql",
        db_uri=f"sqlite:///{database_path.as_posix()}",
        db_prefix="test_news_topic_analysis",
    )

    result = NewsTopicPipeline(config).run()
    payload = load_dashboard_payload(
        storage_mode="database",
        artifact_dir=artifact_dir,
        db_backend="mysql",
        db_uri=f"sqlite:///{database_path.as_posix()}",
        db_name="news_topic_analysis",
        db_prefix="test_news_topic_analysis",
        run_id=result.metadata["run_id"],
    )

    assert payload["metadata"]["run_id"] == result.metadata["run_id"]
    assert len(payload["articles"]) == len(result.articles)
    assert len(payload["topic_info"]) == len(result.topic_info)
    assert payload["metadata"]["database"]["enabled"] is True


def test_bundled_csv_dataset_pipeline_runs() -> None:
    artifact_dir = Path("artifacts/test_pipeline_csv")
    shutil.rmtree(artifact_dir, ignore_errors=True)

    config = PipelineConfig(
        data_source="csv",
        model_name="lda",
        num_topics=6,
        csv_path=str(default_csv_dataset_path()),
        artifact_dir=artifact_dir,
    )

    result = NewsTopicPipeline(config).run()

    assert len(result.articles) == 96
    assert not result.evaluation_summary.empty
    assert not result.domain_performance.empty
    assert not result.presentation_metrics.empty
    assert result.metadata["collection_summary"]["reports"][0]["target"].endswith("news_articles_extended.csv")
    assert (artifact_dir / "articles.csv").exists()
    assert (artifact_dir / "evaluation_summary.csv").exists()
    assert (artifact_dir / "presentation_report.md").exists()
    image_dir = artifact_dir / "images"
    assert image_dir.exists()
    assert any(path.name.endswith("_project_overview.png") for path in image_dir.iterdir())
    assert any(path.name.endswith("_technology_stack.png") for path in image_dir.iterdir())
