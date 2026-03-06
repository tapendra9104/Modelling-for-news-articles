from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import PipelineArtifacts

TABLE_FILES = {
    "articles": "articles.csv",
    "topic_info": "topic_info.csv",
    "trends": "trends.csv",
    "emerging_topics": "emerging_topics.csv",
    "topic_relationships": "topic_relationships.csv",
    "recommendations": "recommendations.csv",
    "evaluation_summary": "evaluation_summary.csv",
    "domain_performance": "domain_performance.csv",
    "domain_confusion": "domain_confusion.csv",
    "split_performance": "split_performance.csv",
    "presentation_metrics": "presentation_metrics.csv",
}


class ArtifactStore:
    @staticmethod
    def save(result: PipelineArtifacts, output_dir: str | Path | None = None) -> Path:
        target = Path(output_dir or result.artifact_dir)
        target.mkdir(parents=True, exist_ok=True)

        articles = result.articles.copy()
        if "tokens" in articles.columns:
            articles["tokens"] = articles["tokens"].map(
                lambda value: ", ".join(value) if isinstance(value, list) else value
            )

        frames = {
            "articles": articles,
            "topic_info": result.topic_info,
            "trends": result.trends,
            "emerging_topics": result.emerging_topics,
            "topic_relationships": result.topic_relationships,
            "recommendations": result.recommendations,
            "evaluation_summary": result.evaluation_summary,
            "domain_performance": result.domain_performance,
            "domain_confusion": result.domain_confusion,
            "split_performance": result.split_performance,
            "presentation_metrics": result.presentation_metrics,
        }

        for key, frame in frames.items():
            frame.to_csv(target / TABLE_FILES[key], index=False)

        with (target / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(result.metadata, handle, indent=2, default=str)
        if result.presentation_report:
            (target / "presentation_report.md").write_text(result.presentation_report, encoding="utf-8")

        return target

    @staticmethod
    def load(output_dir: str | Path) -> dict[str, Any]:
        target = Path(output_dir)
        payload: dict[str, Any] = {}

        for key, filename in TABLE_FILES.items():
            file_path = target / filename
            parse_dates = ["published_at"] if key == "articles" else ["published_date"] if key == "trends" else None
            payload[key] = pd.read_csv(file_path, parse_dates=parse_dates) if file_path.exists() else pd.DataFrame()

        metadata_path = target / "metadata.json"
        payload["metadata"] = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
        report_path = target / "presentation_report.md"
        payload["presentation_report"] = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        payload["artifact_dir"] = target
        return payload
