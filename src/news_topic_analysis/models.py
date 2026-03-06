from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class NewsArticle:
    article_id: str
    title: str
    content: str
    source: str
    published_at: datetime
    url: str = ""
    language: str = "en"
    expected_domain: str = ""
    dataset_split: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "url": self.url,
            "language": self.language,
            "expected_domain": self.expected_domain,
            "split": self.dataset_split,
        }


@dataclass(slots=True)
class PipelineArtifacts:
    articles: pd.DataFrame
    topic_info: pd.DataFrame
    trends: pd.DataFrame
    emerging_topics: pd.DataFrame
    topic_relationships: pd.DataFrame
    recommendations: pd.DataFrame
    evaluation_summary: pd.DataFrame
    domain_performance: pd.DataFrame
    domain_confusion: pd.DataFrame
    split_performance: pd.DataFrame
    presentation_metrics: pd.DataFrame
    metadata: dict[str, Any]
    artifact_dir: Path
    presentation_report: str = ""
