from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd

from .analytics import (
    attach_cluster_coordinates,
    compute_article_recommendations,
    compute_emerging_topics,
    compute_topic_relationships,
    compute_trends,
)
from .categorization import DomainClassifier
from .collectors import (
    CollectionResult,
    CompositeCollector,
    DemoNewsCollector,
    GenericHTMLCollector,
    LocalCSVCollector,
    RSSNewsCollector,
)
from .evaluation import compute_evaluation_tables
from .models import NewsArticle, PipelineArtifacts
from .presentation_assets import ensure_presentation_assets
from .preprocessing import TextPreprocessor
from .storage import ArtifactStore
from .topic_modeling import build_topic_modeler


@dataclass(slots=True)
class PipelineConfig:
    data_source: str = "demo"
    model_name: str = "lda"
    num_topics: int = 6
    top_words: int = 8
    max_features: int = 3000
    limit_per_source: int = 20
    top_k_recommendations: int = 3
    enable_spacy: bool = False
    enable_nltk: bool = False
    rss_feeds: dict[str, str | list[str]] | None = None
    html_sources: dict[str, str] | None = None
    csv_path: str | None = None
    db_backend: str | None = None
    db_uri: str | None = None
    db_name: str = "news_topic_analysis"
    db_prefix: str = "news_topic_analysis"
    artifact_dir: Path = field(default_factory=lambda: Path("artifacts/latest"))


class NewsTopicPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = TextPreprocessor(
            use_spacy=self.config.enable_spacy,
            use_nltk=self.config.enable_nltk,
        )
        self.domain_classifier = DomainClassifier()

    def _build_collector(self) -> object:
        source = self.config.data_source.lower()
        if source == "demo":
            return DemoNewsCollector()
        if source == "rss":
            return RSSNewsCollector(feed_urls=self.config.rss_feeds)
        if source == "csv":
            return LocalCSVCollector(csv_path=self.config.csv_path)
        if source == "html":
            return GenericHTMLCollector(source_urls=self.config.html_sources)
        if source == "mixed":
            return CompositeCollector(
                [
                    RSSNewsCollector(feed_urls=self.config.rss_feeds),
                    GenericHTMLCollector(source_urls=self.config.html_sources),
                ]
            )
        raise ValueError(f"Unsupported data source: {self.config.data_source}")

    @staticmethod
    def _summarize_collection(result: CollectionResult) -> dict[str, object]:
        status_counts: dict[str, int] = {}
        for report in result.reports:
            status_counts[report.status] = status_counts.get(report.status, 0) + 1

        return {
            "total_articles": len(result.articles),
            "collector_count": len(result.reports),
            "status_counts": status_counts,
            "reports": [
                {
                    "collector": report.collector,
                    "source": report.source,
                    "target": report.target,
                    "status": report.status,
                    "article_count": report.article_count,
                    "error_message": report.error_message,
                }
                for report in result.reports
            ],
        }

    def run(self, articles: Sequence[NewsArticle] | None = None) -> PipelineArtifacts:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        collection_summary = {"total_articles": 0, "collector_count": 0, "status_counts": {}, "reports": []}
        if articles is not None:
            collected_articles = list(articles)
        else:
            collector = self._build_collector()
            if hasattr(collector, "collect_with_report"):
                collection_result = collector.collect_with_report(limit_per_source=self.config.limit_per_source)
                collected_articles = collection_result.articles
                collection_summary = self._summarize_collection(collection_result)
            else:
                collected_articles = collector.collect(limit_per_source=self.config.limit_per_source)
                collection_summary = {
                    "total_articles": len(collected_articles),
                    "collector_count": 1,
                    "status_counts": {"success": 1},
                    "reports": [],
                }
        if not collected_articles:
            raise ValueError("No articles were collected for the pipeline run.")

        frame = pd.DataFrame([article.to_record() for article in collected_articles])
        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
        frame["published_at"] = frame["published_at"].fillna(pd.Timestamp.now(tz="UTC"))
        frame = frame.sort_values("published_at").reset_index(drop=True)

        processed = self.preprocessor.preprocess_frame(frame)
        if processed.empty:
            raise ValueError("Preprocessing removed every article. Check the input data.")

        processed = self.domain_classifier.annotate_frame(processed)

        modeler = build_topic_modeler(
            model_name=self.config.model_name,
            num_topics=self.config.num_topics,
            top_words=self.config.top_words,
            max_features=self.config.max_features,
        )
        modeled = modeler.fit_transform(processed)
        articles_with_clusters = attach_cluster_coordinates(modeled.articles, modeled.document_term_matrix)

        trends = compute_trends(articles_with_clusters)
        emerging_topics = compute_emerging_topics(trends)
        topic_relationships = compute_topic_relationships(modeled.topic_info, modeled.topic_term_matrix)
        recommendations = compute_article_recommendations(
            articles_with_clusters,
            modeled.document_term_matrix,
            top_k=self.config.top_k_recommendations,
        )
        (
            evaluation_summary,
            domain_performance,
            domain_confusion,
            split_performance,
            presentation_metrics,
            presentation_report,
        ) = compute_evaluation_tables(
            articles=articles_with_clusters,
            topic_info=modeled.topic_info,
            trends=trends,
            recommendations=recommendations,
            metadata={"data_source": self.config.data_source, "model_name": modeled.model_name, "run_id": run_id},
        )

        metadata = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": self.config.data_source,
            "model_name": modeled.model_name,
            "vectorizer_name": modeled.vectorizer_name,
            "num_articles": int(len(articles_with_clusters)),
            "num_topics": int(len(modeled.topic_info)),
            "sources": sorted(articles_with_clusters["source"].dropna().astype(str).unique().tolist()),
            "collection_summary": collection_summary,
            "evaluation_available": bool(not evaluation_summary.empty),
            "artifact_dir": str(self.config.artifact_dir),
        }
        if not evaluation_summary.empty:
            metadata["evaluation_summary"] = evaluation_summary.iloc[0].to_dict()

        artifacts = PipelineArtifacts(
            articles=articles_with_clusters,
            topic_info=modeled.topic_info,
            trends=trends,
            emerging_topics=emerging_topics,
            topic_relationships=topic_relationships,
            recommendations=recommendations,
            evaluation_summary=evaluation_summary,
            domain_performance=domain_performance,
            domain_confusion=domain_confusion,
            split_performance=split_performance,
            presentation_metrics=presentation_metrics,
            metadata=metadata,
            artifact_dir=self.config.artifact_dir,
            presentation_report=presentation_report,
        )

        if self.config.db_backend and self.config.db_uri:
            from .database import DatabaseArtifactStore, DatabaseConfig

            database_summary = DatabaseArtifactStore(
                DatabaseConfig(
                    backend=self.config.db_backend,
                    uri=self.config.db_uri,
                    name=self.config.db_name,
                    prefix=self.config.db_prefix,
                )
            ).save(artifacts)
            artifacts.metadata["database"] = database_summary
        else:
            artifacts.metadata["database"] = {"enabled": False}

        ArtifactStore.save(artifacts, self.config.artifact_dir)
        ensure_presentation_assets(
            artifact_dir=self.config.artifact_dir,
            metadata=artifacts.metadata,
            articles=artifacts.articles,
            evaluation_summary=artifacts.evaluation_summary,
        )
        return artifacts
