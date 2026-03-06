from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .models import PipelineArtifacts


@dataclass(slots=True)
class DatabaseConfig:
    backend: str
    uri: str
    name: str = "news_topic_analysis"
    prefix: str = "news_topic_analysis"


class DatabaseArtifactStore:
    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config

    def save(self, artifacts: PipelineArtifacts) -> dict[str, Any]:
        backend = self.config.backend.lower()
        if backend == "mongodb":
            return self._save_mongodb(artifacts)
        if backend == "mysql":
            return self._save_mysql(artifacts)
        raise ValueError(f"Unsupported database backend: {self.config.backend}")

    def load(self, run_id: str | None = None) -> dict[str, Any]:
        backend = self.config.backend.lower()
        if backend == "mongodb":
            return self._load_mongodb(run_id=run_id)
        if backend == "mysql":
            return self._load_mysql(run_id=run_id)
        raise ValueError(f"Unsupported database backend: {self.config.backend}")

    def _save_mysql(self, artifacts: PipelineArtifacts) -> dict[str, Any]:
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise RuntimeError(
                "MySQL persistence requires the database extra. Install with `pip install -e .[database]`."
            ) from exc

        engine = create_engine(self.config.uri)
        run_id = str(artifacts.metadata["run_id"])
        table_counts: dict[str, int] = {}
        runs_table = f"{self.config.prefix}_runs"

        for label, frame in self._build_frames(artifacts).items():
            db_frame = self._prepare_frame_for_database(frame, run_id)
            table_name = f"{self.config.prefix}_{label}"
            db_frame.to_sql(table_name, con=engine, if_exists="append", index=False, method="multi")
            table_counts[table_name] = int(len(db_frame))

        database_summary = {
            "enabled": True,
            "backend": "mysql",
            "target": self._mask_uri(self.config.uri),
            "database_name": self._database_name_from_uri(self.config.uri) or self.config.name,
            "table_prefix": self.config.prefix,
            "tables_written": {runs_table: 1, **table_counts},
        }
        metadata_payload = dict(artifacts.metadata)
        metadata_payload["database"] = database_summary
        metadata_frame = pd.DataFrame([self._serialize_metadata(metadata_payload)])
        metadata_frame.to_sql(runs_table, con=engine, if_exists="append", index=False, method="multi")

        engine.dispose()
        return database_summary

    def _save_mongodb(self, artifacts: PipelineArtifacts) -> dict[str, Any]:
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError(
                "MongoDB persistence requires the database extra. Install with `pip install -e .[database]`."
            ) from exc

        client = MongoClient(self.config.uri)
        database_name = self._database_name_from_uri(self.config.uri) or self.config.name
        database = client[database_name]
        run_id = str(artifacts.metadata["run_id"])
        collection_counts: dict[str, int] = {}

        for label, frame in self._build_frames(artifacts).items():
            collection_name = f"{self.config.prefix}_{label}"
            records = self._prepare_records_for_mongodb(frame, run_id)
            if records:
                database[collection_name].insert_many(records)
            collection_counts[collection_name] = len(records)

        runs_collection = f"{self.config.prefix}_runs"
        database_summary = {
            "enabled": True,
            "backend": "mongodb",
            "target": self._mask_uri(self.config.uri),
            "database_name": database_name,
            "collection_prefix": self.config.prefix,
            "collections_written": {runs_collection: 1, **collection_counts},
        }
        metadata_payload = dict(artifacts.metadata)
        metadata_payload["database"] = database_summary
        database[runs_collection].insert_one(self._normalize_record(self._serialize_metadata(metadata_payload)))

        client.close()
        return database_summary

    def _load_mysql(self, run_id: str | None = None) -> dict[str, Any]:
        try:
            from sqlalchemy import create_engine, text
        except ImportError as exc:
            raise RuntimeError(
                "MySQL persistence requires the database extra. Install with `pip install -e .[database]`."
            ) from exc

        engine = create_engine(self.config.uri)
        runs_table = f"{self.config.prefix}_runs"
        runs_frame = pd.read_sql_query(text(f"SELECT * FROM {runs_table}"), con=engine)
        if runs_frame.empty:
            raise ValueError("No runs were found in the configured SQL database.")

        if run_id:
            selected_runs = runs_frame[runs_frame["run_id"] == run_id]
            if selected_runs.empty:
                raise ValueError(f"Run id `{run_id}` was not found in the configured SQL database.")
            run_row = selected_runs.iloc[-1]
        else:
            sort_column = "generated_at" if "generated_at" in runs_frame.columns else "run_id"
            run_row = runs_frame.sort_values(sort_column).iloc[-1]

        selected_run_id = str(run_row["run_id"])
        payload = self._empty_payload()
        payload["metadata"] = self._deserialize_metadata(run_row.to_dict())
        payload["artifact_dir"] = Path(str(payload["metadata"].get("artifact_dir", "database")))

        for label in self._frame_labels():
            table_name = f"{self.config.prefix}_{label}"
            query = text(f"SELECT * FROM {table_name} WHERE run_id = :run_id")
            frame = pd.read_sql_query(query, con=engine, params={"run_id": selected_run_id})
            if "run_id" in frame.columns:
                frame = frame.drop(columns=["run_id"])
            payload[label] = self._restore_frame_types(label, frame)

        engine.dispose()
        return payload

    def _load_mongodb(self, run_id: str | None = None) -> dict[str, Any]:
        try:
            from pymongo import DESCENDING, MongoClient
        except ImportError as exc:
            raise RuntimeError(
                "MongoDB persistence requires the database extra. Install with `pip install -e .[database]`."
            ) from exc

        client = MongoClient(self.config.uri)
        database_name = self._database_name_from_uri(self.config.uri) or self.config.name
        database = client[database_name]
        runs_collection = f"{self.config.prefix}_runs"

        run_query = {"run_id": run_id} if run_id else {}
        cursor = database[runs_collection].find(run_query).sort("generated_at", DESCENDING).limit(1)
        run_records = list(cursor)
        if not run_records:
            client.close()
            if run_id:
                raise ValueError(f"Run id `{run_id}` was not found in the configured MongoDB database.")
            raise ValueError("No runs were found in the configured MongoDB database.")

        run_record = run_records[0]
        selected_run_id = str(run_record["run_id"])
        payload = self._empty_payload()
        payload["metadata"] = self._deserialize_metadata(run_record)
        payload["artifact_dir"] = Path(str(payload["metadata"].get("artifact_dir", "database")))

        for label in self._frame_labels():
            collection_name = f"{self.config.prefix}_{label}"
            records = list(database[collection_name].find({"run_id": selected_run_id}, {"_id": 0}))
            frame = pd.DataFrame(records)
            if "run_id" in frame.columns:
                frame = frame.drop(columns=["run_id"])
            payload[label] = self._restore_frame_types(label, frame)

        client.close()
        return payload

    def _build_frames(self, artifacts: PipelineArtifacts) -> dict[str, pd.DataFrame]:
        articles = artifacts.articles.copy()
        if "tokens" in articles.columns:
            articles["tokens"] = articles["tokens"].map(
                lambda value: ", ".join(value) if isinstance(value, list) else value
            )

        return {
            "articles": articles,
            "topic_info": artifacts.topic_info.copy(),
            "trends": artifacts.trends.copy(),
            "emerging_topics": artifacts.emerging_topics.copy(),
            "topic_relationships": artifacts.topic_relationships.copy(),
            "recommendations": artifacts.recommendations.copy(),
            "evaluation_summary": artifacts.evaluation_summary.copy(),
            "domain_performance": artifacts.domain_performance.copy(),
            "domain_confusion": artifacts.domain_confusion.copy(),
            "split_performance": artifacts.split_performance.copy(),
            "presentation_metrics": artifacts.presentation_metrics.copy(),
        }

    @staticmethod
    def _prepare_frame_for_database(frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
        db_frame = frame.copy()
        db_frame.insert(0, "run_id", run_id)
        for column in db_frame.columns:
            db_frame[column] = db_frame[column].map(DatabaseArtifactStore._normalize_scalar)
        return db_frame

    @staticmethod
    def _prepare_records_for_mongodb(frame: pd.DataFrame, run_id: str) -> list[dict[str, Any]]:
        db_frame = DatabaseArtifactStore._prepare_frame_for_database(frame, run_id)
        return [
            DatabaseArtifactStore._normalize_record(record)
            for record in db_frame.to_dict(orient="records")
        ]

    @staticmethod
    def _empty_payload() -> dict[str, Any]:
        return {
            "articles": pd.DataFrame(),
            "topic_info": pd.DataFrame(),
            "trends": pd.DataFrame(),
            "emerging_topics": pd.DataFrame(),
            "topic_relationships": pd.DataFrame(),
            "recommendations": pd.DataFrame(),
            "evaluation_summary": pd.DataFrame(),
            "domain_performance": pd.DataFrame(),
            "domain_confusion": pd.DataFrame(),
            "split_performance": pd.DataFrame(),
            "presentation_metrics": pd.DataFrame(),
            "metadata": {},
            "artifact_dir": Path("database"),
            "presentation_report": "",
        }

    @staticmethod
    def _frame_labels() -> list[str]:
        return [
            "articles",
            "topic_info",
            "trends",
            "emerging_topics",
            "topic_relationships",
            "recommendations",
            "evaluation_summary",
            "domain_performance",
            "domain_confusion",
            "split_performance",
            "presentation_metrics",
        ]

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                serialized[key] = json.dumps(value, default=str)
            else:
                serialized[key] = DatabaseArtifactStore._normalize_scalar(value)
        return serialized

    @staticmethod
    def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: DatabaseArtifactStore._normalize_scalar(value)
            for key, value in record.items()
        }

    @staticmethod
    def _deserialize_metadata(record: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key, value in record.items():
            if key == "_id":
                continue
            metadata[key] = DatabaseArtifactStore._try_json_load(value)
        return metadata

    @staticmethod
    def _restore_frame_types(label: str, frame: pd.DataFrame) -> pd.DataFrame:
        restored = frame.copy()
        if label == "articles":
            if "published_at" in restored.columns:
                restored["published_at"] = pd.to_datetime(restored["published_at"], utc=True, errors="coerce")
            if "tokens" in restored.columns:
                restored["tokens"] = restored["tokens"].map(
                    lambda value: value.split(", ") if isinstance(value, str) and value else []
                )
        elif label == "trends" and "published_date" in restored.columns:
            restored["published_date"] = pd.to_datetime(restored["published_date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
        return restored

    @staticmethod
    def _try_json_load(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return value
        if text[0] not in "[{":
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _normalize_scalar(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except TypeError:
                pass
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value

    @staticmethod
    def _mask_uri(uri: str) -> str:
        parsed = urlparse(uri)
        if not parsed.password:
            return uri
        netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
        return parsed._replace(netloc=netloc).geturl()

    @staticmethod
    def _database_name_from_uri(uri: str) -> str | None:
        parsed = urlparse(uri)
        database_name = parsed.path.lstrip("/")
        return database_name or None
