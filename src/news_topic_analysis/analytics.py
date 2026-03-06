from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def attach_cluster_coordinates(articles: pd.DataFrame, document_term_matrix: object) -> pd.DataFrame:
    annotated = articles.copy()
    row_count = len(annotated)
    if row_count == 0:
        annotated["cluster_x"] = []
        annotated["cluster_y"] = []
        return annotated

    try:
        feature_count = int(document_term_matrix.shape[1])
    except Exception:
        feature_count = 0

    if row_count < 2 or feature_count < 2:
        annotated["cluster_x"] = np.zeros(row_count)
        annotated["cluster_y"] = np.zeros(row_count)
        return annotated

    components = min(2, row_count - 1, feature_count)
    if components < 2:
        annotated["cluster_x"] = np.zeros(row_count)
        annotated["cluster_y"] = np.zeros(row_count)
        return annotated

    reducer = TruncatedSVD(n_components=components, random_state=42)
    coordinates = reducer.fit_transform(document_term_matrix)
    annotated["cluster_x"] = np.round(coordinates[:, 0], 4)
    annotated["cluster_y"] = np.round(coordinates[:, 1], 4)
    return annotated


def compute_trends(articles: pd.DataFrame) -> pd.DataFrame:
    if articles.empty:
        return pd.DataFrame(columns=["published_date", "topic_id", "topic_label", "article_count", "share"])

    trend_frame = articles.copy()
    trend_frame["published_date"] = pd.to_datetime(trend_frame["published_at"], utc=True).dt.strftime("%Y-%m-%d")
    trend_counts = (
        trend_frame.groupby(["published_date", "topic_id", "topic_label"], as_index=False)
        .size()
        .rename(columns={"size": "article_count"})
        .sort_values(["published_date", "article_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    totals = trend_counts.groupby("published_date")["article_count"].transform("sum")
    trend_counts["share"] = np.round(trend_counts["article_count"] / totals, 4)
    return trend_counts


def compute_emerging_topics(trends: pd.DataFrame, recent_periods: int = 3) -> pd.DataFrame:
    if trends.empty:
        return pd.DataFrame(
            columns=["topic_id", "topic_label", "recent_mean", "baseline_mean", "delta", "lift"]
        )

    pivot = trends.pivot_table(
        index="published_date",
        columns=["topic_id", "topic_label"],
        values="article_count",
        fill_value=0,
    ).sort_index()

    recent = pivot.tail(recent_periods)
    baseline = pivot.iloc[:-recent_periods]
    if baseline.empty:
        baseline = pivot.head(max(1, len(pivot) - 1))

    rows: list[dict[str, object]] = []
    for topic_id, topic_label in pivot.columns:
        recent_mean = float(recent[(topic_id, topic_label)].mean())
        baseline_mean = float(baseline[(topic_id, topic_label)].mean()) if not baseline.empty else 0.0
        delta = recent_mean - baseline_mean
        lift = recent_mean / max(baseline_mean, 0.25)
        rows.append(
            {
                "topic_id": int(topic_id),
                "topic_label": topic_label,
                "recent_mean": round(recent_mean, 4),
                "baseline_mean": round(baseline_mean, 4),
                "delta": round(delta, 4),
                "lift": round(lift, 4),
            }
        )

    return pd.DataFrame(rows).sort_values(["delta", "lift"], ascending=False).reset_index(drop=True)


def compute_topic_relationships(topic_info: pd.DataFrame, topic_term_matrix: np.ndarray) -> pd.DataFrame:
    if topic_info.empty or len(topic_info) < 2:
        return pd.DataFrame(
            columns=["topic_a", "topic_a_label", "topic_b", "topic_b_label", "similarity"]
        )

    similarity_matrix = cosine_similarity(topic_term_matrix)
    rows: list[dict[str, object]] = []

    for left_index in range(len(topic_info)):
        for right_index in range(left_index + 1, len(topic_info)):
            rows.append(
                {
                    "topic_a": int(topic_info.iloc[left_index]["topic_id"]),
                    "topic_a_label": str(topic_info.iloc[left_index]["topic_label"]),
                    "topic_b": int(topic_info.iloc[right_index]["topic_id"]),
                    "topic_b_label": str(topic_info.iloc[right_index]["topic_label"]),
                    "similarity": round(float(similarity_matrix[left_index, right_index]), 4),
                }
            )

    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def compute_article_recommendations(
    articles: pd.DataFrame,
    document_term_matrix: object,
    top_k: int = 3,
) -> pd.DataFrame:
    if articles.empty or len(articles) < 2:
        return pd.DataFrame(
            columns=[
                "article_id",
                "article_title",
                "recommended_article_id",
                "recommended_title",
                "similarity_score",
                "topic_match",
            ]
        )

    similarity_matrix = cosine_similarity(document_term_matrix)
    rows: list[dict[str, object]] = []

    normalized_articles = articles.reset_index(drop=True)
    for index, article in normalized_articles.iterrows():
        ranked_indices = np.argsort(similarity_matrix[index])[::-1]
        picked = 0
        for candidate_index in ranked_indices:
            if candidate_index == index:
                continue
            candidate = normalized_articles.iloc[candidate_index]
            rows.append(
                {
                    "article_id": article["article_id"],
                    "article_title": article["title"],
                    "recommended_article_id": candidate["article_id"],
                    "recommended_title": candidate["title"],
                    "similarity_score": round(float(similarity_matrix[index, candidate_index]), 4),
                    "topic_match": bool(article["topic_id"] == candidate["topic_id"]),
                }
            )
            picked += 1
            if picked >= top_k:
                break

    return pd.DataFrame(rows)
