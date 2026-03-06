from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_evaluation_tables(
    articles: pd.DataFrame,
    topic_info: pd.DataFrame,
    trends: pd.DataFrame,
    recommendations: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    evaluation_frame = _labeled_articles(articles)
    if evaluation_frame.empty:
        presentation_metrics = _build_presentation_metrics(
            articles=articles,
            topic_info=topic_info,
            trends=trends,
            recommendations=recommendations,
            metadata=metadata,
            evaluation_summary=pd.DataFrame(),
        )
        report = _build_presentation_report(
            articles=articles,
            topic_info=topic_info,
            trends=trends,
            metadata=metadata,
            evaluation_summary=pd.DataFrame(),
            domain_performance=pd.DataFrame(),
            split_performance=pd.DataFrame(),
        )
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            presentation_metrics,
            report,
        )

    expected = evaluation_frame["expected_domain"].astype(str)
    predicted = evaluation_frame["predicted_domain"].astype(str)
    labels = sorted(set(expected) | set(predicted))

    precision, recall, f1_score, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        expected,
        predicted,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        expected,
        predicted,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(expected, predicted)

    evaluation_summary = pd.DataFrame(
        [
            {
                "labeled_articles": int(len(evaluation_frame)),
                "correct_predictions": int((expected == predicted).sum()),
                "accuracy": round(float(accuracy), 4),
                "macro_precision": round(float(macro_precision), 4),
                "macro_recall": round(float(macro_recall), 4),
                "macro_f1": round(float(macro_f1), 4),
                "weighted_precision": round(float(weighted_precision), 4),
                "weighted_recall": round(float(weighted_recall), 4),
                "weighted_f1": round(float(weighted_f1), 4),
                "num_expected_domains": int(expected.nunique()),
            }
        ]
    )

    domain_performance = pd.DataFrame(
        {
            "domain": labels,
            "precision": [round(float(value), 4) for value in precision],
            "recall": [round(float(value), 4) for value in recall],
            "f1_score": [round(float(value), 4) for value in f1_score],
            "support": [int(value) for value in support],
        }
    ).sort_values("support", ascending=False).reset_index(drop=True)

    matrix = confusion_matrix(expected, predicted, labels=labels)
    domain_confusion = (
        pd.DataFrame(matrix, index=labels, columns=labels)
        .reset_index(names="expected_domain")
        .melt(id_vars="expected_domain", var_name="predicted_domain", value_name="count")
        .sort_values(["expected_domain", "predicted_domain"])
        .reset_index(drop=True)
    )

    split_performance = _compute_split_performance(evaluation_frame)

    presentation_metrics = _build_presentation_metrics(
        articles=articles,
        topic_info=topic_info,
        trends=trends,
        recommendations=recommendations,
        metadata=metadata,
        evaluation_summary=evaluation_summary,
    )
    report = _build_presentation_report(
        articles=articles,
        topic_info=topic_info,
        trends=trends,
        metadata=metadata,
        evaluation_summary=evaluation_summary,
        domain_performance=domain_performance,
        split_performance=split_performance,
    )
    return (
        evaluation_summary,
        domain_performance,
        domain_confusion,
        split_performance,
        presentation_metrics,
        report,
    )


def _labeled_articles(articles: pd.DataFrame) -> pd.DataFrame:
    if "expected_domain" not in articles.columns:
        return pd.DataFrame()
    labeled = articles.copy()
    labeled["expected_domain"] = labeled["expected_domain"].fillna("").astype(str).str.strip()
    labeled = labeled[labeled["expected_domain"] != ""].reset_index(drop=True)
    return labeled


def _compute_split_performance(articles: pd.DataFrame) -> pd.DataFrame:
    if "split" not in articles.columns:
        return pd.DataFrame()

    split_rows: list[dict[str, Any]] = []
    for split_name, split_frame in articles.groupby(articles["split"].fillna("").astype(str)):
        clean_split = split_name.strip() or "unspecified"
        expected = split_frame["expected_domain"].astype(str)
        predicted = split_frame["predicted_domain"].astype(str)
        accuracy = accuracy_score(expected, predicted) if len(split_frame) else 0.0
        split_rows.append(
            {
                "split": clean_split,
                "labeled_articles": int(len(split_frame)),
                "accuracy": round(float(accuracy), 4),
            }
        )

    return pd.DataFrame(split_rows).sort_values("split").reset_index(drop=True)


def _build_presentation_metrics(
    articles: pd.DataFrame,
    topic_info: pd.DataFrame,
    trends: pd.DataFrame,
    recommendations: pd.DataFrame,
    metadata: dict[str, Any],
    evaluation_summary: pd.DataFrame,
) -> pd.DataFrame:
    largest_topic = ""
    if not topic_info.empty:
        largest_topic_row = topic_info.sort_values("size", ascending=False).iloc[0]
        largest_topic = str(largest_topic_row["topic_label"])

    evaluation_accuracy = ""
    if not evaluation_summary.empty:
        evaluation_accuracy = str(evaluation_summary.iloc[0]["accuracy"])

    rows = [
        {"metric_group": "project", "metric_name": "articles_analyzed", "metric_value": int(len(articles))},
        {"metric_group": "project", "metric_name": "sources", "metric_value": int(articles["source"].nunique())},
        {"metric_group": "project", "metric_name": "topics_discovered", "metric_value": int(len(topic_info))},
        {"metric_group": "project", "metric_name": "time_periods", "metric_value": int(trends["published_date"].nunique()) if not trends.empty else 0},
        {"metric_group": "project", "metric_name": "recommendation_links", "metric_value": int(len(recommendations))},
        {"metric_group": "insight", "metric_name": "largest_topic", "metric_value": largest_topic},
        {"metric_group": "evaluation", "metric_name": "domain_accuracy", "metric_value": evaluation_accuracy},
        {"metric_group": "run", "metric_name": "data_source", "metric_value": metadata.get("data_source", "")},
        {"metric_group": "run", "metric_name": "model_name", "metric_value": metadata.get("model_name", "")},
    ]
    return pd.DataFrame(rows)


def _build_presentation_report(
    articles: pd.DataFrame,
    topic_info: pd.DataFrame,
    trends: pd.DataFrame,
    metadata: dict[str, Any],
    evaluation_summary: pd.DataFrame,
    domain_performance: pd.DataFrame,
    split_performance: pd.DataFrame,
) -> str:
    lines = [
        "# Final Year Project Presentation Summary",
        "",
        f"- Run ID: {metadata.get('run_id', 'n/a')}",
        f"- Data source: {metadata.get('data_source', 'n/a')}",
        f"- Model: {metadata.get('model_name', 'n/a')}",
        f"- Articles analyzed: {len(articles)}",
        f"- Topics discovered: {len(topic_info)}",
        f"- Sources covered: {articles['source'].nunique()}",
    ]

    if not trends.empty:
        lines.append(f"- Time periods covered: {trends['published_date'].nunique()}")

    if not topic_info.empty:
        top_topics = topic_info.sort_values("size", ascending=False).head(3)
        lines.append("")
        lines.append("## Top Topics")
        for row in top_topics.itertuples():
            lines.append(f"- {row.topic_label}: {int(row.size)} articles")

    if not evaluation_summary.empty:
        summary_row = evaluation_summary.iloc[0]
        lines.append("")
        lines.append("## Evaluation")
        lines.append(f"- Domain accuracy: {summary_row['accuracy']}")
        lines.append(f"- Macro F1 score: {summary_row['macro_f1']}")
        lines.append(f"- Weighted F1 score: {summary_row['weighted_f1']}")

    if not domain_performance.empty:
        lines.append("")
        lines.append("## Best Domain Scores")
        for row in domain_performance.sort_values("f1_score", ascending=False).head(3).itertuples():
            lines.append(f"- {row.domain}: F1={row.f1_score}, precision={row.precision}, recall={row.recall}")

    if not split_performance.empty:
        lines.append("")
        lines.append("## Split Performance")
        for row in split_performance.itertuples():
            lines.append(f"- {row.split}: accuracy={row.accuracy} over {int(row.labeled_articles)} articles")

    return "\n".join(lines) + "\n"
