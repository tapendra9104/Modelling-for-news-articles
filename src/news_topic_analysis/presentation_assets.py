from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

MPL_CONFIG_DIR = Path("artifacts/mplconfig")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def ensure_presentation_assets(
    artifact_dir: str | Path,
    metadata: dict[str, Any],
    articles: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
) -> dict[str, Path]:
    asset_dir = _resolve_asset_dir(artifact_dir, metadata)
    run_token = _safe_run_token(str(metadata.get("run_id", "latest")))
    asset_paths = {
        "project_overview": asset_dir / f"{run_token}_project_overview.png",
        "technology_stack": asset_dir / f"{run_token}_technology_stack.png",
    }

    if not asset_paths["project_overview"].exists():
        _create_project_overview_image(asset_paths["project_overview"], metadata, articles, evaluation_summary)
    if not asset_paths["technology_stack"].exists():
        _create_technology_stack_image(asset_paths["technology_stack"], metadata, articles)

    return asset_paths


def _resolve_asset_dir(artifact_dir: str | Path, metadata: dict[str, Any]) -> Path:
    requested_dir = Path(artifact_dir)
    if str(requested_dir).strip() and requested_dir.name != "database":
        target = requested_dir / "images"
    else:
        run_token = _safe_run_token(str(metadata.get("run_id", "latest")))
        target = Path("artifacts") / "dashboard_assets" / run_token
    target.mkdir(parents=True, exist_ok=True)
    return target


def _safe_run_token(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)


def _create_project_overview_image(
    output_path: Path,
    metadata: dict[str, Any],
    articles: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(figsize=(15, 9), facecolor="#F4EFE6")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    title = "AI News Topic Modeling and Trend Analysis"
    subtitle = (
        f"Run {metadata.get('run_id', 'n/a')} | "
        f"Model {str(metadata.get('model_name', 'n/a')).upper()} | "
        f"Source {str(metadata.get('data_source', 'n/a')).upper()}"
    )
    axis.text(0.05, 0.94, title, fontsize=24, fontweight="bold", color="#122C34")
    axis.text(0.05, 0.90, subtitle, fontsize=12, color="#31525B")

    accuracy = "n/a"
    if not evaluation_summary.empty and "accuracy" in evaluation_summary.columns:
        accuracy = f"{float(evaluation_summary.iloc[0]['accuracy']):.2f}"

    _draw_block(
        axis,
        x=0.05,
        y=0.67,
        width=0.27,
        height=0.18,
        title="Why This Project Matters",
        body=(
            "Turns large volumes of unstructured news into clear topics, trend signals, and related-article insights."
        ),
        facecolor="#FFF9F0",
        edgecolor="#D9A441",
        wrap_width=24,
    )
    _draw_block(
        axis,
        x=0.05,
        y=0.39,
        width=0.27,
        height=0.19,
        title="Who Uses It",
        body=(
            "Journalists monitor breaking themes.\nResearchers study narrative shifts.\nBusinesses track sector risk and visibility."
        ),
        facecolor="#FFF9F0",
        edgecolor="#D9A441",
        wrap_width=24,
    )
    _draw_block(
        axis,
        x=0.05,
        y=0.12,
        width=0.27,
        height=0.18,
        title="Current Run Snapshot",
        body=(
            f"Articles: {len(articles)}\n"
            f"Sources: {int(articles['source'].nunique()) if 'source' in articles else 0}\n"
            f"Topics: {int(metadata.get('num_topics', 0))}\n"
            f"Accuracy: {accuracy}"
        ),
        facecolor="#FFF9F0",
        edgecolor="#D9A441",
        wrap_width=20,
    )

    workflow_steps = [
        ("Collect", "RSS, HTML, and CSV inputs"),
        ("Preprocess", "Clean, tokenize, and lemmatize"),
        ("Model", "LDA, NMF, or BERTopic"),
        ("Analyze", "Trends, clusters, and similarity"),
        ("Deliver", "Dashboards, reports, and recommendations"),
    ]

    x_positions = [0.37, 0.50, 0.63, 0.76, 0.89]
    for index, ((label, description), x_position) in enumerate(zip(workflow_steps, x_positions, strict=True)):
        _draw_block(
            axis,
            x=x_position - 0.055,
            y=0.46,
            width=0.11,
            height=0.19,
            title=label,
            body=description,
            facecolor="#DDEAF0" if index % 2 == 0 else "#EAF4EA",
            edgecolor="#31525B",
            title_size=13,
            body_size=9,
            wrap_width=16,
        )
        if index < len(workflow_steps) - 1:
            arrow = FancyArrowPatch(
                (x_position + 0.055, 0.555),
                (x_positions[index + 1] - 0.055, 0.555),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2,
                color="#31525B",
            )
            axis.add_patch(arrow)

    _draw_block(
        axis,
        x=0.37,
        y=0.12,
        width=0.57,
        height=0.18,
        title="Operational Value",
        body=(
            "Cuts manual review, highlights fast-moving themes, and gives a repeatable base for newsroom analytics, market intelligence, and academic demonstrations."
        ),
        facecolor="#122C34",
        edgecolor="#122C34",
        title_color="#F4EFE6",
        body_color="#F4EFE6",
        wrap_width=74,
    )

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _create_technology_stack_image(output_path: Path, metadata: dict[str, Any], articles: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(15, 9), facecolor="#F8F7F4")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    active_sources = ", ".join(sorted(articles["source"].dropna().astype(str).unique().tolist())) if "source" in articles else "n/a"
    axis.text(0.05, 0.94, "Technology Stack and System Design", fontsize=24, fontweight="bold", color="#173F35")
    axis.text(
        0.05,
        0.90,
        (
            f"Primary language: Python | Active model: {str(metadata.get('model_name', 'n/a')).upper()} | "
            f"Observed sources: {active_sources or 'n/a'}"
        ),
        fontsize=12,
        color="#355C4D",
    )

    stack_blocks = [
        (
            0.05,
            0.54,
            "Data Ingestion",
            "RSS collectors, HTML scraping, and local CSV datasets support offline demos and live runs.",
            "#F5E6CC",
        ),
        (
            0.37,
            0.54,
            "NLP Preprocessing",
            "pandas manages datasets while NLTK and optional spaCy handle cleaning, tokenization, and lemmatization.",
            "#DCEEF2",
        ),
        (
            0.69,
            0.54,
            "Topic Intelligence",
            "scikit-learn powers vectorization, LDA, and NMF. BERTopic can be enabled for embedding-based discovery.",
            "#DDEFD8",
        ),
        (
            0.05,
            0.20,
            "Analytics and Search",
            "Trend detection, topic relationships, clustering, and article recommendations turn topics into analysis.",
            "#F4D9D0",
        ),
        (
            0.37,
            0.20,
            "Visualization Layer",
            "Streamlit, Plotly, Matplotlib, and WordCloud provide interactive charts and presentation visuals.",
            "#E7E0F3",
        ),
        (
            0.69,
            0.20,
            "Persistence and Delivery",
            "Artifacts are stored as CSV and JSON. Optional MongoDB or MySQL backends support operational storage.",
            "#E4F1D5",
        ),
    ]

    for x_position, y_position, title, body, color in stack_blocks:
        _draw_block(
            axis,
            x=x_position,
            y=y_position,
            width=0.26,
            height=0.22,
            title=title,
            body=body,
            facecolor=color,
            edgecolor="#355C4D",
            wrap_width=24,
        )

    _draw_block(
        axis,
        x=0.05,
        y=0.03,
        width=0.90,
        height=0.12,
        title="Deployment Note",
        body=(
            "Ready for local CLI runs, Streamlit demos, Docker deployment, evaluation, and optional database persistence."
        ),
        facecolor="#173F35",
        edgecolor="#173F35",
        title_color="#F8F7F4",
        body_color="#F8F7F4",
        title_size=12,
        body_size=9,
        wrap_width=88,
    )

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _draw_block(
    axis: Any,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_color: str = "#122C34",
    body_color: str = "#122C34",
    title_size: int = 14,
    body_size: int = 11,
    wrap_width: int | None = None,
) -> None:
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(panel)
    axis.text(x + 0.02, y + height - 0.04, title, fontsize=title_size, fontweight="bold", color=title_color)

    wrapped_lines = []
    for paragraph in body.splitlines():
        if not paragraph.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=wrap_width or max(18, int(width * 55))))
    axis.text(
        x + 0.02,
        y + height - 0.08,
        "\n".join(wrapped_lines),
        fontsize=body_size,
        color=body_color,
        va="top",
        linespacing=1.4,
    )
