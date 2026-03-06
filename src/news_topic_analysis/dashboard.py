from __future__ import annotations

import os
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

MPL_CONFIG_DIR = Path("artifacts/mplconfig")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

from wordcloud import WordCloud

from .database import DatabaseArtifactStore, DatabaseConfig
from .presentation_assets import ensure_presentation_assets
from .storage import ArtifactStore

COLOR_SEQUENCE = ["#0F4C5C", "#C97B3B", "#6B9080", "#9A031E", "#6D597A", "#8D99AE"]
TEXT_COLOR = "#102A43"
MUTED_TEXT_COLOR = "#51657A"
GRID_COLOR = "rgba(16, 42, 67, 0.12)"


def _inject_dashboard_styles() -> None:
    st.markdown(
        """
<style>
:root {
    --ink: #102A43;
    --muted: #51657A;
    --forest: #0F4C5C;
    --bronze: #C97B3B;
    --sage: #6B9080;
    --card: rgba(255, 250, 243, 0.92);
    --line: rgba(16, 42, 67, 0.12);
    --shadow: 0 18px 48px rgba(16, 42, 67, 0.08);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(15, 76, 92, 0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(201, 123, 59, 0.14), transparent 26%),
        linear-gradient(180deg, #F8F4EC 0%, #F4EFE6 52%, #F7F3EB 100%);
    color: var(--ink);
}

html, body, [class*="css"] {
    color: var(--ink);
    font-family: "Trebuchet MS", "Gill Sans", "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
    color: var(--ink);
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
    letter-spacing: -0.02em;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1480px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255, 248, 238, 0.96), rgba(237, 229, 214, 0.94));
    border-right: 1px solid var(--line);
}

[data-baseweb="tab-list"] {
    gap: 0.5rem;
}

button[data-baseweb="tab"] {
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(255, 250, 243, 0.78);
    color: var(--muted);
    font-weight: 700;
    padding: 0.55rem 1rem;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, rgba(15, 76, 92, 0.14), rgba(201, 123, 59, 0.18));
    color: var(--ink);
    border-color: rgba(15, 76, 92, 0.22);
}

div[data-testid="stMetric"],
div[data-testid="stPlotlyChart"],
div[data-testid="stImage"],
div[data-testid="stAlert"],
div[data-testid="stDataFrame"] {
    border-radius: 24px;
    border: 1px solid var(--line);
    background: var(--card);
    box-shadow: var(--shadow);
}

div[data-testid="stMetric"] {
    padding: 0.9rem 1.1rem;
}

div[data-testid="stMetric"] label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    font-size: 0.74rem;
}

div[data-testid="stMetricValue"] {
    color: var(--ink);
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
}

.hero-panel,
.info-panel,
.insight-card,
.section-panel {
    border-radius: 28px;
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
}

.hero-panel {
    padding: 1.6rem 1.7rem;
    background:
        linear-gradient(135deg, rgba(15, 76, 92, 0.94), rgba(26, 84, 100, 0.88)),
        linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0));
    color: #F9F5EE;
}

.hero-kicker {
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 800;
    opacity: 0.84;
}

.hero-title {
    margin-top: 0.4rem;
    font-size: 2.35rem;
    line-height: 1.02;
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
}

.hero-copy {
    margin-top: 0.9rem;
    max-width: 62ch;
    line-height: 1.7;
    color: rgba(249, 245, 238, 0.88);
}

.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.hero-chip {
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.16);
    background: rgba(255, 255, 255, 0.1);
    font-size: 0.83rem;
}

.info-panel {
    padding: 1.2rem 1.25rem;
    background: linear-gradient(180deg, rgba(255, 252, 246, 0.96), rgba(248, 240, 228, 0.96));
}

.info-label {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    font-weight: 800;
}

.info-value {
    margin-top: 0.2rem;
    font-size: 1.28rem;
    font-weight: 800;
    color: var(--ink);
}

.info-note {
    margin-top: 0.3rem;
    color: var(--muted);
    line-height: 1.55;
}

.section-intro {
    margin: 0.25rem 0 1rem 0;
}

.section-kicker {
    color: var(--bronze);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 800;
}

.section-title {
    margin-top: 0.2rem;
    font-size: 1.6rem;
    color: var(--ink);
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
}

.section-copy {
    margin-top: 0.35rem;
    color: var(--muted);
    max-width: 74ch;
    line-height: 1.65;
}

.insight-card {
    min-height: 150px;
    padding: 1rem 1.05rem;
    background: rgba(255, 250, 243, 0.94);
    border-left: 6px solid var(--forest);
}

.insight-card.bronze {
    border-left-color: var(--bronze);
}

.insight-card.sage {
    border-left-color: var(--sage);
}

.insight-label {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    font-weight: 800;
}

.insight-value {
    margin-top: 0.45rem;
    color: var(--ink);
    font-size: 1.95rem;
    line-height: 1.05;
    font-weight: 800;
    font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
}

.insight-note {
    margin-top: 0.55rem;
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.92rem;
}

.section-panel {
    padding: 1.2rem 1.25rem;
    background: rgba(255, 250, 243, 0.88);
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    font-size: 0.74rem;
    font-weight: 800;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _build_wordcloud(text: str) -> object | None:
    if not text.strip():
        return None
    return WordCloud(
        width=1200,
        height=600,
        background_color="#FFF9F2",
        colormap="cividis",
        collocations=False,
    ).generate(text)


def _technology_usage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "Data Collection",
                "technology": "requests + feedparser",
                "used_for": "HTTP retrieval and RSS ingestion from BBC, CNN, and other feeds",
                "where_used": "collectors.py",
            },
            {
                "stage": "Data Collection",
                "technology": "BeautifulSoup",
                "used_for": "HTML parsing and article-body extraction from scraped pages",
                "where_used": "collectors.py",
            },
            {
                "stage": "Preprocessing",
                "technology": "pandas",
                "used_for": "Dataset shaping, text assembly, and artifact table generation",
                "where_used": "preprocessing.py, storage.py, dashboard.py",
            },
            {
                "stage": "Preprocessing",
                "technology": "NLTK / spaCy",
                "used_for": "Tokenization, stopword removal, and lemmatization",
                "where_used": "preprocessing.py",
            },
            {
                "stage": "Topic Modeling",
                "technology": "scikit-learn",
                "used_for": "Vectorization plus LDA and NMF topic extraction",
                "where_used": "topic_modeling.py",
            },
            {
                "stage": "Topic Modeling",
                "technology": "BERTopic",
                "used_for": "Optional embedding-based topic modeling backend",
                "where_used": "topic_modeling.py",
            },
            {
                "stage": "Analytics",
                "technology": "pandas + scikit-learn",
                "used_for": "Trend tables, clustering projection, cosine similarity, and evaluation metrics",
                "where_used": "analytics.py, evaluation.py",
            },
            {
                "stage": "Visualization",
                "technology": "Streamlit + Plotly",
                "used_for": "Executive dashboard, interactive charts, and article exploration",
                "where_used": "dashboard.py",
            },
            {
                "stage": "Visualization",
                "technology": "Matplotlib + WordCloud",
                "used_for": "Presentation graphics and topic word cloud generation",
                "where_used": "presentation_assets.py, dashboard.py",
            },
            {
                "stage": "Persistence",
                "technology": "SQLAlchemy / PyMongo",
                "used_for": "Optional persistence to MySQL-compatible SQL stores and MongoDB",
                "where_used": "database.py",
            },
            {
                "stage": "Application",
                "technology": "Python",
                "used_for": "CLI entrypoints, pipeline orchestration, and dashboard startup",
                "where_used": "streamlit_app.py, cli.py, pipeline.py",
            },
        ]
    )


def _style_figure(fig: Any, title: str, height: int = 420) -> Any:
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 249, 242, 0.88)",
        colorway=COLOR_SEQUENCE,
        font=dict(family="Trebuchet MS, Gill Sans, Segoe UI, sans-serif", color=TEXT_COLOR, size=14),
        margin=dict(l=20, r=20, t=72, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
        ),
        hoverlabel=dict(bgcolor="#FFF9F2", font_color=TEXT_COLOR),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT_COLOR))
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, tickfont=dict(color=MUTED_TEXT_COLOR))
    return fig


def _format_timestamp(value: Any) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return str(value or "n/a")
    return timestamp.strftime("%d %b %Y, %H:%M UTC")


def _metric_card(label: str, value: str, note: str, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return f"""
<div class="insight-card{tone_class}">
    <div class="insight-label">{escape(label)}</div>
    <div class="insight-value">{escape(value)}</div>
    <div class="insight-note">{escape(note)}</div>
</div>
"""


def _info_panel(label: str, value: str, note: str) -> str:
    return f"""
<div class="info-panel">
    <div class="info-label">{escape(label)}</div>
    <div class="info-value">{escape(value)}</div>
    <div class="info-note">{escape(note)}</div>
</div>
"""


def _section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="section-intro">
    <div class="section-kicker">{escape(kicker)}</div>
    <div class="section-title">{escape(title)}</div>
    <div class="section-copy">{escape(copy)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_card_row(cards: list[dict[str, str]]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=True):
        column.markdown(
            _metric_card(
                label=card["label"],
                value=card["value"],
                note=card["note"],
                tone=card.get("tone", ""),
            ),
            unsafe_allow_html=True,
        )


def _safe_top_value(series: pd.Series, fallback: str = "n/a") -> str:
    if series.empty:
        return fallback
    return str(series.index[0])


def load_dashboard_payload(
    storage_mode: str,
    artifact_dir: str | Path,
    db_backend: str | None = None,
    db_uri: str | None = None,
    db_name: str = "news_topic_analysis",
    db_prefix: str = "news_topic_analysis",
    run_id: str | None = None,
) -> dict[str, Any]:
    if storage_mode == "database":
        if not db_backend or not db_uri:
            raise ValueError("Database mode requires both a backend and a database URI.")
        return DatabaseArtifactStore(
            DatabaseConfig(
                backend=db_backend,
                uri=db_uri,
                name=db_name,
                prefix=db_prefix,
            )
        ).load(run_id=run_id or None)
    return ArtifactStore.load(artifact_dir)


def _build_executive_cards(
    articles: pd.DataFrame,
    topic_info: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    emerging_topics: pd.DataFrame,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dominant_domain = _safe_top_value(articles["predicted_domain"].value_counts(), "n/a")
    top_source = _safe_top_value(articles["source"].value_counts(), "n/a")
    largest_topic = "n/a"
    if not topic_info.empty:
        largest_topic = str(topic_info.sort_values("size", ascending=False).iloc[0]["topic_label"])

    if not evaluation_summary.empty and "accuracy" in evaluation_summary.columns:
        accuracy_value = f"{float(evaluation_summary.iloc[0]['accuracy']):.2f}"
        accuracy_note = "Domain classification accuracy on labeled articles"
    else:
        accuracy_value = "n/a"
        accuracy_note = "No labeled evaluation set in this run"

    momentum_value = "n/a"
    momentum_note = "Momentum requires multiple time periods"
    if not emerging_topics.empty:
        lead_momentum = emerging_topics.sort_values(["delta", "lift"], ascending=False).iloc[0]
        momentum_value = str(lead_momentum["topic_label"])
        momentum_note = f"Delta {float(lead_momentum['delta']):.2f} across recent periods"

    primary_cards = [
        {
            "label": "Articles Analyzed",
            "value": f"{len(articles)}",
            "note": "Current run coverage across all collected sources",
            "tone": "forest",
        },
        {
            "label": "Topics Discovered",
            "value": f"{int(topic_info['topic_id'].nunique())}",
            "note": "Distinct topic groups produced by the active model",
            "tone": "bronze",
        },
        {
            "label": "Sources Covered",
            "value": f"{int(articles['source'].nunique())}",
            "note": f"Most represented source: {top_source}",
            "tone": "sage",
        },
        {
            "label": "Evaluation Accuracy",
            "value": accuracy_value,
            "note": accuracy_note,
            "tone": "forest",
        },
    ]

    secondary_cards = [
        {
            "label": "Lead Topic",
            "value": largest_topic,
            "note": "Largest topic cluster in the current run",
            "tone": "bronze",
        },
        {
            "label": "Dominant Domain",
            "value": dominant_domain,
            "note": "Most frequent predicted domain across the corpus",
            "tone": "sage",
        },
        {
            "label": "Strongest Momentum",
            "value": momentum_value,
            "note": momentum_note,
            "tone": "forest",
        },
        {
            "label": "Recommendation Links",
            "value": f"{len(recommendations)}",
            "note": "Similarity links available for article exploration",
            "tone": "bronze",
        },
    ]
    return primary_cards, secondary_cards


def _render_hero(
    metadata: dict[str, Any],
    articles: pd.DataFrame,
    topic_info: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    collection_summary: dict[str, Any],
) -> None:
    failed_targets = int(collection_summary.get("status_counts", {}).get("error", 0))
    evaluated_text = "No labeled evaluation data"
    if not evaluation_summary.empty:
        evaluated_text = f"Accuracy {float(evaluation_summary.iloc[0]['accuracy']):.2f} on labeled articles"

    lead_topic = "n/a"
    if not topic_info.empty:
        lead_topic = str(topic_info.sort_values("size", ascending=False).iloc[0]["topic_label"])

    hero_left, hero_right = st.columns((3, 2))
    hero_left.markdown(
        f"""
<div class="hero-panel">
    <div class="hero-kicker">Editorial Intelligence Workspace</div>
    <div class="hero-title">AI-Based Topic Modeling and Trend Analysis</div>
    <div class="hero-copy">
        This dashboard turns high-volume news coverage into an executive view of themes, momentum, topic
        relationships, and recommendation paths. It is designed to support newsroom monitoring, research analysis,
        and professional final-year-project presentation.
    </div>
    <div class="hero-chips">
        <span class="hero-chip">Model: {escape(str(metadata.get("model_name", "n/a")).upper())}</span>
        <span class="hero-chip">Source Mode: {escape(str(metadata.get("data_source", "n/a")))}</span>
        <span class="hero-chip">Generated: {escape(_format_timestamp(metadata.get("generated_at", "n/a")))}</span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    hero_right.markdown(
        _info_panel(
            label="Run Profile",
            value=f"{len(articles)} articles | {int(articles['source'].nunique())} sources",
            note=(
                f"Lead topic: {lead_topic}. "
                f"{evaluated_text}. "
                f"{failed_targets} collector target(s) failed in this run."
            ),
        ),
        unsafe_allow_html=True,
    )
    hero_right.markdown(
        _info_panel(
            label="Operational Readout",
            value=str(metadata.get("run_id", "n/a")),
            note=(
                f"Discovered {int(topic_info['topic_id'].nunique())} topic groups and "
                f"{int(articles['predicted_domain'].nunique())} predicted domains."
            ),
        ),
        unsafe_allow_html=True,
    )


def _render_executive_summary(
    articles: pd.DataFrame,
    topic_info: pd.DataFrame,
    trends: pd.DataFrame,
    emerging_topics: pd.DataFrame,
    recommendations: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
) -> None:
    _section_intro(
        "Executive Overview",
        "High-Level Performance and Topic Signals",
        "This section summarizes the current run with executive cards, distribution views, and the strongest topic momentum signals.",
    )

    primary_cards, secondary_cards = _build_executive_cards(
        articles=articles,
        topic_info=topic_info,
        evaluation_summary=evaluation_summary,
        recommendations=recommendations,
        emerging_topics=emerging_topics,
    )
    _render_card_row(primary_cards)
    _render_card_row(secondary_cards)

    overview_left, overview_right = st.columns((2, 1))
    topic_distribution = px.bar(
        topic_info.sort_values("size", ascending=False),
        x="topic_label",
        y="size",
        color="topic_label",
    )
    overview_left.plotly_chart(_style_figure(topic_distribution, "Topic Distribution", 420), use_container_width=True)

    domain_distribution = px.pie(
        articles,
        names="predicted_domain",
        hole=0.45,
    )
    overview_right.plotly_chart(
        _style_figure(domain_distribution, "Predicted Domain Mix", 420),
        use_container_width=True,
    )

    if not trends.empty:
        trend_chart = px.line(
            trends,
            x="published_date",
            y="article_count",
            color="topic_label",
            markers=True,
        )
        st.plotly_chart(_style_figure(trend_chart, "Topic Trends Over Time", 460), use_container_width=True)
    else:
        st.info("Trend lines require at least one valid published date.")

    momentum_left, momentum_right = st.columns((3, 2))
    if not emerging_topics.empty:
        momentum_frame = emerging_topics.head(8).sort_values("delta", ascending=True)
        momentum_chart = px.bar(
            momentum_frame,
            x="delta",
            y="topic_label",
            orientation="h",
            color="lift",
            color_continuous_scale=["#E8D9C5", "#C97B3B", "#0F4C5C"],
        )
        momentum_left.plotly_chart(
            _style_figure(momentum_chart, "Emerging Topic Momentum", 390),
            use_container_width=True,
        )
    else:
        momentum_left.info("Not enough time periods are available to estimate topic momentum yet.")

    source_mix = articles["source"].value_counts().rename_axis("source").reset_index(name="article_count")
    source_chart = px.bar(source_mix, x="source", y="article_count", color="source")
    momentum_right.plotly_chart(_style_figure(source_chart, "Source Coverage", 390), use_container_width=True)


def _render_topic_lab(
    articles: pd.DataFrame,
    topic_info: pd.DataFrame,
    emerging_topics: pd.DataFrame,
    topic_relationships: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> None:
    _section_intro(
        "Topic Intelligence",
        "Deep Dive Into Topic Clusters and Recommendations",
        "Use this workspace to inspect individual topics, review clustered articles, compare related topics, and trace recommendation links between articles.",
    )

    selector_left, selector_right = st.columns((2, 1))
    selected_topic = selector_left.selectbox("Topic focus", topic_info["topic_label"].tolist())
    selected_topic_info = topic_info[topic_info["topic_label"] == selected_topic].iloc[0]
    topic_articles = articles[articles["topic_label"] == selected_topic].copy()
    topic_articles["published_at"] = pd.to_datetime(topic_articles["published_at"], utc=True).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    selector_right.markdown(
        _info_panel(
            label="Topic Profile",
            value=str(selected_topic),
            note=(
                f"Size: {int(selected_topic_info['size'])} articles. "
                f"Dominant domain: {selected_topic_info['dominant_domain']}. "
                f"Keywords: {selected_topic_info['keywords']}."
            ),
        ),
        unsafe_allow_html=True,
    )

    detail_left, detail_right = st.columns((3, 2))
    detail_left.dataframe(
        topic_articles[["published_at", "source", "title", "predicted_domain", "topic_score"]]
        .rename(
            columns={
                "published_at": "Published",
                "source": "Source",
                "title": "Title",
                "predicted_domain": "Domain",
                "topic_score": "Score",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    wordcloud_text = " ".join(topic_articles["processed_text"].fillna("").astype(str).tolist())
    cloud = _build_wordcloud(wordcloud_text)
    if cloud is not None:
        detail_right.image(cloud.to_array(), use_container_width=True)
    else:
        detail_right.info("Not enough text is available to generate a word cloud for this topic.")

    support_left, support_right = st.columns(2)
    related_topics = topic_relationships[
        (topic_relationships["topic_a_label"] == selected_topic) | (topic_relationships["topic_b_label"] == selected_topic)
    ].copy()
    if related_topics.empty:
        support_left.info("No related-topic comparison is available for the selected topic.")
    else:
        support_left.dataframe(
            related_topics.head(8).rename(
                columns={
                    "topic_a_label": "Topic A",
                    "topic_b_label": "Topic B",
                    "similarity": "Similarity",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    topic_source_mix = topic_articles["source"].value_counts().rename_axis("source").reset_index(name="article_count")
    source_chart = px.bar(topic_source_mix, x="source", y="article_count", color="source")
    support_right.plotly_chart(_style_figure(source_chart, "Selected Topic by Source", 360), use_container_width=True)

    st.subheader("Article Clusters")
    cluster_chart = px.scatter(
        articles,
        x="cluster_x",
        y="cluster_y",
        color="topic_label",
        symbol="predicted_domain",
        hover_data=["title", "source", "published_at"],
    )
    st.plotly_chart(_style_figure(cluster_chart, "Article Clusters by Topic", 500), use_container_width=True)

    recommendation_left, recommendation_right = st.columns((2, 1))
    article_options = (
        articles[["article_id", "title", "source"]]
        .drop_duplicates()
        .assign(label=lambda frame: frame["title"] + " [" + frame["source"] + "]")
    )
    selected_label = recommendation_left.selectbox("Recommendation source article", article_options["label"].tolist())
    selected_article_id = article_options.loc[article_options["label"] == selected_label, "article_id"].iloc[0]
    article_recommendations = recommendations[recommendations["article_id"] == selected_article_id]

    if article_recommendations.empty:
        recommendation_left.info("No recommendations are available for the selected article.")
    else:
        recommendation_left.dataframe(
            article_recommendations[["recommended_title", "similarity_score", "topic_match"]]
            .rename(
                columns={
                    "recommended_title": "Recommended Article",
                    "similarity_score": "Similarity",
                    "topic_match": "Same Topic",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if emerging_topics.empty:
        recommendation_right.info("Emerging-topic comparisons are not available for this run.")
    else:
        recommendation_right.dataframe(
            emerging_topics.head(10).rename(
                columns={
                    "topic_label": "Topic",
                    "recent_mean": "Recent Mean",
                    "baseline_mean": "Baseline Mean",
                    "delta": "Delta",
                    "lift": "Lift",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_run_diagnostics(metadata: dict[str, Any]) -> None:
    st.subheader("Collection and Persistence")

    collection_summary = metadata.get("collection_summary", {})
    reports = pd.DataFrame(collection_summary.get("reports", []))
    status_counts = collection_summary.get("status_counts", {})
    diagnostic_cards = [
        {
            "label": "Collected Articles",
            "value": str(collection_summary.get("total_articles", 0)),
            "note": "Items retained after ingestion",
            "tone": "forest",
        },
        {
            "label": "Collector Targets",
            "value": str(collection_summary.get("collector_count", 0)),
            "note": "Configured source endpoints in this run",
            "tone": "bronze",
        },
        {
            "label": "Successful Targets",
            "value": str(status_counts.get("success", 0)),
            "note": "Collector targets that completed successfully",
            "tone": "sage",
        },
        {
            "label": "Failed Targets",
            "value": str(status_counts.get("error", 0)),
            "note": "Collector targets that returned an error",
            "tone": "bronze",
        },
    ]
    _render_card_row(diagnostic_cards)

    diagnostics_left, diagnostics_right = st.columns((3, 2))
    if reports.empty:
        diagnostics_left.info("Collector diagnostics are not available for this run.")
    else:
        diagnostics_left.dataframe(
            reports.rename(
                columns={
                    "collector": "Collector",
                    "source": "Source",
                    "target": "Target",
                    "status": "Status",
                    "article_count": "Articles",
                    "error_message": "Error",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    database_summary = metadata.get("database", {})
    if database_summary and database_summary.get("enabled"):
        summary_frame = pd.DataFrame(
            [{"property": key.replace("_", " ").title(), "value": str(value)} for key, value in database_summary.items()]
        )
        diagnostics_right.dataframe(summary_frame, use_container_width=True, hide_index=True)
    else:
        diagnostics_right.info("This run was not persisted to a database.")


def _render_evaluation_section(
    articles: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    domain_performance: pd.DataFrame,
    domain_confusion: pd.DataFrame,
    split_performance: pd.DataFrame,
    presentation_metrics: pd.DataFrame,
    presentation_report: str,
) -> None:
    st.subheader("Evaluation and Delivery Assets")

    if evaluation_summary.empty:
        st.info("No labeled evaluation data is available for this run.")
        return

    summary = evaluation_summary.iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Labeled Articles", int(summary["labeled_articles"]))
    metric_columns[1].metric("Accuracy", float(summary["accuracy"]))
    metric_columns[2].metric("Macro F1", float(summary["macro_f1"]))
    metric_columns[3].metric("Weighted F1", float(summary["weighted_f1"]))

    chart_left, chart_right = st.columns(2)
    if not domain_performance.empty:
        performance_chart = px.bar(
            domain_performance.melt(
                id_vars=["domain", "support"],
                value_vars=["precision", "recall", "f1_score"],
                var_name="metric",
                value_name="score",
            ),
            x="domain",
            y="score",
            color="metric",
            barmode="group",
        )
        chart_left.plotly_chart(
            _style_figure(performance_chart, "Per-Domain Evaluation Metrics", 420),
            use_container_width=True,
        )

    if not domain_confusion.empty:
        confusion_pivot = domain_confusion.pivot(
            index="expected_domain",
            columns="predicted_domain",
            values="count",
        ).fillna(0)
        confusion_chart = px.imshow(
            confusion_pivot,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=["#FFF4E6", "#C97B3B", "#0F4C5C"],
        )
        chart_right.plotly_chart(
            _style_figure(confusion_chart, "Expected vs Predicted Domain Confusion", 420),
            use_container_width=True,
        )

    table_left, table_right = st.columns(2)
    if not split_performance.empty:
        split_chart = px.bar(
            split_performance,
            x="split",
            y="accuracy",
            text="labeled_articles",
            color="split",
        )
        table_left.plotly_chart(_style_figure(split_chart, "Accuracy by Dataset Split", 360), use_container_width=True)
    else:
        table_left.info("Split-level evaluation is not available.")

    expected_domain_counts = (
        articles["expected_domain"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().value_counts()
    )
    if not expected_domain_counts.empty:
        expected_chart = px.pie(
            names=expected_domain_counts.index,
            values=expected_domain_counts.values,
            hole=0.45,
        )
        table_right.plotly_chart(
            _style_figure(expected_chart, "Expected Domain Distribution", 360),
            use_container_width=True,
        )
    else:
        table_right.info("Expected-domain labels are not available.")

    report_left, report_right = st.columns((2, 1))
    if presentation_report:
        report_left.markdown("#### Presentation Summary")
        report_left.markdown(presentation_report)
    else:
        report_left.info("Presentation report is not available.")

    if not presentation_metrics.empty:
        report_right.dataframe(
            presentation_metrics.rename(
                columns={
                    "metric_group": "Group",
                    "metric_name": "Metric",
                    "metric_value": "Value",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        report_right.info("Presentation metrics are not available.")


def _render_project_brief(
    metadata: dict[str, Any],
    articles: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    asset_paths: dict[str, Path],
) -> None:
    _section_intro(
        "Project Brief",
        "Professional Context, Workflow, and Technology Map",
        "This section explains what the project is for, how the pipeline works, and exactly which technologies are used in each layer of the solution.",
    )

    overview_left, overview_right = st.columns((3, 2))
    overview_left.markdown(
        """
<div class="section-panel">
    <div class="eyebrow">Project Value</div>
    <p>
        The platform converts large volumes of news text into actionable intelligence. It automatically discovers hidden
        themes, tracks which topics are accelerating over time, highlights related story clusters, and generates
        recommendation links between similar articles.
    </p>
    <p>
        It is relevant for newsroom operations, media research, market monitoring, policy analysis, and professional
        academic demonstration because it combines data ingestion, NLP, machine learning, analytics, and visualization
        in one workflow.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    accuracy_value = "n/a"
    if not evaluation_summary.empty and "accuracy" in evaluation_summary.columns:
        accuracy_value = f"{float(evaluation_summary.iloc[0]['accuracy']):.2f}"

    overview_right.markdown(
        _info_panel(
            label="Run Context",
            value=f"{str(metadata.get('model_name', 'n/a')).upper()} on {metadata.get('data_source', 'n/a')}",
            note=(
                f"Articles: {len(articles)} | Sources: {int(articles['source'].nunique())} | "
                f"Topics: {int(metadata.get('num_topics', 0))} | Accuracy: {accuracy_value}"
            ),
        ),
        unsafe_allow_html=True,
    )

    image_left, image_right = st.columns(2)
    image_left.image(str(asset_paths["project_overview"]), caption="Auto-generated project overview graphic")
    image_right.image(str(asset_paths["technology_stack"]), caption="Auto-generated technology stack graphic")

    workflow_left, workflow_right = st.columns((2, 3))
    workflow_left.markdown(
        """
<div class="section-panel">
    <div class="eyebrow">Workflow</div>
    <ol>
        <li>Collect articles from RSS feeds, HTML pages, or local CSV datasets.</li>
        <li>Clean, tokenize, and normalize text for modeling.</li>
        <li>Discover topics with LDA, NMF, or BERTopic.</li>
        <li>Measure trends, relationships, clusters, and recommendation links.</li>
        <li>Present executive and technical insights in one dashboard.</li>
    </ol>
</div>
        """,
        unsafe_allow_html=True,
    )
    workflow_right.dataframe(
        _technology_usage_frame(),
        use_container_width=True,
        hide_index=True,
        column_config={
            "stage": "Stage",
            "technology": "Technology",
            "used_for": "Used For",
            "where_used": "Where Used",
        },
    )


def run_dashboard(
    default_artifact_dir: str | Path = "artifacts/latest",
    default_storage_mode: str = "artifacts",
    default_db_backend: str | None = None,
    default_db_uri: str | None = None,
    default_db_name: str = "news_topic_analysis",
    default_db_prefix: str = "news_topic_analysis",
    default_run_id: str | None = None,
) -> None:
    st.set_page_config(page_title="News Topic Analysis", layout="wide")
    _inject_dashboard_styles()

    storage_mode = st.sidebar.selectbox(
        "Dashboard Source",
        ["artifacts", "database"],
        index=0 if default_storage_mode == "artifacts" else 1,
    )
    artifact_dir = str(default_artifact_dir)
    db_backend = default_db_backend or ""
    db_uri = default_db_uri or ""
    db_name = default_db_name
    db_prefix = default_db_prefix
    run_id = default_run_id or ""

    st.sidebar.markdown("### Workspace")
    st.sidebar.caption("Professional analytics view for topic discovery, trend monitoring, and presentation output.")

    if storage_mode == "artifacts":
        artifact_dir = st.sidebar.text_input("Artifact Directory", str(default_artifact_dir))
    else:
        db_backend = st.sidebar.selectbox(
            "Database Backend",
            ["mysql", "mongodb"],
            index=0 if (default_db_backend or "mysql") == "mysql" else 1,
        )
        db_uri = st.sidebar.text_input("DB URI", db_uri, type="password")
        db_name = st.sidebar.text_input("DB Name", db_name)
        db_prefix = st.sidebar.text_input("DB Prefix", db_prefix)
        run_id = st.sidebar.text_input("Run ID", run_id, help="Leave blank to load the latest run.")

    try:
        payload = load_dashboard_payload(
            storage_mode=storage_mode,
            artifact_dir=artifact_dir,
            db_backend=db_backend or None,
            db_uri=db_uri or None,
            db_name=db_name,
            db_prefix=db_prefix,
            run_id=run_id or None,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    articles = payload["articles"]
    topic_info = payload["topic_info"]
    trends = payload["trends"]
    emerging_topics = payload["emerging_topics"]
    topic_relationships = payload["topic_relationships"]
    recommendations = payload["recommendations"]
    evaluation_summary = payload.get("evaluation_summary", pd.DataFrame())
    domain_performance = payload.get("domain_performance", pd.DataFrame())
    domain_confusion = payload.get("domain_confusion", pd.DataFrame())
    split_performance = payload.get("split_performance", pd.DataFrame())
    presentation_metrics = payload.get("presentation_metrics", pd.DataFrame())
    metadata = payload["metadata"]
    presentation_report = payload.get("presentation_report", "")

    if articles.empty or topic_info.empty:
        st.error("No artifacts found. Run the pipeline first with `news-topics run`.")
        st.stop()

    asset_paths = ensure_presentation_assets(
        artifact_dir=payload.get("artifact_dir", artifact_dir),
        metadata=metadata,
        articles=articles,
        evaluation_summary=evaluation_summary,
    )

    collection_summary = metadata.get("collection_summary", {})
    st.caption(
        " | ".join(
            [
                f"Run: {metadata.get('run_id', 'n/a')}",
                f"Model: {metadata.get('model_name', 'n/a')}",
                f"Source Mode: {metadata.get('data_source', 'n/a')}",
                f"Generated: {_format_timestamp(metadata.get('generated_at', 'n/a'))}",
            ]
        )
    )

    _render_hero(
        metadata=metadata,
        articles=articles,
        topic_info=topic_info,
        evaluation_summary=evaluation_summary,
        collection_summary=collection_summary,
    )

    tabs = st.tabs(
        [
            "Executive Overview",
            "Topic Intelligence",
            "Operations Review",
            "Project Brief",
        ]
    )

    with tabs[0]:
        _render_executive_summary(
            articles=articles,
            topic_info=topic_info,
            trends=trends,
            emerging_topics=emerging_topics,
            recommendations=recommendations,
            evaluation_summary=evaluation_summary,
        )

    with tabs[1]:
        _render_topic_lab(
            articles=articles,
            topic_info=topic_info,
            emerging_topics=emerging_topics,
            topic_relationships=topic_relationships,
            recommendations=recommendations,
        )

    with tabs[2]:
        _section_intro(
            "Operations Review",
            "Diagnostics, Quality, and Delivery Readiness",
            "This area tracks ingestion health, database persistence status, evaluation results, and presentation-ready output tables.",
        )
        _render_run_diagnostics(metadata)
        _render_evaluation_section(
            articles=articles,
            evaluation_summary=evaluation_summary,
            domain_performance=domain_performance,
            domain_confusion=domain_confusion,
            split_performance=split_performance,
            presentation_metrics=presentation_metrics,
            presentation_report=presentation_report,
        )

    with tabs[3]:
        _render_project_brief(
            metadata=metadata,
            articles=articles,
            evaluation_summary=evaluation_summary,
            asset_paths=asset_paths,
        )
