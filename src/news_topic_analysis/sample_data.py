from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .models import NewsArticle


def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _article_id(source: str, title: str, published_at: str) -> str:
    raw = f"{source}::{title}::{published_at}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return project_root() / "data"


def raw_data_root() -> Path:
    return data_root() / "raw"


def bundled_dataset_paths() -> dict[str, Path]:
    root = raw_data_root()
    return {
        "demo": root / "news_articles_demo.csv",
        "extended": root / "news_articles_extended.csv",
        "evaluation": root / "news_articles_evaluation.csv",
    }


def default_csv_dataset_path() -> Path:
    return bundled_dataset_paths()["extended"]


def load_articles_from_csv(csv_path: str | Path) -> list[NewsArticle]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    frame = pd.read_csv(path)
    required_columns = {"title", "content", "source", "published_at"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset `{path}` is missing required columns: {', '.join(missing_columns)}")

    articles: list[NewsArticle] = []
    for index, row in frame.iterrows():
        published_at = pd.to_datetime(row["published_at"], utc=True, errors="coerce")
        if pd.isna(published_at):
            raise ValueError(f"Invalid `published_at` value at row {index} in `{path}`.")

        title = str(row["title"]).strip()
        source = str(row["source"]).strip()
        content = str(row["content"]).strip()
        raw_published_at = str(row["published_at"])
        article_id = str(row["article_id"]).strip() if "article_id" in frame.columns and pd.notna(row["article_id"]) else ""
        if not article_id:
            article_id = _article_id(source, title, raw_published_at)

        url = (
            str(row["url"]).strip()
            if "url" in frame.columns and pd.notna(row["url"]) and str(row["url"]).strip()
            else f"https://local.dataset/{_slugify(title)}"
        )
        language = (
            str(row["language"]).strip()
            if "language" in frame.columns and pd.notna(row["language"]) and str(row["language"]).strip()
            else "en"
        )
        expected_domain = (
            str(row["expected_domain"]).strip()
            if "expected_domain" in frame.columns and pd.notna(row["expected_domain"])
            else ""
        )
        dataset_split = (
            str(row["split"]).strip()
            if "split" in frame.columns and pd.notna(row["split"])
            else ""
        )

        articles.append(
            NewsArticle(
                article_id=article_id,
                title=title,
                content=content,
                source=source,
                published_at=published_at.to_pydatetime(),
                url=url,
                language=language,
                expected_domain=expected_domain,
                dataset_split=dataset_split,
            )
        )
    return articles


def build_demo_articles() -> list[NewsArticle]:
    demo_dataset_path = bundled_dataset_paths()["demo"]
    if demo_dataset_path.exists():
        return load_articles_from_csv(demo_dataset_path)

    records = [
        {
            "title": "Parliament backs election finance reform plan",
            "content": "Government lawmakers approved an election finance package after a long parliament debate on campaign funding, voter transparency, coalition strategy, and policy oversight.",
            "source": "BBC Demo",
            "published_at": "2026-02-18T08:00:00+00:00",
        },
        {
            "title": "Cabinet outlines new industrial policy after budget session",
            "content": "Ministers said the industrial policy will support manufacturing, regional jobs, export growth, and infrastructure while opposition leaders demanded stricter accountability from government agencies.",
            "source": "BBC Demo",
            "published_at": "2026-02-19T09:15:00+00:00",
        },
        {
            "title": "Regional leaders push coalition talks ahead of national vote",
            "content": "Party officials accelerated coalition negotiations, election outreach, and legislative messaging as polling tightened before the national vote and parliament reconvened.",
            "source": "Reuters Demo",
            "published_at": "2026-02-21T10:00:00+00:00",
        },
        {
            "title": "Senate committee reviews cyber policy and digital regulation",
            "content": "The senate committee examined cybersecurity rules, digital platform regulation, public data protection, and cross-border policy coordination with technology regulators.",
            "source": "CNN Demo",
            "published_at": "2026-02-24T12:00:00+00:00",
        },
        {
            "title": "National team seals cricket series after dominant chase",
            "content": "The cricket team completed a dominant chase with late boundary hitting, disciplined bowling, and sharp fielding as the match crowd celebrated the series-clinching score.",
            "source": "BBC Demo",
            "published_at": "2026-02-18T14:30:00+00:00",
        },
        {
            "title": "Coach praises young squad after dramatic football comeback",
            "content": "The football coach praised the young squad after a dramatic league comeback driven by pressing, midfield control, and a stoppage-time goal that lifted team morale.",
            "source": "CNN Demo",
            "published_at": "2026-02-20T18:10:00+00:00",
        },
        {
            "title": "Analysts track player workload before major tournament",
            "content": "Sports analysts reviewed training data, player workload, injury management, and tournament rotation plans as the coaching staff prepared for knockout matches.",
            "source": "Reuters Demo",
            "published_at": "2026-02-23T07:45:00+00:00",
        },
        {
            "title": "League final draws record audience and sponsorship demand",
            "content": "The league final attracted record audience figures, sponsorship demand, merchandise sales, and global broadcast attention after a high-scoring team performance.",
            "source": "CNN Demo",
            "published_at": "2026-02-26T20:05:00+00:00",
        },
        {
            "title": "AI startup launches diagnostic platform for hospitals",
            "content": "An AI startup launched clinical software for hospitals, combining machine learning, medical imaging support, workflow automation, and analytics for faster diagnosis.",
            "source": "Reuters Demo",
            "published_at": "2026-02-18T06:50:00+00:00",
        },
        {
            "title": "Cloud vendors compete over enterprise automation tools",
            "content": "Technology companies expanded cloud automation tools, developer platforms, cybersecurity monitoring, and software integration features for large enterprise clients.",
            "source": "BBC Demo",
            "published_at": "2026-02-20T11:20:00+00:00",
        },
        {
            "title": "Chipmakers accelerate investment in data center hardware",
            "content": "Chipmakers announced investment in data center hardware, AI accelerators, semiconductor capacity, and efficient software stacks for enterprise inference workloads.",
            "source": "CNN Demo",
            "published_at": "2026-02-22T13:00:00+00:00",
        },
        {
            "title": "University lab partners with startup on robotics software",
            "content": "Researchers partnered with a robotics startup to test autonomous software, simulation pipelines, machine learning models, and industrial inspection workflows.",
            "source": "BBC Demo",
            "published_at": "2026-02-25T15:10:00+00:00",
        },
        {
            "title": "Markets rally as central bank signals slower rate path",
            "content": "Global markets rallied after the central bank signaled a slower interest rate path, lifting equities, bond sentiment, banking stocks, and corporate borrowing confidence.",
            "source": "Reuters Demo",
            "published_at": "2026-02-19T16:00:00+00:00",
        },
        {
            "title": "Retail earnings beat forecasts on strong digital sales",
            "content": "Retail groups posted stronger earnings, citing digital sales, supply chain improvement, consumer demand, and disciplined inventory management during the quarter.",
            "source": "CNN Demo",
            "published_at": "2026-02-21T17:25:00+00:00",
        },
        {
            "title": "Logistics firms warn of freight pressure and export delays",
            "content": "Logistics firms warned that freight pressure, export delays, shipping costs, and warehouse congestion could squeeze profit margins for manufacturers and retailers.",
            "source": "BBC Demo",
            "published_at": "2026-02-24T08:40:00+00:00",
        },
        {
            "title": "Fintech lenders target small business cash-flow gap",
            "content": "Fintech lenders rolled out small business finance products focused on cash-flow forecasting, digital underwriting, repayment flexibility, and growth capital access.",
            "source": "Reuters Demo",
            "published_at": "2026-02-27T09:55:00+00:00",
        },
        {
            "title": "Hospitals scale vaccine outreach in rural districts",
            "content": "Hospitals expanded vaccine outreach, rural clinics, public health education, and preventive care appointments to improve immunization coverage across districts.",
            "source": "BBC Demo",
            "published_at": "2026-02-18T05:35:00+00:00",
        },
        {
            "title": "Researchers report progress in antiviral treatment trial",
            "content": "Medical researchers reported encouraging antiviral trial data, improved patient recovery, laboratory validation, and broader healthcare planning for infectious disease control.",
            "source": "CNN Demo",
            "published_at": "2026-02-22T09:30:00+00:00",
        },
        {
            "title": "Public health dashboard flags seasonal respiratory surge",
            "content": "A public health dashboard flagged a seasonal respiratory surge, prompting hospitals to review staffing, emergency care capacity, medical supplies, and community guidance.",
            "source": "Reuters Demo",
            "published_at": "2026-02-24T19:15:00+00:00",
        },
        {
            "title": "Insurance providers expand telemedicine support benefits",
            "content": "Insurance providers expanded telemedicine support, digital care reimbursement, mental health access, and remote consultation benefits for patients and doctors.",
            "source": "BBC Demo",
            "published_at": "2026-02-26T10:05:00+00:00",
        },
        {
            "title": "Cities brace for heatwave with new water resilience plans",
            "content": "City officials introduced climate resilience plans covering water storage, urban heat mitigation, drought preparation, and emergency energy management ahead of a heatwave.",
            "source": "CNN Demo",
            "published_at": "2026-02-19T06:25:00+00:00",
        },
        {
            "title": "Energy groups invest in offshore wind expansion",
            "content": "Energy groups increased investment in offshore wind, renewable infrastructure, carbon reduction, and climate targets as governments promoted cleaner power systems.",
            "source": "Reuters Demo",
            "published_at": "2026-02-21T11:45:00+00:00",
        },
        {
            "title": "Farmers seek climate support after irregular rainfall",
            "content": "Farmers requested climate support, crop insurance, irrigation upgrades, and weather forecasting tools after irregular rainfall disrupted planting cycles and yields.",
            "source": "BBC Demo",
            "published_at": "2026-02-23T12:35:00+00:00",
        },
        {
            "title": "Coastal engineers test flood barriers before storm season",
            "content": "Coastal engineers tested flood barriers, storm drainage systems, sea-level defenses, and disaster planning scenarios before the next storm season.",
            "source": "CNN Demo",
            "published_at": "2026-02-27T13:20:00+00:00",
        },
    ]

    articles: list[NewsArticle] = []
    for record in records:
        published_at = datetime.fromisoformat(record["published_at"]).astimezone(timezone.utc)
        title = record["title"]
        source = record["source"]
        article_id = _article_id(source, title, record["published_at"])
        articles.append(
            NewsArticle(
                article_id=article_id,
                title=title,
                content=record["content"],
                source=source,
                published_at=published_at,
                url=f"https://demo.local/{_slugify(title)}",
            )
        )
    return articles
