from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, Protocol
from urllib.parse import urljoin, urlparse

from .models import NewsArticle
from .sample_data import (
    build_demo_articles,
    bundled_dataset_paths,
    default_csv_dataset_path,
    load_articles_from_csv,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_RSS_FEEDS: dict[str, str | list[str]] = {
    "BBC": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "http://feeds.bbci.co.uk/news/rss.xml",
    ],
    "CNN": [
        "https://rss.cnn.com/rss/edition.rss",
        "http://rss.cnn.com/rss/edition.rss",
    ],
}

DEFAULT_HTML_SOURCES: dict[str, str] = {
    "Reuters": "https://www.reuters.com/world/",
}


class ArticleCollector(Protocol):
    def collect(self, limit_per_source: int = 20) -> list[NewsArticle]:
        ...

    def collect_with_report(self, limit_per_source: int = 20) -> "CollectionResult":
        ...


@dataclass(slots=True)
class CollectorReport:
    collector: str
    source: str
    target: str
    status: str
    article_count: int
    error_message: str = ""


@dataclass(slots=True)
class CollectionResult:
    articles: list[NewsArticle]
    reports: list[CollectorReport]


def _make_article_id(source: str, title: str, published_at: datetime, url: str) -> str:
    raw = f"{source}::{title}::{published_at.isoformat()}::{url}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)

    try:
        return parsedate_to_datetime(text).astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _html_to_text(value: str) -> str:
    if not value:
        return ""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return value
    return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)


def _deduplicate(articles: Iterable[NewsArticle]) -> list[NewsArticle]:
    unique: dict[str, NewsArticle] = {}
    for article in articles:
        unique[article.article_id] = article
    return sorted(unique.values(), key=lambda article: article.published_at)


def _iter_json_ld_objects(payload: object) -> Iterable[dict[str, object]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_json_ld_objects(item)
        return
    if not isinstance(payload, dict):
        return
    if "@graph" in payload:
        yield from _iter_json_ld_objects(payload["@graph"])
        return
    yield payload


class DemoNewsCollector:
    def collect(self, limit_per_source: int = 20) -> list[NewsArticle]:
        return self.collect_with_report(limit_per_source=limit_per_source).articles

    def collect_with_report(self, limit_per_source: int = 20) -> CollectionResult:
        del limit_per_source
        articles = build_demo_articles()
        return CollectionResult(
            articles=articles,
            reports=[
                CollectorReport(
                    collector=self.__class__.__name__,
                    source="demo",
                    target=str(bundled_dataset_paths()["demo"]),
                    status="success",
                    article_count=len(articles),
                )
            ],
        )


class LocalCSVCollector:
    def __init__(self, csv_path: str | None = None) -> None:
        self.csv_path = csv_path or str(default_csv_dataset_path())

    def collect(self, limit_per_source: int = 20) -> list[NewsArticle]:
        del limit_per_source
        return self.collect_with_report().articles

    def collect_with_report(self, limit_per_source: int = 20) -> CollectionResult:
        del limit_per_source
        articles = load_articles_from_csv(self.csv_path)
        return CollectionResult(
            articles=articles,
            reports=[
                CollectorReport(
                    collector=self.__class__.__name__,
                    source="local-csv",
                    target=str(self.csv_path),
                    status="success",
                    article_count=len(articles),
                )
            ],
        )


class CompositeCollector:
    def __init__(self, collectors: Iterable[ArticleCollector]) -> None:
        self.collectors = list(collectors)

    def collect(self, limit_per_source: int = 20) -> list[NewsArticle]:
        return self.collect_with_report(limit_per_source=limit_per_source).articles

    def collect_with_report(self, limit_per_source: int = 20) -> CollectionResult:
        articles: list[NewsArticle] = []
        reports: list[CollectorReport] = []
        for collector in self.collectors:
            try:
                if hasattr(collector, "collect_with_report"):
                    result = collector.collect_with_report(limit_per_source=limit_per_source)
                    articles.extend(result.articles)
                    reports.extend(result.reports)
                else:
                    collected_articles = collector.collect(limit_per_source=limit_per_source)
                    articles.extend(collected_articles)
                    reports.append(
                        CollectorReport(
                            collector=collector.__class__.__name__,
                            source=collector.__class__.__name__,
                            target="n/a",
                            status="success",
                            article_count=len(collected_articles),
                        )
                    )
            except Exception as exc:
                LOGGER.warning("Collector %s failed: %s", collector.__class__.__name__, exc)
                reports.append(
                    CollectorReport(
                        collector=collector.__class__.__name__,
                        source=collector.__class__.__name__,
                        target="n/a",
                        status="error",
                        article_count=0,
                        error_message=str(exc),
                    )
                )
        return CollectionResult(articles=_deduplicate(articles), reports=reports)


class RSSNewsCollector:
    def __init__(self, feed_urls: dict[str, str | list[str]] | None = None, timeout: int = 10) -> None:
        self.feed_urls = feed_urls or DEFAULT_RSS_FEEDS
        self.timeout = timeout

    def collect(self, limit_per_source: int = 20) -> list[NewsArticle]:
        return self.collect_with_report(limit_per_source=limit_per_source).articles

    def collect_with_report(self, limit_per_source: int = 20) -> CollectionResult:
        try:
            import feedparser
            import requests
        except ImportError as exc:
            raise RuntimeError("RSS collection requires feedparser and requests.") from exc

        headers = {"User-Agent": "news-topic-analysis/0.1"}
        session = requests.Session()
        articles: list[NewsArticle] = []
        reports: list[CollectorReport] = []

        for source, feed_url in self.feed_urls.items():
            starting_count = len(articles)
            candidate_urls = [feed_url] if isinstance(feed_url, str) else list(feed_url)
            last_exception: Exception | None = None
            successful_url = ""

            for candidate_url in candidate_urls:
                try:
                    response = session.get(candidate_url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    feed = feedparser.parse(response.content)

                    for entry in feed.entries[:limit_per_source]:
                        title = str(getattr(entry, "title", "")).strip()
                        if not title:
                            continue

                        link = str(getattr(entry, "link", "")).strip()
                        published_at = _coerce_datetime(
                            getattr(entry, "published", "")
                            or getattr(entry, "updated", "")
                            or getattr(entry, "created", "")
                        )

                        summary = _html_to_text(str(getattr(entry, "summary", "")))
                        content_blocks = getattr(entry, "content", []) or []
                        content_parts = []
                        for block in content_blocks:
                            value = getattr(block, "value", "")
                            content_parts.append(_html_to_text(str(value)))
                        content = " ".join(part for part in content_parts if part).strip() or summary
                        if not content:
                            continue

                        articles.append(
                            NewsArticle(
                                article_id=_make_article_id(source, title, published_at, link),
                                title=title,
                                content=content,
                                source=source,
                                published_at=published_at,
                                url=link,
                            )
                        )

                    successful_url = candidate_url
                    break
                except Exception as exc:
                    last_exception = exc

            if successful_url:
                reports.append(
                    CollectorReport(
                        collector=self.__class__.__name__,
                        source=source,
                        target=successful_url,
                        status="success",
                        article_count=len(articles) - starting_count,
                    )
                )
            else:
                reports.append(
                    CollectorReport(
                        collector=self.__class__.__name__,
                        source=source,
                        target=", ".join(candidate_urls),
                        status="error",
                        article_count=0,
                        error_message=str(last_exception or "unknown RSS error"),
                    )
                )
                LOGGER.warning("RSS source %s failed: %s", source, last_exception)

        return CollectionResult(articles=_deduplicate(articles), reports=reports)


class GenericHTMLCollector:
    def __init__(self, source_urls: dict[str, str] | None = None, timeout: int = 10) -> None:
        self.source_urls = source_urls or DEFAULT_HTML_SOURCES
        self.timeout = timeout

    def collect(self, limit_per_source: int = 20) -> list[NewsArticle]:
        return self.collect_with_report(limit_per_source=limit_per_source).articles

    def collect_with_report(self, limit_per_source: int = 20) -> CollectionResult:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("HTML collection requires requests.") from exc

        session = requests.Session()
        session.headers.update({"User-Agent": "news-topic-analysis/0.1"})
        articles: list[NewsArticle] = []
        reports: list[CollectorReport] = []

        for source, listing_url in self.source_urls.items():
            starting_count = len(articles)
            try:
                listing_response = session.get(listing_url, timeout=self.timeout)
                listing_response.raise_for_status()
                links = self._extract_links(listing_url, listing_response.text)

                for link in links[:limit_per_source]:
                    article = self._extract_article(source, link, session)
                    if article is not None:
                        articles.append(article)

                reports.append(
                    CollectorReport(
                        collector=self.__class__.__name__,
                        source=source,
                        target=listing_url,
                        status="success",
                        article_count=len(articles) - starting_count,
                    )
                )
            except Exception as exc:
                reports.append(
                    CollectorReport(
                        collector=self.__class__.__name__,
                        source=source,
                        target=listing_url,
                        status="error",
                        article_count=0,
                        error_message=str(exc),
                    )
                )
                LOGGER.warning("HTML source %s failed: %s", source, exc)

        return CollectionResult(articles=_deduplicate(articles), reports=reports)

    def _extract_links(self, base_url: str, html: str) -> list[str]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError("HTML collection requires beautifulsoup4.") from exc

        soup = BeautifulSoup(html, "html.parser")
        root_domain = urlparse(base_url).netloc
        blocked_tokens = ("/video/", "/live/", "/pictures/", "/graphics/")
        links: list[str] = []
        seen: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = urljoin(base_url, str(anchor["href"]))
            parsed = urlparse(href)
            if root_domain not in parsed.netloc:
                continue
            if not parsed.path or len(parsed.path.split("/")) < 3:
                continue
            if any(token in parsed.path.lower() for token in blocked_tokens):
                continue
            if href in seen:
                continue
            seen.add(href)
            links.append(href)

        return links

    def _extract_article(self, source: str, url: str, session: object) -> NewsArticle | None:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError("HTML collection requires beautifulsoup4.") from exc

        response = session.get(url, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = ""
        body = ""
        published_at = datetime.now(timezone.utc)

        for script in soup.find_all("script", type="application/ld+json"):
            raw_json = script.string or script.get_text(strip=True)
            if not raw_json:
                continue
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            for candidate in _iter_json_ld_objects(parsed):
                if not title:
                    title = str(candidate.get("headline") or candidate.get("name") or "").strip()
                if not body:
                    body = str(candidate.get("articleBody") or candidate.get("description") or "").strip()
                if "datePublished" in candidate:
                    published_at = _coerce_datetime(candidate["datePublished"])

        if not title:
            heading = soup.find("h1")
            title = heading.get_text(" ", strip=True) if heading is not None else ""

        if not body:
            containers = []
            article_tag = soup.find("article")
            main_tag = soup.find("main")
            if article_tag is not None:
                containers.append(article_tag)
            if main_tag is not None:
                containers.append(main_tag)
            containers.append(soup)

            paragraphs: list[str] = []
            for container in containers:
                paragraphs = [
                    node.get_text(" ", strip=True)
                    for node in container.find_all("p")
                    if len(node.get_text(" ", strip=True)) >= 60
                ]
                if paragraphs:
                    break
            body = " ".join(paragraphs).strip()

        if not title or not body:
            return None

        return NewsArticle(
            article_id=_make_article_id(source, title, published_at, url),
            title=title,
            content=body,
            source=source,
            published_at=published_at,
            url=url,
        )
