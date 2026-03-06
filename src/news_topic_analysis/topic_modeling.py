from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass(slots=True)
class TopicModelOutput:
    articles: pd.DataFrame
    topic_info: pd.DataFrame
    document_topic_matrix: np.ndarray
    document_term_matrix: Any
    topic_term_matrix: np.ndarray
    feature_names: list[str]
    model_name: str
    vectorizer_name: str


class TopicModeler:
    def __init__(
        self,
        model_name: str = "lda",
        num_topics: int = 6,
        top_words: int = 8,
        max_features: int = 3000,
        min_df: int = 1,
        max_df: float = 0.95,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name.lower()
        self.num_topics = num_topics
        self.top_words = top_words
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state

    def fit_transform(self, articles: pd.DataFrame) -> TopicModelOutput:
        if self.model_name == "bertopic":
            return self._fit_bertopic(articles)
        return self._fit_sklearn_model(articles)

    def _fit_sklearn_model(self, articles: pd.DataFrame) -> TopicModelOutput:
        texts = articles["processed_text"].tolist()
        if len(texts) < 2:
            raise ValueError("At least two articles are required for topic modeling.")

        vectorizer_name = "count"
        if self.model_name == "lda":
            vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
            )
            document_term_matrix = vectorizer.fit_transform(texts)
            n_components = max(1, min(self.num_topics, document_term_matrix.shape[0], document_term_matrix.shape[1]))
            model = LatentDirichletAllocation(
                n_components=n_components,
                learning_method="batch",
                random_state=self.random_state,
            )
        elif self.model_name == "nmf":
            vectorizer_name = "tfidf"
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
            )
            document_term_matrix = vectorizer.fit_transform(texts)
            n_components = max(1, min(self.num_topics, document_term_matrix.shape[0], document_term_matrix.shape[1]))
            model = NMF(
                n_components=n_components,
                init="nndsvda",
                random_state=self.random_state,
                max_iter=400,
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if document_term_matrix.shape[1] == 0:
            raise ValueError("Vectorization produced no usable vocabulary.")

        document_topic_matrix = model.fit_transform(document_term_matrix)
        topic_term_matrix = np.asarray(model.components_)
        feature_names = vectorizer.get_feature_names_out().tolist()
        topic_assignments = document_topic_matrix.argmax(axis=1)
        topic_scores = document_topic_matrix.max(axis=1)

        topic_rows: list[dict[str, object]] = []
        label_map: dict[int, str] = {}
        for topic_id, component in enumerate(topic_term_matrix):
            top_indices = np.argsort(component)[::-1][: self.top_words]
            keywords = [feature_names[index] for index in top_indices]
            dominant_domain = self._dominant_domain(articles, topic_assignments == topic_id)
            label = self._make_label(keywords, dominant_domain)
            label_map[topic_id] = label
            topic_rows.append(
                {
                    "topic_id": topic_id,
                    "topic_label": label,
                    "dominant_domain": dominant_domain,
                    "keywords": ", ".join(keywords),
                    "size": int((topic_assignments == topic_id).sum()),
                }
            )

        annotated = articles.copy()
        annotated["topic_id"] = topic_assignments.astype(int)
        annotated["topic_score"] = np.round(topic_scores.astype(float), 4)
        annotated["topic_label"] = annotated["topic_id"].map(label_map)

        return TopicModelOutput(
            articles=annotated,
            topic_info=pd.DataFrame(topic_rows).sort_values("topic_id").reset_index(drop=True),
            document_topic_matrix=document_topic_matrix,
            document_term_matrix=document_term_matrix,
            topic_term_matrix=topic_term_matrix,
            feature_names=feature_names,
            model_name=self.model_name,
            vectorizer_name=vectorizer_name,
        )

    def _fit_bertopic(self, articles: pd.DataFrame) -> TopicModelOutput:
        try:
            from bertopic import BERTopic
        except ImportError as exc:
            raise RuntimeError("BERTopic is optional. Install the bertopic extra to use it.") from exc

        texts = articles["processed_text"].tolist()
        if len(texts) < 2:
            raise ValueError("At least two articles are required for BERTopic.")

        tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        if tfidf_matrix.shape[1] == 0:
            raise ValueError("Vectorization produced no usable vocabulary.")

        embedding_components = max(2, min(50, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1]))
        reducer = TruncatedSVD(n_components=embedding_components, random_state=self.random_state)
        embeddings = reducer.fit_transform(tfidf_matrix)

        topic_model = BERTopic(
            nr_topics=self.num_topics,
            min_topic_size=max(2, len(texts) // max(self.num_topics, 2)),
            calculate_probabilities=True,
            verbose=False,
            embedding_model=None,
            vectorizer_model=CountVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
            ),
        )

        raw_topic_ids, probabilities = topic_model.fit_transform(texts, embeddings=embeddings)
        normalized_ids, label_map, topic_info, topic_term_matrix, feature_names = self._build_bertopic_outputs(
            topic_model,
            raw_topic_ids,
        )

        if probabilities is None:
            topic_scores = np.ones(len(texts), dtype=float)
            document_topic_matrix = np.ones((len(texts), 1), dtype=float)
        else:
            probability_matrix = np.asarray(probabilities)
            if probability_matrix.ndim == 1:
                topic_scores = probability_matrix.astype(float)
                document_topic_matrix = probability_matrix.reshape(-1, 1)
            else:
                topic_scores = probability_matrix.max(axis=1).astype(float)
                document_topic_matrix = probability_matrix

        annotated = articles.copy()
        annotated["topic_id"] = normalized_ids
        annotated["topic_score"] = np.round(topic_scores, 4)
        annotated["topic_label"] = annotated["topic_id"].map(label_map)

        return TopicModelOutput(
            articles=annotated,
            topic_info=topic_info,
            document_topic_matrix=document_topic_matrix,
            document_term_matrix=tfidf_matrix,
            topic_term_matrix=topic_term_matrix,
            feature_names=feature_names,
            model_name="bertopic",
            vectorizer_name="tfidf",
        )

    def _build_bertopic_outputs(
        self,
        topic_model: Any,
        raw_topic_ids: list[int],
    ) -> tuple[np.ndarray, dict[int, str], pd.DataFrame, np.ndarray, list[str]]:
        unique_topic_ids = sorted({int(topic_id) for topic_id in raw_topic_ids})
        remap = {original_id: index for index, original_id in enumerate(unique_topic_ids)}
        normalized_ids = np.array([remap[int(topic_id)] for topic_id in raw_topic_ids], dtype=int)

        keywords_by_topic: dict[int, list[tuple[str, float]]] = {}
        label_map: dict[int, str] = {}
        topic_rows: list[dict[str, object]] = []

        for original_id in unique_topic_ids:
            normalized_id = remap[original_id]
            raw_keywords = topic_model.get_topic(original_id) or []
            keywords = [(word, float(score)) for word, score in raw_keywords[: self.top_words]]
            keyword_terms = [word for word, _ in keywords]
            keywords_by_topic[normalized_id] = keywords
            dominant_domain = self._dominant_domain(articles, normalized_ids == normalized_id)
            label = self._make_label(keyword_terms or [f"topic_{normalized_id}"], dominant_domain)
            label_map[normalized_id] = label
            topic_rows.append(
                {
                    "topic_id": normalized_id,
                    "topic_label": label,
                    "dominant_domain": dominant_domain,
                    "keywords": ", ".join(keyword_terms or [f"topic_{normalized_id}"]),
                    "size": int((normalized_ids == normalized_id).sum()),
                }
            )

        feature_names = sorted({word for keywords in keywords_by_topic.values() for word, _ in keywords})
        topic_term_matrix = np.zeros((len(topic_rows), len(feature_names)), dtype=float)
        feature_index = {word: index for index, word in enumerate(feature_names)}

        for topic_id, keywords in keywords_by_topic.items():
            for word, score in keywords:
                topic_term_matrix[topic_id, feature_index[word]] = score

        topic_info = pd.DataFrame(topic_rows).sort_values("topic_id").reset_index(drop=True)
        return normalized_ids, label_map, topic_info, topic_term_matrix, feature_names

    @staticmethod
    def _dominant_domain(articles: pd.DataFrame, mask: np.ndarray) -> str:
        if "predicted_domain" not in articles.columns:
            return "General"
        domain_counts = articles.loc[mask, "predicted_domain"].value_counts()
        if domain_counts.empty:
            return "General"
        return str(domain_counts.index[0])

    @staticmethod
    def _make_label(keywords: list[str], dominant_domain: str = "General") -> str:
        visible_keywords = [keyword for keyword in keywords if keyword][:3]
        keyword_label = " / ".join(keyword.title() for keyword in visible_keywords) or "Misc Topic"
        return f"{dominant_domain}: {keyword_label}" if dominant_domain and dominant_domain != "General" else keyword_label


def build_topic_modeler(
    model_name: str,
    num_topics: int,
    top_words: int,
    max_features: int = 3000,
    min_df: int = 1,
    max_df: float = 0.95,
    random_state: int = 42,
) -> TopicModeler:
    return TopicModeler(
        model_name=model_name,
        num_topics=num_topics,
        top_words=top_words,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        random_state=random_state,
    )
