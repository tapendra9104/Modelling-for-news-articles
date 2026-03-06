from __future__ import annotations

import pandas as pd

DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "Politics": {
        "campaign",
        "coalition",
        "election",
        "government",
        "minister",
        "parliament",
        "policy",
        "regulation",
        "senate",
        "vote",
    },
    "Sports": {
        "coach",
        "cricket",
        "football",
        "goal",
        "league",
        "match",
        "player",
        "score",
        "stadium",
        "tournament",
    },
    "Technology": {
        "accelerator",
        "algorithm",
        "automation",
        "chip",
        "cloud",
        "machine",
        "robotics",
        "semiconductor",
        "software",
        "startup",
    },
    "Business": {
        "bank",
        "business",
        "earnings",
        "equities",
        "export",
        "finance",
        "inflation",
        "logistics",
        "market",
        "retail",
    },
    "Health": {
        "antiviral",
        "clinic",
        "diagnosis",
        "doctor",
        "health",
        "hospital",
        "medical",
        "patient",
        "telemedicine",
        "vaccine",
    },
    "Climate": {
        "carbon",
        "climate",
        "drought",
        "energy",
        "flood",
        "heatwave",
        "irrigation",
        "renewable",
        "storm",
        "wind",
    },
}


class DomainClassifier:
    def __init__(self, taxonomy: dict[str, set[str]] | None = None) -> None:
        self.taxonomy = taxonomy or DOMAIN_KEYWORDS

    def predict(self, processed_text: str) -> str:
        tokens = set(processed_text.split())
        best_domain = "General"
        best_score = 0

        for domain, keywords in self.taxonomy.items():
            score = len(tokens & keywords)
            if score > best_score:
                best_domain = domain
                best_score = score

        return best_domain

    def annotate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        annotated = frame.copy()
        annotated["predicted_domain"] = annotated["processed_text"].map(self.predict)
        return annotated
