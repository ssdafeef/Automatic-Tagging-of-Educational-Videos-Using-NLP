import csv
import os
from typing import List, Dict, Optional, Tuple


CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "auto_tags_college_expanded.csv")


def _normalize(s: str) -> str:
    return s.strip().lower()


def _difficulty_rank(d: str) -> int:
    # higher is harder
    ranks = {"easy": 0, "medium": 1, "hard": 2}
    return ranks.get(d.strip().lower(), 0)


class KeywordMatcher:
    """Load the CSV of keywords and provide matching utilities.

    Behavior:
    - Loads `auto_tags_college_expanded.csv` which must have columns: Keyword,Subject,Difficulty
    - When given a transcript, finds rows whose `Keyword` appears as a substring (case-insensitive)
      or whose token overlap (words) is above a small threshold.
    - If multiple rows match, returns the row with the highest difficulty (hard > medium > easy).
    - Returns structured info: matched_keyword, subject, difficulty, matches (list of matching rows).
    """

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or CSV_PATH
        self.rows: List[Dict[str, str]] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.csv_path):
            # don't raise — keep empty rows so caller can handle absence
            self.rows = []
            return
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.rows = [
                {"keyword": _normalize(r.get("Keyword", "")),
                 "subject": r.get("Subject", ""),
                 "difficulty": r.get("Difficulty", "").strip().lower()}
                for r in reader
                if r.get("Keyword")
            ]

    def match(self, transcript: str, min_token_overlap: int = 2) -> Dict:
        """Match transcript against loaded keywords.

        Returns dict with keys:
          - chosen: (keyword, subject, difficulty) or None
          - all_matches: list of rows that matched (keyword, subject, difficulty, score)
        """
        if not transcript or not self.rows:
            return {"chosen": None, "all_matches": []}

        t_norm = _normalize(transcript)
        t_tokens = set([w for w in t_norm.split() if len(w) > 1])

        matches: List[Dict] = []

        for r in self.rows:
            kw = r["keyword"]
            score = 0
            # substring match gives a boost
            if kw in t_norm:
                score += 10
            # token overlap
            kw_tokens = set([w for w in kw.split() if len(w) > 1])
            overlap = len(kw_tokens & t_tokens)
            score += overlap

            if score >= (10 if kw in t_norm else min_token_overlap):
                matches.append({"keyword": kw, "subject": r.get("subject"), "difficulty": r.get("difficulty"), "score": score})

        if not matches:
            return {"chosen": None, "all_matches": []}

        # pick highest difficulty; break ties with score then longer keyword
        matches.sort(key=lambda m: (_difficulty_rank(m["difficulty"]), m["score"], len(m["keyword"])), reverse=True)
        chosen = matches[0]
        return {"chosen": chosen, "all_matches": matches}


def choose_hardest_from_keywords(keywords: List[str]) -> Optional[Tuple[str, str, str]]:
    """Utility: pick hardest difficulty among provided keyword strings (expects difficulty word at end if present)."""
    # Not implemented here — primary API is KeywordMatcher.match
    return None
