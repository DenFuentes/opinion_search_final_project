"""Baseline0: Simple Boolean search over review texts.

This module builds an inverted index over Review.text and
supports AND / OR boolean search. It is intentionally simple and
self-contained so you can compare it against Method 1 (M1) and
Method 2 (M2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Iterable, Tuple
import re

from .data_loader import load_reviews
from .review import Review


# -----------------------------
# Tokenization / preprocessing
# -----------------------------

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")

# Very small stopword list just to avoid indexing super-common junk
DEFAULT_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "it",
    "for",
    "on",
    "this",
    "that",
}


def _tokenize(text: str, *, lowercase: bool = True) -> List[str]:
    """Tokenize text into alphanumeric tokens.

    - Lowercases by default
    - Returns only tokens of length >= 2
    """

    if not text:
        return []

    if lowercase:
        text = text.lower()

    tokens = [m.group(0) for m in TOKEN_PATTERN.finditer(text)]
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens


@dataclass
class BooleanSearchResult:
    review: Review
    score: int  # number of matched query terms


class BaselineBooleanSearch:
    """Baseline0: Boolean search over review text.

    Builds an inverted index term -> set(doc_ids) and supports
    simple AND/OR queries. Ranking is by the number of matched
    query terms (higher is better), then by original index.
    """

    def __init__(
        self,
        reviews: List[Review],
        *,
        stopwords: Iterable[str] | None = DEFAULT_STOPWORDS,
        lowercase: bool = True,
    ) -> None:
        self.reviews: List[Review] = reviews
        self.lowercase = lowercase
        self.stopwords: Set[str] = set(stopwords) if stopwords is not None else set()
        # term -> set of doc indices
        self.inverted_index: Dict[str, Set[int]] = {}

        self._build_index()

    # -----------------
    # Index construction
    # -----------------

    def _build_index(self) -> None:
        """Build the inverted index over review.text."""

        for doc_id, review in enumerate(self.reviews):
            tokens = _tokenize(review.text, lowercase=self.lowercase)
            for token in tokens:
                if token in self.stopwords:
                    continue
                postings = self.inverted_index.setdefault(token, set())
                postings.add(doc_id)

    # -------------
    # Helper methods
    # -------------

    def _normalize_query(self, query: str) -> List[str]:
        tokens = _tokenize(query, lowercase=self.lowercase)
        return [t for t in tokens if t not in self.stopwords]

    def _parse_opinion_query(self, query: str) -> Tuple[List[str], List[str]]:
        """Parse 'aspect:opinion' format.
        
        Returns:
            (aspect_terms, opinion_terms)
        """
        parts = query.split(':', 1)
        aspect = self._normalize_query(parts[0])
        opinion = self._normalize_query(parts[1])
        return aspect, opinion

    def _get_docs_with_any_term(self, terms: List[str]) -> Set[int]:
        """Get all docs containing ANY of the terms (OR operation)."""
        result = set()
        for term in terms:
            result |= self.inverted_index.get(term, set())
        return result

    def _score_candidates(
        self, 
        candidates: Set[int], 
        query_terms: List[str]
    ) -> List[Tuple[int, int]]:
        """Score candidates by counting matched query terms.
        
        Returns:
            List of (doc_id, score) tuples
        """
        scores: List[Tuple[int, int]] = []
        for doc_id in candidates:
            doc_tokens = _tokenize(self.reviews[doc_id].text, lowercase=self.lowercase)
            doc_token_set = set(doc_tokens)
            match_count = sum(1 for t in query_terms if t in doc_token_set)
            scores.append((doc_id, match_count))
        return scores

    # -------------
    # Search API
    # -------------

    def search(
        self,
        query: str,
        *,
        mode: str = "and",
        top_k: int = 10,
    ) -> List[BooleanSearchResult]:
        """Run boolean search supporting 'aspect:opinion' format.
        
        Args:
            query: User query string. Can be 'aspect:opinion' or plain text.
            mode: "and" or "or" (only used for plain text queries)
            top_k: Number of results to return.
            
        Returns:
            List of BooleanSearchResult sorted by score descending.
        """
        
        # Check if it's an opinion query
        if ':' in query:
            aspect_terms, opinion_terms = self._parse_opinion_query(query)
            
            if not aspect_terms or not opinion_terms:
                # Fallback to regular search
                terms = self._normalize_query(query)
                return self._regular_search(terms, mode=mode, top_k=top_k)
            
            # Baseline: (aspect1 OR aspect2 OR ...) AND (opinion1 OR opinion2 OR ...)
            aspect_docs = self._get_docs_with_any_term(aspect_terms)
            opinion_docs = self._get_docs_with_any_term(opinion_terms)
            
            # Intersection
            candidates = aspect_docs & opinion_docs
            
            # All query terms for scoring
            all_query_terms = aspect_terms + opinion_terms
            
        else:
            # Regular boolean search
            terms = self._normalize_query(query)
            return self._regular_search(terms, mode=mode, top_k=top_k)
        
        if not candidates:
            return []
        
        # Score and rank
        scores = self._score_candidates(candidates, all_query_terms)
        
        # Sort: higher score first, then smaller doc_id as tie-breaker
        scores.sort(key=lambda x: (-x[1], x[0]))
        
        results: List[BooleanSearchResult] = []
        for doc_id, score in scores[:top_k]:
            results.append(BooleanSearchResult(review=self.reviews[doc_id], score=score))
        
        return results

    def _regular_search(
        self,
        terms: List[str],
        *,
        mode: str = "and",
        top_k: int = 10,
    ) -> List[BooleanSearchResult]:
        """Regular AND/OR boolean search."""
        
        if not terms:
            return []
        
        # Collect candidate document ids via AND/OR over postings
        postings_sets: List[Set[int]] = [
            self.inverted_index.get(term, set()) for term in terms
        ]
        
        if mode == "and":
            # Start with all docs in first postings, then intersect
            candidates: Set[int] = set(postings_sets[0])
            for s in postings_sets[1:]:
                candidates &= s
        else:  # mode == "or"
            candidates = set()
            for s in postings_sets:
                candidates |= s
        
        if not candidates:
            return []
        
        # Score and rank
        scores = self._score_candidates(candidates, terms)
        
        # Sort: higher score first, then smaller doc_id as tie-breaker
        scores.sort(key=lambda x: (-x[1], x[0]))
        
        results: List[BooleanSearchResult] = []
        for doc_id, score in scores[:top_k]:
            results.append(BooleanSearchResult(review=self.reviews[doc_id], score=score))
        
        return results


# -----------------
# Convenience helpers
# -----------------


def build_baseline0_from_pickle(pickle_path: str) -> BaselineBooleanSearch:
    """Helper to load reviews and build the Baseline0 index.

    Example
    -------
    >>> engine = build_baseline0_from_pickle("data/reviews_segment.pkl")
    >>> results = engine.search("audio quality:poor", top_k=5)
    >>> for r in results:
    ...     print(r.score, r.review.title)
    """

    reviews = load_reviews(pickle_path)
    return BaselineBooleanSearch(reviews)


if __name__ == "__main__":  # Simple manual CLI for quick testing
    import argparse

    parser = argparse.ArgumentParser(description="Baseline0 boolean search over reviews.")
    parser.add_argument(
        "pickle_path",
        type=str,
        help="Path to reviews_segment.pkl (or similar) file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="and",
        choices=["and", "or"],
        help="Boolean mode: 'and' (default) or 'or'",
    )

    args = parser.parse_args()

    print("Loading reviews and building index...")
    engine = build_baseline0_from_pickle(args.pickle_path)
    print(f"Loaded {len(engine.reviews)} reviews. Ready to search.\n")

    try:
        while True:
            query = input("Query (empty to exit): ").strip()
            if not query:
                break

            results = engine.search(query, mode=args.mode, top_k=5)
            if not results:
                print("No results found.\n")
                continue

            print("Top results:")
            for idx, res in enumerate(results, start=1):
                title = res.review.title or "<no title>"
                text = res.review.text or ""
                print(f"[{idx}] score={res.score} | {title}")
                # Print a short snippet of the review text
                snippet = text[:200].replace("\n", " ")
                print(f"    {snippet}...")
            print()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")