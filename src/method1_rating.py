

"""Method 1: Boolean Search + Rating Filter

This method enhances the baseline by filtering results based on:
1. Opinion polarity from the lexicon (positive vs negative)
2. Star rating correlation with polarity
"""

from __future__ import annotations
from typing import List, Set
import os

from .baseline0 import BaselineBooleanSearch, BooleanSearchResult
from .review import Review


class RatingFilterSearch:
    """Method 1: Boolean search with opinion polarity and rating filtering."""
    
    def __init__(
        self,
        reviews: List[Review],
        positive_lexicon_path: str = None,
        negative_lexicon_path: str = None,
    ):
        # Initialize baseline search
        self.baseline = BaselineBooleanSearch(reviews)
        self.reviews = reviews
        
        # Load opinion lexicons
        if positive_lexicon_path is None:
            # Default path relative to project root
            base_dir = os.path.dirname(os.path.dirname(__file__))
            positive_lexicon_path = os.path.join(
                base_dir, "data", "opinion-lexicon", "positive-words.txt"
            )
        if negative_lexicon_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            negative_lexicon_path = os.path.join(
                base_dir, "data", "opinion-lexicon", "negative-words.txt"
            )
        
        self.positive_words = self._load_lexicon(positive_lexicon_path)
        self.negative_words = self._load_lexicon(negative_lexicon_path)
        
        print(f"Loaded {len(self.positive_words)} positive words")
        print(f"Loaded {len(self.negative_words)} negative words")
    
    def _load_lexicon(self, path: str) -> Set[str]:
        """Load opinion words from file, skipping comment lines."""
        words = set()
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments (lines starting with ;)
                if line and not line.startswith(';'):
                    words.add(line.lower())
        return words
    
    def _get_opinion_polarity(self, opinion_terms: List[str]) -> str:
        """Determine if opinion terms are positive, negative, or neutral.
        
        Returns:
            'positive', 'negative', or 'neutral'
        """
        positive_count = sum(1 for term in opinion_terms if term in self.positive_words)
        negative_count = sum(1 for term in opinion_terms if term in self.negative_words)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> List[BooleanSearchResult]:
        """Search with opinion polarity and rating filtering.
        
        Args:
            query: Opinion query in format 'aspect:opinion'
            top_k: Number of results to return
            
        Returns:
            List of BooleanSearchResult filtered by rating
        """
        # Check if it's an opinion query
        if ':' not in query:
            # Fallback to baseline
            return self.baseline.search(query, mode="and", top_k=top_k)
        
        # Parse query
        aspect_terms, opinion_terms = self.baseline._parse_opinion_query(query)
        
        if not aspect_terms or not opinion_terms:
            return self.baseline.search(query, mode="and", top_k=top_k)
        
        # Get baseline candidates
        aspect_docs = self.baseline._get_docs_with_any_term(aspect_terms)
        opinion_docs = self.baseline._get_docs_with_any_term(opinion_terms)
        candidates = aspect_docs & opinion_docs
        
        if not candidates:
            return []
        
        # Determine opinion polarity
        polarity = self._get_opinion_polarity(opinion_terms)
        
        # Filter by rating based on polarity
        if polarity == 'positive':
            # Keep only high-rated reviews (> 3 stars)
            candidates = {c for c in candidates if self.reviews[c].rating > 3}
        elif polarity == 'negative':
            # Keep only low-rated reviews (<= 3 stars)
            candidates = {c for c in candidates if self.reviews[c].rating <= 3}
        # If neutral, keep all candidates
        
        if not candidates:
            return []
        
        # Score and rank
        all_query_terms = aspect_terms + opinion_terms
        scores = self.baseline._score_candidates(candidates, all_query_terms)
        
        # Sort by score descending, then by doc_id
        scores.sort(key=lambda x: (-x[1], x[0]))
        
        # Build results
        results = []
        for doc_id, score in scores[:top_k]:
            results.append(BooleanSearchResult(review=self.reviews[doc_id], score=score))
        
        return results


# Convenience function
def build_method1_from_pickle(pickle_path: str) -> RatingFilterSearch:
    """Load reviews and build Method 1 search engine."""
    from .data_loader import load_reviews
    reviews = load_reviews(pickle_path)
    return RatingFilterSearch(reviews)