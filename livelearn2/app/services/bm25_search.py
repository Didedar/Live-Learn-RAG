"""BM25 search implementation for hybrid retrieval."""

import math
import re
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from ..models.documents import Chunk


@dataclass
class BM25Result:
    """BM25 search result with score and metadata."""
    chunk: Chunk
    score: float
    term_frequencies: Dict[str, int]
    matched_terms: Set[str]


class BM25Search:
    """
    BM25 (Okapi BM25) implementation for keyword-based retrieval.
    
    BM25 formula:
    score(D,Q) = Σ IDF(qi) * f(qi,D) * (k1 + 1) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))
    
    where:
    - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
    - f(qi,D) = frequency of term qi in document D
    - |D| = length of document D in words
    - avgdl = average document length in the collection
    - k1, b = tuning parameters
    """
    
    def __init__(
        self,
        k1: float = 1.5,  # Controls term frequency saturation
        b: float = 0.75,  # Controls length normalization
        epsilon: float = 0.25  # Floor value for IDF
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Document statistics
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # Term document frequencies
        self.doc_lengths: Dict[int, int] = {}  # Document lengths
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.vocabulary: Set[str] = set()
        
        # Index state
        self.is_indexed: bool = False
        
        logger.info(f"Initialized BM25 search with k1={k1}, b={b}, epsilon={epsilon}")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing.
        
        Args:
            text: Input text
            
        Returns:
            List of normalized tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Extract words (Cyrillic and Latin)
        tokens = re.findall(r'\b[а-яёa-z0-9]+\b', text)
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def build_index(self, db: Session) -> None:
        """
        Build BM25 index from database chunks.
        
        Args:
            db: Database session
        """
        logger.info("Building BM25 index from database chunks...")
        
        # Get all original chunks
        chunks = db.query(Chunk).filter(
            Chunk.source == 'original',
            Chunk.content.isnot(None)
        ).all()
        
        if not chunks:
            logger.warning("No chunks found for BM25 indexing")
            return
        
        # Reset statistics
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = {}
        self.vocabulary = set()
        
        # First pass: collect document statistics
        total_length = 0
        
        for chunk in chunks:
            tokens = self.preprocess_text(chunk.content)
            doc_length = len(tokens)
            
            self.doc_lengths[chunk.id] = doc_length
            total_length += doc_length
            
            # Count unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1
                self.vocabulary.add(term)
        
        # Calculate average document length
        self.total_docs = len(chunks)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        
        self.is_indexed = True
        
        logger.info(
            f"BM25 index built: {self.total_docs} documents, "
            f"{len(self.vocabulary)} unique terms, "
            f"avg_doc_length={self.avg_doc_length:.1f}"
        )
    
    def calculate_idf(self, term: str) -> float:
        """
        Calculate Inverse Document Frequency for a term.
        
        Args:
            term: Term to calculate IDF for
            
        Returns:
            IDF score
        """
        if not self.is_indexed:
            return 0.0
        
        doc_freq = self.doc_freqs.get(term, 0)
        
        if doc_freq == 0:
            return 0.0
        
        # BM25 IDF formula with epsilon floor
        idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        return max(self.epsilon, idf)
    
    def calculate_bm25_score(self, query_terms: List[str], chunk: Chunk) -> Tuple[float, Dict[str, int], Set[str]]:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            chunk: Document chunk
            
        Returns:
            Tuple of (score, term_frequencies, matched_terms)
        """
        if not self.is_indexed:
            return 0.0, {}, set()
        
        doc_tokens = self.preprocess_text(chunk.content)
        doc_length = len(doc_tokens)
        
        # Count term frequencies in document
        term_freqs = Counter(doc_tokens)
        
        score = 0.0
        matched_terms = set()
        
        for term in query_terms:
            if term in term_freqs:
                matched_terms.add(term)
                
                # Get term frequency and IDF
                tf = term_freqs[term]
                idf = self.calculate_idf(term)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                
                term_score = idf * (numerator / denominator)
                score += term_score
        
        return score, dict(term_freqs), matched_terms
    
    def search(
        self,
        db: Session,
        query: str,
        top_k: int = 10,
        min_score: float = 0.1,
        rebuild_index: bool = False
    ) -> List[BM25Result]:
        """
        Search using BM25 algorithm.
        
        Args:
            db: Database session
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum BM25 score threshold
            rebuild_index: Whether to rebuild the index
            
        Returns:
            List of BM25Result objects
        """
        logger.debug(f"BM25 search for query: {query[:100]}...")
        
        # Build or rebuild index if needed
        if not self.is_indexed or rebuild_index:
            self.build_index(db)
        
        if not self.is_indexed:
            logger.warning("BM25 index not available")
            return []
        
        # Preprocess query
        query_terms = self.preprocess_text(query)
        
        if not query_terms:
            logger.warning("No valid query terms after preprocessing")
            return []
        
        logger.debug(f"Query terms: {query_terms}")
        
        # Get all chunks for scoring
        chunks = db.query(Chunk).filter(
            Chunk.source == 'original',
            Chunk.content.isnot(None)
        ).all()
        
        # Calculate BM25 scores
        results = []
        
        for chunk in chunks:
            score, term_freqs, matched_terms = self.calculate_bm25_score(query_terms, chunk)
            
            if score >= min_score and matched_terms:
                results.append(BM25Result(
                    chunk=chunk,
                    score=score,
                    term_frequencies=term_freqs,
                    matched_terms=matched_terms
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"BM25 search found {len(results)} results, returning top {top_k}")
        
        return results[:top_k]
    
    def get_term_statistics(self, term: str) -> Dict[str, float]:
        """
        Get statistics for a specific term.
        
        Args:
            term: Term to analyze
            
        Returns:
            Dictionary with term statistics
        """
        if not self.is_indexed:
            return {}
        
        doc_freq = self.doc_freqs.get(term, 0)
        idf = self.calculate_idf(term)
        
        return {
            'term': term,
            'document_frequency': doc_freq,
            'inverse_document_frequency': idf,
            'collection_frequency': doc_freq / self.total_docs if self.total_docs > 0 else 0
        }
    
    def get_index_statistics(self) -> Dict[str, any]:
        """
        Get overall index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'is_indexed': self.is_indexed,
            'total_documents': self.total_docs,
            'vocabulary_size': len(self.vocabulary),
            'average_document_length': self.avg_doc_length,
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }
        }
    
    def explain_score(self, query: str, chunk: Chunk) -> Dict[str, any]:
        """
        Explain BM25 score calculation for debugging.
        
        Args:
            query: Search query
            chunk: Document chunk
            
        Returns:
            Detailed score breakdown
        """
        if not self.is_indexed:
            return {'error': 'Index not built'}
        
        query_terms = self.preprocess_text(query)
        doc_tokens = self.preprocess_text(chunk.content)
        doc_length = len(doc_tokens)
        term_freqs = Counter(doc_tokens)
        
        explanation = {
            'query_terms': query_terms,
            'document_length': doc_length,
            'average_document_length': self.avg_doc_length,
            'term_scores': [],
            'total_score': 0.0
        }
        
        total_score = 0.0
        
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            idf = self.calculate_idf(term)
            
            if tf > 0:
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                term_score = idf * (numerator / denominator)
                total_score += term_score
                
                explanation['term_scores'].append({
                    'term': term,
                    'term_frequency': tf,
                    'inverse_document_frequency': idf,
                    'document_frequency': self.doc_freqs.get(term, 0),
                    'term_score': term_score,
                    'calculation': {
                        'numerator': numerator,
                        'denominator': denominator,
                        'length_normalization': 1 - self.b + self.b * doc_length / self.avg_doc_length
                    }
                })
        
        explanation['total_score'] = total_score
        return explanation

