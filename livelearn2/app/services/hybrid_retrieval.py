"""Hybrid retrieval system combining Dense embeddings and BM25."""

import math
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from ..models.documents import Chunk
from ..utils.vectors import cosine_similarity, batch_cosine_similarity
from .bm25_search import BM25Search, BM25Result
from .mock_embeddings import MockEmbeddings


@dataclass
class HybridResult:
    """Result from hybrid retrieval with detailed scoring."""
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    normalized_dense: float = 0.0
    normalized_bm25: float = 0.0
    final_score: float = 0.0
    alpha: float = 0.6
    retrieval_method: str = "hybrid"
    matched_terms: set = None
    
    def __post_init__(self):
        if self.matched_terms is None:
            self.matched_terms = set()


class HybridRetrieval:
    """
    Hybrid retrieval system implementing the formula:
    score = α · z(dense) + (1 - α) · z(bm25)
    
    where:
    - z(dense) = normalized dense embedding similarity score
    - z(bm25) = normalized BM25 keyword score
    - α ≈ 0.6 (default weight for dense retrieval)
    """
    
    def __init__(
        self,
        embeddings_service: Optional[MockEmbeddings] = None,
        bm25_service: Optional[BM25Search] = None,
        alpha: float = 0.6,
        dense_weight: float = 0.6,  # Same as alpha, for clarity
        bm25_weight: float = 0.4    # 1 - alpha
    ):
        """
        Initialize hybrid retrieval system.
        
        Args:
            embeddings_service: Dense embedding service
            bm25_service: BM25 search service
            alpha: Weight for dense scores (0.6 recommended)
            dense_weight: Explicit weight for dense retrieval
            bm25_weight: Explicit weight for BM25 retrieval
        """
        self.embeddings = embeddings_service or MockEmbeddings()
        self.bm25 = bm25_service or BM25Search()
        
        # Ensure weights sum to 1
        if abs(dense_weight + bm25_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1: {dense_weight} + {bm25_weight} = {dense_weight + bm25_weight}")
            # Normalize weights
            total = dense_weight + bm25_weight
            dense_weight = dense_weight / total
            bm25_weight = bm25_weight / total
        
        self.alpha = alpha
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        
        logger.info(
            f"Initialized hybrid retrieval with α={self.alpha:.2f} "
            f"(dense: {self.dense_weight:.2f}, bm25: {self.bm25_weight:.2f})"
        )
    
    def z_score_normalize(self, scores: List[float]) -> List[float]:
        """
        Z-score normalization: (x - mean) / std
        
        Args:
            scores: List of raw scores
            
        Returns:
            List of normalized scores
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        # Avoid division by zero
        if std_score < 1e-8:
            return [0.0] * len(scores)
        
        normalized = (scores_array - mean_score) / std_score
        
        # Ensure non-negative scores (shift to [0, inf])
        min_val = np.min(normalized)
        if min_val < 0:
            normalized = normalized - min_val
        
        return normalized.tolist()
    
    def min_max_normalize(self, scores: List[float]) -> List[float]:
        """
        Min-max normalization: (x - min) / (max - min)
        
        Args:
            scores: List of raw scores
            
        Returns:
            List of normalized scores in [0, 1]
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        # Avoid division by zero
        if max_score - min_score < 1e-8:
            return [1.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    async def dense_retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[Chunk, float]]:
        """
        Dense retrieval using embeddings.
        
        Args:
            db: Database session
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        try:
            logger.debug(f"Dense retrieval for query: {query[:50]}...")
            
            # Generate query embedding
            query_embedding = await self.embeddings.embed_query(query)
            
            # Get all chunks with embeddings
            chunks = db.query(Chunk).filter(
                Chunk.embedding.isnot(None),
                Chunk.embedding != "[]",
                Chunk.source == "original"
            ).all()
            
            if not chunks:
                logger.warning("No chunks with embeddings found")
                return []
            
            # Calculate similarities
            embeddings = [chunk.embedding for chunk in chunks]
            similarities = batch_cosine_similarity(query_embedding, embeddings)
            
            # Create results
            results = list(zip(chunks, similarities))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Dense retrieval found {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []
    
    def bm25_retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[Chunk, float, set]]:
        """
        BM25 retrieval using keyword matching.
        
        Args:
            db: Database session
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of (chunk, bm25_score, matched_terms) tuples
        """
        try:
            logger.debug(f"BM25 retrieval for query: {query[:50]}...")
            
            # Perform BM25 search
            bm25_results = self.bm25.search(
                db=db,
                query=query,
                top_k=top_k,
                min_score=0.01  # Lower threshold for hybrid fusion
            )
            
            # Convert to tuples
            results = [
                (result.chunk, result.score, result.matched_terms)
                for result in bm25_results
            ]
            
            logger.debug(f"BM25 retrieval found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return []
    
    async def hybrid_search(
        self,
        db: Session,
        query: str,
        top_k: int = 10,
        dense_k: int = 20,
        bm25_k: int = 20,
        normalization: str = "min_max"  # "min_max" or "z_score"
    ) -> List[HybridResult]:
        """
        Perform hybrid search combining dense and BM25 retrieval.
        
        Args:
            db: Database session
            query: Search query
            top_k: Final number of results to return
            dense_k: Number of dense results to retrieve
            bm25_k: Number of BM25 results to retrieve
            normalization: Normalization method ("min_max" or "z_score")
            
        Returns:
            List of HybridResult objects
        """
        try:
            logger.info(f"Hybrid search for query: {query[:100]}...")
            
            # Step 1: Dense retrieval
            dense_results = await self.dense_retrieve(db, query, dense_k)
            logger.debug(f"Dense retrieval: {len(dense_results)} results")
            
            # Step 2: BM25 retrieval
            bm25_results = self.bm25_retrieve(db, query, bm25_k)
            logger.debug(f"BM25 retrieval: {len(bm25_results)} results")
            
            # Step 3: Create unified result map
            result_map: Dict[int, HybridResult] = {}
            
            # Add dense results
            dense_scores = [score for _, score in dense_results]
            if dense_scores:
                if normalization == "z_score":
                    normalized_dense = self.z_score_normalize(dense_scores)
                else:
                    normalized_dense = self.min_max_normalize(dense_scores)
                
                for (chunk, raw_score), norm_score in zip(dense_results, normalized_dense):
                    result_map[chunk.id] = HybridResult(
                        chunk=chunk,
                        dense_score=raw_score,
                        normalized_dense=norm_score,
                        alpha=self.alpha
                    )
            
            # Add BM25 results
            bm25_scores = [score for _, score, _ in bm25_results]
            if bm25_scores:
                if normalization == "z_score":
                    normalized_bm25 = self.z_score_normalize(bm25_scores)
                else:
                    normalized_bm25 = self.min_max_normalize(bm25_scores)
                
                for (chunk, raw_score, matched_terms), norm_score in zip(bm25_results, normalized_bm25):
                    if chunk.id in result_map:
                        # Update existing result
                        result = result_map[chunk.id]
                        result.bm25_score = raw_score
                        result.normalized_bm25 = norm_score
                        result.matched_terms = matched_terms
                        result.retrieval_method = "hybrid"
                    else:
                        # Create new result
                        result_map[chunk.id] = HybridResult(
                            chunk=chunk,
                            bm25_score=raw_score,
                            normalized_bm25=norm_score,
                            matched_terms=matched_terms,
                            retrieval_method="bm25_only",
                            alpha=self.alpha
                        )
            
            # Step 4: Calculate final hybrid scores
            hybrid_results = []
            
            for result in result_map.values():
                # Apply the hybrid formula: α · z(dense) + (1 - α) · z(bm25)
                final_score = (
                    self.alpha * result.normalized_dense + 
                    (1 - self.alpha) * result.normalized_bm25
                )
                
                result.final_score = final_score
                
                # Update retrieval method
                if result.dense_score > 0 and result.bm25_score > 0:
                    result.retrieval_method = "hybrid"
                elif result.dense_score > 0:
                    result.retrieval_method = "dense_only"
                else:
                    result.retrieval_method = "bm25_only"
                
                hybrid_results.append(result)
            
            # Step 5: Sort by final score and return top-k
            hybrid_results.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.info(
                f"Hybrid search completed: {len(hybrid_results)} total results, "
                f"returning top {top_k}"
            )
            
            # Log score distribution
            if hybrid_results:
                top_result = hybrid_results[0]
                logger.debug(
                    f"Top result: dense={top_result.normalized_dense:.3f}, "
                    f"bm25={top_result.normalized_bm25:.3f}, "
                    f"final={top_result.final_score:.3f}, "
                    f"method={top_result.retrieval_method}"
                )
            
            return hybrid_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def explain_hybrid_score(self, result: HybridResult) -> Dict[str, Any]:
        """
        Explain how the hybrid score was calculated.
        
        Args:
            result: HybridResult to explain
            
        Returns:
            Detailed explanation dictionary
        """
        return {
            "chunk_id": result.chunk.id,
            "retrieval_method": result.retrieval_method,
            "scores": {
                "dense_raw": result.dense_score,
                "bm25_raw": result.bm25_score,
                "dense_normalized": result.normalized_dense,
                "bm25_normalized": result.normalized_bm25,
                "final_hybrid": result.final_score
            },
            "weights": {
                "alpha": result.alpha,
                "dense_weight": result.alpha,
                "bm25_weight": 1 - result.alpha
            },
            "formula": f"{result.alpha:.2f} × {result.normalized_dense:.3f} + {1-result.alpha:.2f} × {result.normalized_bm25:.3f} = {result.final_score:.3f}",
            "matched_terms": list(result.matched_terms),
            "content_preview": result.chunk.content[:200] + "..." if len(result.chunk.content) > 200 else result.chunk.content
        }
    
    async def search_with_explanation(
        self,
        db: Session,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hybrid search with detailed explanations.
        
        Args:
            db: Database session
            query: Search query
            top_k: Number of results
            
        Returns:
            Dictionary with results and explanations
        """
        results = await self.hybrid_search(db, query, top_k)
        
        return {
            "query": query,
            "total_results": len(results),
            "hybrid_parameters": {
                "alpha": self.alpha,
                "dense_weight": self.dense_weight,
                "bm25_weight": self.bm25_weight
            },
            "results": [self.explain_hybrid_score(result) for result in results],
            "bm25_index_stats": self.bm25.get_index_statistics()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hybrid retrieval statistics."""
        return {
            "hybrid_parameters": {
                "alpha": self.alpha,
                "dense_weight": self.dense_weight,
                "bm25_weight": self.bm25_weight
            },
            "bm25_statistics": self.bm25.get_index_statistics(),
            "embedding_service": type(self.embeddings).__name__
        }

