"""Vector operations utilities for RAG system."""

from typing import List, Tuple, Union

import numpy as np
from loguru import logger


def normalize_vector(vector: Union[List[float], np.ndarray]) -> List[float]:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector as list or numpy array
        
    Returns:
        Normalized vector as list
        
    Raises:
        ValueError: If vector is empty or all zeros
    """
    try:
        # Convert to numpy array
        v = np.array(vector, dtype=np.float32)
        
        if len(v) == 0:
            raise ValueError("Empty vector provided")
        
        # Calculate norm
        norm = np.linalg.norm(v)
        
        if norm == 0.0:
            logger.warning("Zero vector provided for normalization")
            # Return zero vector of same length
            return [0.0] * len(v)
        
        # Normalize and convert back to list
        normalized = v / norm
        return normalized.tolist()
        
    except Exception as e:
        logger.error(f"Failed to normalize vector: {e}")
        raise ValueError(f"Vector normalization failed: {e}") from e


def cosine_similarity(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
        
    Raises:
        ValueError: If vectors have different dimensions or are invalid
    """
    try:
        # Convert to numpy arrays
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        
        if len(a) != len(b):
            raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
        
        if len(a) == 0:
            raise ValueError("Empty vectors provided")
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Handle zero vectors
        if norm_a == 0.0 or norm_b == 0.0:
            logger.warning("Zero vector in cosine similarity calculation")
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        
        # Clamp to valid range due to floating point errors
        similarity = max(-1.0, min(1.0, float(similarity)))
        
        return similarity
        
    except Exception as e:
        logger.error(f"Failed to calculate cosine similarity: {e}")
        raise ValueError(f"Cosine similarity calculation failed: {e}") from e


def euclidean_distance(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Euclidean distance
    """
    try:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        
        if len(a) != len(b):
            raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
        
        distance = np.linalg.norm(a - b)
        return float(distance)
        
    except Exception as e:
        logger.error(f"Failed to calculate Euclidean distance: {e}")
        raise ValueError(f"Euclidean distance calculation failed: {e}") from e


def batch_cosine_similarity(
    query_vector: Union[List[float], np.ndarray],
    document_vectors: List[Union[List[float], np.ndarray]]
) -> List[float]:
    """
    Calculate cosine similarity between a query vector and multiple document vectors.
    
    Args:
        query_vector: Query vector
        document_vectors: List of document vectors
        
    Returns:
        List of similarity scores
    """
    try:
        if not document_vectors:
            return []
        
        query = np.array(query_vector, dtype=np.float32)
        
        # Convert all vectors to numpy array for efficient computation
        doc_matrix = np.array([np.array(vec, dtype=np.float32) for vec in document_vectors])
        
        # Normalize query vector
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            logger.warning("Zero query vector in batch similarity")
            return [0.0] * len(document_vectors)
        
        query_normalized = query / query_norm
        
        # Normalize document vectors
        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        
        # Handle zero vectors
        zero_mask = doc_norms == 0
        doc_norms[zero_mask] = 1.0  # Avoid division by zero
        
        doc_normalized = doc_matrix / doc_norms[:, np.newaxis]
        
        # Calculate cosine similarities
        similarities = np.dot(doc_normalized, query_normalized)
        
        # Set similarities for zero vectors to 0
        similarities[zero_mask] = 0.0
        
        # Clamp to valid range
        similarities = np.clip(similarities, -1.0, 1.0)
        
        return similarities.tolist()
        
    except Exception as e:
        logger.error(f"Failed batch cosine similarity calculation: {e}")
        # Return zeros as fallback
        return [0.0] * len(document_vectors)


def top_k_similar(
    query_vector: Union[List[float], np.ndarray],
    document_vectors: List[Union[List[float], np.ndarray]],
    k: int,
    return_indices: bool = False
) -> Union[List[float], Tuple[List[float], List[int]]]:
    """
    Find top-k most similar vectors to query.
    
    Args:
        query_vector: Query vector
        document_vectors: List of document vectors
        k: Number of top results to return
        return_indices: Whether to return indices along with scores
        
    Returns:
        Top-k similarity scores, optionally with indices
    """
    try:
        if not document_vectors:
            return ([], []) if return_indices else []
        
        # Calculate all similarities
        similarities = batch_cosine_similarity(query_vector, document_vectors)
        
        # Get top-k indices
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]  # Descending order
        
        top_scores = [similarities[i] for i in top_indices]
        
        if return_indices:
            return top_scores, top_indices.tolist()
        else:
            return top_scores
            
    except Exception as e:
        logger.error(f"Failed top-k similarity search: {e}")
        if return_indices:
            return [], []
        else:
            return []


def vector_stats(vectors: List[Union[List[float], np.ndarray]]) -> dict:
    """
    Calculate statistics for a collection of vectors.
    
    Args:
        vectors: List of vectors
        
    Returns:
        Dictionary with statistics
    """
    try:
        if not vectors:
            return {
                "count": 0,
                "dimensions": 0,
                "mean_norm": 0.0,
                "std_norm": 0.0,
                "min_norm": 0.0,
                "max_norm": 0.0
            }
        
        # Convert to numpy arrays
        vector_array = np.array([np.array(v, dtype=np.float32) for v in vectors])
        
        # Calculate norms
        norms = np.linalg.norm(vector_array, axis=1)
        
        stats = {
            "count": len(vectors),
            "dimensions": vector_array.shape[1] if len(vector_array.shape) > 1 else 0,
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms)),
            "zero_vectors": int(np.sum(norms == 0))
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to calculate vector stats: {e}")
        return {"error": str(e)}


def is_valid_vector(vector: Union[List[float], np.ndarray], min_dim: int = 1) -> bool:
    """
    Check if a vector is valid for similarity calculations.
    
    Args:
        vector: Vector to validate
        min_dim: Minimum required dimensions
        
    Returns:
        True if vector is valid, False otherwise
    """
    try:
        if vector is None:
            return False
        
        v = np.array(vector, dtype=np.float32)
        
        # Check dimensions
        if len(v) < min_dim:
            return False
        
        # Check for NaN or infinity
        if np.any(np.isnan(v)) or np.any(np.isinf(v)):
            return False
        
        # Vector is valid
        return True
        
    except Exception:
        return False


def similarity_threshold_filter(
    similarities: List[float],
    threshold: float = 0.1,
    indices: List[int] = None
) -> Tuple[List[float], List[int]]:
    """
    Filter similarities by threshold.
    
    Args:
        similarities: List of similarity scores
        threshold: Minimum similarity threshold
        indices: Optional list of corresponding indices
        
    Returns:
        Filtered similarities and indices
    """
    try:
        if indices is None:
            indices = list(range(len(similarities)))
        
        filtered_sims = []
        filtered_indices = []
        
        for sim, idx in zip(similarities, indices):
            if sim >= threshold:
                filtered_sims.append(sim)
                filtered_indices.append(idx)
        
        return filtered_sims, filtered_indices
        
    except Exception as e:
        logger.error(f"Failed to filter similarities: {e}")
        return similarities, indices or list(range(len(similarities)))