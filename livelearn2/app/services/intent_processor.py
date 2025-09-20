"""Intent processing service for feedback targeting."""

import hashlib
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from loguru import logger
from sqlalchemy.orm import Session

from ..models.intent_feedback import IntentKey, FeedbackCluster
from ..utils.text_processing import chunk_text
from .mock_embeddings import MockEmbeddings


class IntentProcessor:
    """Process and normalize user intents for feedback targeting."""
    
    def __init__(self, embeddings_service: Optional[MockEmbeddings] = None):
        self.embeddings = embeddings_service or MockEmbeddings()
        
        # Stop words for Russian and English
        self.stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
            'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
            'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
            'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
            'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
            'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
            'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
            'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
            'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
            'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть',
            'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
            'their', 'what', 'where', 'when', 'why', 'how', 'which', 'who', 'whom', 'whose'
        }
        
        logger.info("Intent processor initialized")
    
    def normalize_query(self, query_text: str) -> Dict[str, Any]:
        """
        Normalize query text to extract intent features.
        
        Args:
            query_text: Raw query text
            
        Returns:
            Normalized intent data
        """
        try:
            # Clean and lowercase
            cleaned = query_text.lower().strip()
            
            # Remove punctuation except for important ones
            cleaned = re.sub(r'[^\w\s\-\?\!]', ' ', cleaned)
            
            # Tokenize
            tokens = cleaned.split()
            
            # Remove stop words
            filtered_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
            
            # Extract entities (simple pattern-based for now)
            entities = self._extract_entities(query_text)
            
            # Sort tokens for consistent ordering
            sorted_tokens = sorted(filtered_tokens)
            
            # Create normalized text
            normalized_text = ' '.join(sorted_tokens)
            
            return {
                'normalized_text': normalized_text,
                'tokens': sorted_tokens,
                'entities': entities,
                'original_text': query_text
            }
            
        except Exception as e:
            logger.error(f"Error normalizing query: {e}")
            return {
                'normalized_text': query_text.lower(),
                'tokens': query_text.lower().split(),
                'entities': [],
                'original_text': query_text
            }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract simple entities from text."""
        entities = []
        
        # Numbers
        numbers = re.findall(r'\b\d+\b', text)
        entities.extend([f"NUM:{num}" for num in numbers])
        
        # Years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        entities.extend([f"YEAR:{year}" for year in years])
        
        # Email patterns
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities.extend([f"EMAIL:{email}" for email in emails])
        
        # URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        entities.extend([f"URL:{url}" for url in urls])
        
        # Simple capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend([f"PROPER:{noun}" for noun in proper_nouns if len(noun) > 2])
        
        return list(set(entities))  # Remove duplicates
    
    def generate_intent_key(self, normalized_data: Dict[str, Any]) -> str:
        """
        Generate a deterministic intent key from normalized data.
        
        Args:
            normalized_data: Output from normalize_query
            
        Returns:
            SHA-256 hash as intent key
        """
        try:
            # Create a deterministic string representation
            key_parts = [
                normalized_data['normalized_text'],
                '|'.join(sorted(normalized_data['entities'])),
                str(len(normalized_data['tokens']))
            ]
            
            key_string = '||'.join(key_parts)
            
            # Generate SHA-256 hash
            intent_key = hashlib.sha256(key_string.encode('utf-8')).hexdigest()
            
            logger.debug(f"Generated intent key: {intent_key[:16]}... for: {normalized_data['normalized_text'][:50]}...")
            return intent_key
            
        except Exception as e:
            logger.error(f"Error generating intent key: {e}")
            # Fallback to simple hash of original text
            return hashlib.sha256(normalized_data['original_text'].encode('utf-8')).hexdigest()
    
    async def process_and_store_intent(
        self,
        db: Session,
        query_text: str
    ) -> str:
        """
        Process query and store/retrieve intent key.
        
        Args:
            db: Database session
            query_text: User query
            
        Returns:
            Intent key
        """
        try:
            # Normalize query
            normalized_data = self.normalize_query(query_text)
            intent_key = self.generate_intent_key(normalized_data)
            
            # Check if intent key already exists
            existing_intent = db.query(IntentKey).filter(
                IntentKey.id == intent_key
            ).first()
            
            if existing_intent:
                logger.debug(f"Found existing intent key: {intent_key[:16]}...")
                return intent_key
            
            # Generate embedding for the normalized text
            embedding = await self.embeddings.embed_query(normalized_data['normalized_text'])
            
            # Create new intent key record
            intent_record = IntentKey(
                id=intent_key,
                normalized_text=normalized_data['normalized_text'],
                entities=normalized_data['entities'],
                tokens=normalized_data['tokens'],
                embedding=embedding
            )
            
            db.add(intent_record)
            db.commit()
            
            logger.info(f"Created new intent key: {intent_key[:16]}... for query: {query_text[:50]}...")
            return intent_key
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error processing intent: {e}")
            # Return fallback key
            return hashlib.sha256(query_text.encode('utf-8')).hexdigest()
    
    async def find_similar_intents(
        self,
        db: Session,
        intent_key: str,
        similarity_threshold: float = 0.8,
        limit: int = 10
    ) -> List[str]:
        """
        Find similar intent keys for feedback propagation.
        
        Args:
            db: Database session
            intent_key: Target intent key
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results
            
        Returns:
            List of similar intent keys
        """
        try:
            # Get target intent
            target_intent = db.query(IntentKey).filter(
                IntentKey.id == intent_key
            ).first()
            
            if not target_intent:
                logger.warning(f"Intent key not found: {intent_key}")
                return []
            
            # Get all other intents
            all_intents = db.query(IntentKey).filter(
                IntentKey.id != intent_key
            ).all()
            
            if not all_intents:
                return []
            
            # Calculate similarities
            target_embedding = target_intent.embedding
            similar_intents = []
            
            for intent in all_intents:
                try:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(target_embedding, intent.embedding)
                    
                    if similarity >= similarity_threshold:
                        similar_intents.append((intent.id, similarity))
                        
                except Exception as e:
                    logger.warning(f"Error calculating similarity for intent {intent.id}: {e}")
                    continue
            
            # Sort by similarity and return top results
            similar_intents.sort(key=lambda x: x[1], reverse=True)
            result_keys = [intent_id for intent_id, _ in similar_intents[:limit]]
            
            logger.debug(f"Found {len(result_keys)} similar intents for {intent_key[:16]}...")
            return result_keys
            
        except Exception as e:
            logger.error(f"Error finding similar intents: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def cluster_intents(
        self,
        db: Session,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 2
    ) -> int:
        """
        Cluster similar intents for feedback propagation.
        
        Args:
            db: Database session
            similarity_threshold: Minimum similarity for clustering
            min_cluster_size: Minimum intents per cluster
            
        Returns:
            Number of clusters created
        """
        try:
            logger.info("Starting intent clustering...")
            
            # Get all intents
            all_intents = db.query(IntentKey).all()
            
            if len(all_intents) < min_cluster_size:
                logger.info("Not enough intents for clustering")
                return 0
            
            # Simple clustering algorithm
            clusters = []
            processed = set()
            
            for intent in all_intents:
                if intent.id in processed:
                    continue
                
                # Start new cluster
                cluster_members = [intent.id]
                processed.add(intent.id)
                
                # Find similar intents
                for other_intent in all_intents:
                    if other_intent.id in processed:
                        continue
                    
                    similarity = self._cosine_similarity(intent.embedding, other_intent.embedding)
                    
                    if similarity >= similarity_threshold:
                        cluster_members.append(other_intent.id)
                        processed.add(other_intent.id)
                
                # Only keep clusters with minimum size
                if len(cluster_members) >= min_cluster_size:
                    clusters.append(cluster_members)
            
            # Store clusters in database
            cluster_count = 0
            for i, members in enumerate(clusters):
                # Calculate centroid
                embeddings = []
                for member_id in members:
                    member_intent = next(intent for intent in all_intents if intent.id == member_id)
                    embeddings.append(member_intent.embedding)
                
                # Simple centroid calculation
                centroid = [sum(dim) / len(embeddings) for dim in zip(*embeddings)]
                
                cluster_record = FeedbackCluster(
                    cluster_name=f"cluster_{i}",
                    center_embedding=centroid,
                    threshold=similarity_threshold,
                    intent_keys=members,
                    member_count=len(members)
                )
                
                db.add(cluster_record)
                cluster_count += 1
            
            db.commit()
            
            logger.info(f"Created {cluster_count} intent clusters")
            return cluster_count
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error clustering intents: {e}")
            return 0


