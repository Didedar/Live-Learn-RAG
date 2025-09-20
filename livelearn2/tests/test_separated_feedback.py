"""Tests for separated feedback system - ensuring no leakage."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.documents import Document, Chunk
from app.models.intent_feedback import IntentKey, IntentFeedback, FeedbackLabel, FeedbackScope
from app.services.separated_rag_pipeline import SeparatedRAGPipeline
from app.services.intent_processor import IntentProcessor
from app.services.separated_feedback_handler import SeparatedFeedbackHandler


@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


@pytest.fixture
def mock_embeddings():
    """Mock embeddings service."""
    embeddings = AsyncMock()
    embeddings.embed_query.return_value = [0.1] * 384
    embeddings.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
    return embeddings


@pytest.fixture
def mock_llm():
    """Mock LLM service."""
    llm = AsyncMock()
    llm.generate.return_value = "Test answer based on provided context."
    return llm


@pytest.fixture
def separated_pipeline(mock_embeddings, mock_llm):
    """Create separated RAG pipeline with mocks."""
    return SeparatedRAGPipeline(
        embeddings_service=mock_embeddings,
        llm_service=mock_llm
    )


class TestSeparationIntegrity:
    """Test that feedback and documents remain separated."""
    
    @pytest.mark.asyncio
    async def test_docs_index_purity(self, test_db, separated_pipeline):
        """Test that docs_index contains only original content."""
        # Ingest some test content
        await separated_pipeline.ingest_text(
            db=test_db,
            text="This is original document content for testing.",
            metadata={"type": "test_doc"},
            uri="test://doc1"
        )
        
        # Verify only original chunks exist
        all_chunks = test_db.query(Chunk).all()
        assert len(all_chunks) > 0
        
        for chunk in all_chunks:
            assert chunk.source == "original", f"Found non-original chunk: {chunk.source}"
        
        # Verify no feedback-generated content
        feedback_chunks = test_db.query(Chunk).filter(
            Chunk.source == "user_feedback"
        ).all()
        assert len(feedback_chunks) == 0, "Found feedback contamination in docs_index"
    
    @pytest.mark.asyncio
    async def test_feedback_store_isolation(self, test_db, separated_pipeline):
        """Test that feedback is stored separately from documents."""
        # First, create a message session by asking a question
        result = await separated_pipeline.ask(
            question="What is the test content about?",
            db=test_db,
            user_id="test_user"
        )
        
        message_id = result["message_id"]
        
        # Store feedback
        feedback_id = await separated_pipeline.store_user_feedback(
            db=test_db,
            message_id=message_id,
            feedback_label="reject",
            target_doc_ids=[1],
            target_chunk_ids=[1],
            user_id="test_user",
            notes="This information is incorrect"
        )
        
        # Verify feedback is in separate store
        feedback = test_db.query(IntentFeedback).filter(
            IntentFeedback.id == feedback_id
        ).first()
        
        assert feedback is not None
        assert feedback.label == FeedbackLabel.REJECT
        assert feedback.notes == "This information is incorrect"
        
        # Verify feedback didn't create new documents or chunks
        original_chunks_count = test_db.query(Chunk).filter(
            Chunk.source == "original"
        ).count()
        
        all_chunks_count = test_db.query(Chunk).count()
        
        assert original_chunks_count == all_chunks_count, "Feedback created new chunks in docs_index"
    
    @pytest.mark.asyncio
    async def test_intent_based_gating(self, test_db, separated_pipeline):
        """Test that feedback only applies to matching intents."""
        # Create two different questions with different intents
        result1 = await separated_pipeline.ask(
            question="What is machine learning?",
            db=test_db,
            user_id="test_user"
        )
        
        result2 = await separated_pipeline.ask(
            question="How to cook pasta?",
            db=test_db,
            user_id="test_user"
        )
        
        # Store feedback for first question
        feedback_id = await separated_pipeline.store_user_feedback(
            db=test_db,
            message_id=result1["message_id"],
            feedback_label="reject",
            target_doc_ids=[1],
            target_chunk_ids=[],
            user_id="test_user",
            notes="ML info is wrong"
        )
        
        # Verify feedback has proper intent key
        feedback = test_db.query(IntentFeedback).filter(
            IntentFeedback.id == feedback_id
        ).first()
        
        assert feedback.intent_key is not None
        
        # Verify different queries have different intent keys
        intent_keys = test_db.query(IntentKey).all()
        assert len(intent_keys) >= 2, "Different questions should have different intent keys"
        
        # Verify intent keys are different for different topics
        ml_intent = None
        pasta_intent = None
        
        for intent in intent_keys:
            if "machine learning" in intent.normalized_text.lower():
                ml_intent = intent.id
            elif "cook pasta" in intent.normalized_text.lower():
                pasta_intent = intent.id
        
        if ml_intent and pasta_intent:
            assert ml_intent != pasta_intent, "Different topics should have different intent keys"
    
    @pytest.mark.asyncio
    async def test_feedback_application_without_contamination(self, test_db, separated_pipeline):
        """Test that feedback application doesn't contaminate context."""
        # Ingest test content
        await separated_pipeline.ingest_text(
            db=test_db,
            text="Machine learning is a subset of artificial intelligence.",
            uri="test://ml_doc"
        )
        
        # Ask question and get response
        result1 = await separated_pipeline.ask(
            question="What is machine learning?",
            db=test_db,
            user_id="test_user"
        )
        
        # Store negative feedback
        await separated_pipeline.store_user_feedback(
            db=test_db,
            message_id=result1["message_id"],
            feedback_label="reject",
            target_doc_ids=[1],
            target_chunk_ids=[1],
            user_id="test_user",
            notes="This definition is incomplete"
        )
        
        # Ask the same question again
        result2 = await separated_pipeline.ask(
            question="What is machine learning?",
            db=test_db,
            user_id="test_user"
        )
        
        # Verify contexts still come from original documents only
        for context in result2["contexts"]:
            metadata = context.get("metadata", {})
            assert metadata.get("source", "original") == "original"
            
            # Verify no feedback text in context
            context_text = context.get("text", "")
            assert "This definition is incomplete" not in context_text
            assert "reject" not in context_text.lower()
    
    @pytest.mark.asyncio
    async def test_correction_text_not_in_context(self, test_db, separated_pipeline):
        """Test that correction text is not added to retrieval context."""
        # Ingest test content
        await separated_pipeline.ingest_text(
            db=test_db,
            text="The capital of France is Lyon.",
            uri="test://geography"
        )
        
        # Ask question
        result = await separated_pipeline.ask(
            question="What is the capital of France?",
            db=test_db,
            user_id="test_user"
        )
        
        # Store correction feedback
        await separated_pipeline.store_user_feedback(
            db=test_db,
            message_id=result["message_id"],
            feedback_label="fix",
            target_doc_ids=[1],
            target_chunk_ids=[1],
            user_id="test_user",
            correction_text="The capital of France is Paris, not Lyon."
        )
        
        # Ask the same question again
        result2 = await separated_pipeline.ask(
            question="What is the capital of France?",
            db=test_db,
            user_id="test_user"
        )
        
        # Verify correction text is NOT in any context
        for context in result2["contexts"]:
            context_text = context.get("text", "")
            assert "Paris" not in context_text, "Correction text leaked into context"
            assert "The capital of France is Paris" not in context_text
        
        # Verify original (incorrect) content is still there
        found_original = False
        for context in result2["contexts"]:
            if "Lyon" in context.get("text", ""):
                found_original = True
                break
        
        assert found_original, "Original content was incorrectly removed"
    
    def test_separation_integrity_view(self, test_db):
        """Test the separation integrity database view."""
        # This would test the SQL view created in migration
        # For now, we'll test the concept programmatically
        
        # Count original chunks
        original_chunks = test_db.query(Chunk).filter(
            Chunk.source == "original"
        ).count()
        
        # Count any feedback contamination
        feedback_chunks = test_db.query(Chunk).filter(
            Chunk.source == "user_feedback"
        ).count()
        
        # Count feedback entries
        feedback_entries = test_db.query(IntentFeedback).count()
        
        # Verify integrity
        assert feedback_chunks == 0, f"Found {feedback_chunks} feedback chunks in docs_index"
        
        # If we have feedback, it should be in separate store only
        if feedback_entries > 0:
            # Verify all feedback has intent keys
            feedback_with_intent = test_db.query(IntentFeedback).filter(
                IntentFeedback.intent_key.isnot(None)
            ).count()
            
            assert feedback_with_intent == feedback_entries, "Some feedback lacks intent keys"


class TestIntentProcessor:
    """Test intent processing and normalization."""
    
    @pytest.mark.asyncio
    async def test_intent_normalization(self, mock_embeddings):
        """Test query normalization produces consistent results."""
        processor = IntentProcessor(mock_embeddings)
        
        # Test same intent with different phrasing
        query1 = "What is machine learning?"
        query2 = "What does machine learning mean?"
        query3 = "Can you explain machine learning?"
        
        norm1 = processor.normalize_query(query1)
        norm2 = processor.normalize_query(query2)
        norm3 = processor.normalize_query(query3)
        
        # Should have similar normalized forms
        assert "machine" in norm1["tokens"]
        assert "learning" in norm1["tokens"]
        assert "machine" in norm2["tokens"]
        assert "learning" in norm2["tokens"]
        
        # Generate intent keys
        key1 = processor.generate_intent_key(norm1)
        key2 = processor.generate_intent_key(norm2)
        key3 = processor.generate_intent_key(norm3)
        
        # Similar intents might have different keys (that's OK)
        # But identical normalized queries should have same keys
        same_norm = processor.normalize_query("What is machine learning?")
        same_key = processor.generate_intent_key(same_norm)
        
        assert key1 == same_key, "Identical queries should have same intent key"
    
    @pytest.mark.asyncio
    async def test_different_intents_different_keys(self, mock_embeddings):
        """Test that different topics get different intent keys."""
        processor = IntentProcessor(mock_embeddings)
        
        queries = [
            "What is machine learning?",
            "How to cook pasta?",
            "What is the weather today?",
            "How to fix a car engine?"
        ]
        
        intent_keys = []
        for query in queries:
            normalized = processor.normalize_query(query)
            key = processor.generate_intent_key(normalized)
            intent_keys.append(key)
        
        # All keys should be different
        assert len(set(intent_keys)) == len(intent_keys), "Different topics should have different intent keys"


class TestFeedbackHandler:
    """Test separated feedback handler."""
    
    @pytest.mark.asyncio
    async def test_evidence_overlap_gating(self, test_db, mock_embeddings):
        """Test that feedback only applies when evidence overlaps."""
        handler = SeparatedFeedbackHandler(mock_embeddings)
        
        # Create test documents
        doc1 = Document(uri="test://doc1", doc_metadata={})
        doc2 = Document(uri="test://doc2", doc_metadata={})
        test_db.add_all([doc1, doc2])
        test_db.flush()
        
        # Create chunks
        chunk1 = Chunk(
            document_id=doc1.id,
            ordinal=0,
            content="Content about ML",
            embedding=[0.1] * 384,
            source="original"
        )
        chunk2 = Chunk(
            document_id=doc2.id,
            ordinal=0,
            content="Content about cooking",
            embedding=[0.2] * 384,
            source="original"
        )
        test_db.add_all([chunk1, chunk2])
        test_db.commit()
        
        # Store feedback for doc1
        feedback_id = await handler.store_feedback(
            db=test_db,
            message_id="test_msg",
            query_text="What is ML?",
            feedback_label=FeedbackLabel.REJECT,
            evidence_docs=[doc1.id],
            evidence_chunks=[],
            user_id="test_user"
        )
        
        # Test retrieval with doc2 (no overlap)
        retrieved_docs = [
            {
                'doc_id': doc2.id,
                'chunk_id': chunk2.id,
                'score': 0.8,
                'content': 'Content about cooking',
                'metadata': {}
            }
        ]
        
        reranked_docs, applied_feedback = await handler.apply_feedback_to_query(
            db=test_db,
            query_text="What is ML?",
            retrieved_docs=retrieved_docs,
            user_id="test_user"
        )
        
        # Feedback should NOT be applied (no evidence overlap)
        assert len(applied_feedback) == 0, "Feedback applied without evidence overlap"
        assert len(reranked_docs) == len(retrieved_docs), "Documents were filtered incorrectly"
    
    @pytest.mark.asyncio
    async def test_user_scope_filtering(self, test_db, mock_embeddings):
        """Test that local scope feedback only applies to same user."""
        handler = SeparatedFeedbackHandler(mock_embeddings)
        
        # Store feedback for user1
        feedback_id = await handler.store_feedback(
            db=test_db,
            message_id="test_msg",
            query_text="Test query",
            feedback_label=FeedbackLabel.REJECT,
            evidence_docs=[1],
            evidence_chunks=[],
            user_id="user1"
        )
        
        # Test with different user
        retrieved_docs = [
            {
                'doc_id': 1,
                'chunk_id': 1,
                'score': 0.8,
                'content': 'Test content',
                'metadata': {}
            }
        ]
        
        reranked_docs, applied_feedback = await handler.apply_feedback_to_query(
            db=test_db,
            query_text="Test query",
            retrieved_docs=retrieved_docs,
            user_id="user2"  # Different user
        )
        
        # Local feedback should NOT apply to different user
        assert len(applied_feedback) == 0, "Local feedback applied to different user"


@pytest.mark.asyncio
async def test_end_to_end_separation(test_db, mock_embeddings, mock_llm):
    """End-to-end test of complete separation."""
    pipeline = SeparatedRAGPipeline(mock_embeddings, mock_llm)
    
    # 1. Ingest original content
    doc_id, chunk_count = await pipeline.ingest_text(
        db=test_db,
        text="Machine learning is a method of data analysis.",
        uri="test://ml_doc"
    )
    
    # 2. Ask question
    result1 = await pipeline.ask(
        question="What is machine learning?",
        db=test_db,
        user_id="test_user"
    )
    
    # 3. Provide negative feedback
    feedback_id = await pipeline.store_user_feedback(
        db=test_db,
        message_id=result1["message_id"],
        feedback_label="reject",
        target_doc_ids=[doc_id],
        target_chunk_ids=[],
        user_id="test_user",
        notes="This definition is too brief"
    )
    
    # 4. Ask same question again
    result2 = await pipeline.ask(
        question="What is machine learning?",
        db=test_db,
        user_id="test_user"
    )
    
    # 5. Verify complete separation
    
    # Check that docs_index is pure
    all_chunks = test_db.query(Chunk).all()
    for chunk in all_chunks:
        assert chunk.source == "original"
        assert "This definition is too brief" not in chunk.content
    
    # Check that feedback is in separate store
    feedback = test_db.query(IntentFeedback).filter(
        IntentFeedback.id == feedback_id
    ).first()
    assert feedback is not None
    assert feedback.notes == "This definition is too brief"
    
    # Check that contexts are still from original docs only
    for context in result2["contexts"]:
        assert "This definition is too brief" not in context["text"]
        assert context["metadata"]["source"] == "original"
    
    # Check that feedback was applied for reranking
    assert result2.get("feedback_applied_count", 0) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


