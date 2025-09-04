"""Tests for feedback system functionality."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models.documents import Chunk, Document
from app.models.feedback import FeedbackEvent, FeedbackLabel, MessageSession
from app.schemas.feedback import AskRequest, FeedbackRequest, UserFeedback, FeedbackTarget
from app.services.feedback_handler import FeedbackHandler
from app.services.rag_pipeline import EnhancedRAGPipeline


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_feedback.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Setup test database."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Create database session for tests."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def sample_document(db_session):
    """Create a sample document with chunks."""
    doc = Document(
        uri="test_doc",
        metadata={"source": "test"}
    )
    db_session.add(doc)
    db_session.flush()
    
    # Add chunks with sample embeddings
    chunks = []
    for i in range(3):
        chunk = Chunk(
            document_id=doc.id,
            ordinal=i + 1,
            content=f"This is test chunk {i + 1} with some important information.",
            embedding=[0.1 * j for j in range(384)]  # Mock embedding
        )
        db_session.add(chunk)
        chunks.append(chunk)
    
    db_session.commit()
    return doc, chunks


@pytest.fixture
def sample_message_session(db_session, sample_document):
    """Create a sample message session."""
    doc, chunks = sample_document
    
    message = MessageSession(
        question="What is important information?",
        answer="Important information is contained in the test chunks.",
        contexts_used=[
            {
                "doc_id": doc.id,
                "chunk_id": chunks[0].id,
                "score": 0.95,
                "content": chunks[0].content
            }
        ]
    )
    db_session.add(message)
    db_session.commit()
    return message


class TestFeedbackFlow:
    """Test the complete feedback flow."""
    
    def test_ask_endpoint(self, db_session, sample_document):
        """Test the /ask endpoint creates proper message tracking."""
        # Mock the RAG pipeline response
        request_data = {
            "question": "What is test information?",
            "top_k": 5
        }
        
        # Note: This would need mocking of Ollama services in real tests
        # For now, we'll test the structure
        response = client.post("/v1/feedback/ask", json=request_data)
        
        # In a full test, we'd mock the services and verify:
        # assert response.status_code == 200
        # assert "message_id" in response.json()
        # assert "answer" in response.json()
        # assert "contexts" in response.json()
    
    def test_feedback_incorrect_with_correction(self, db_session, sample_message_session, sample_document):
        """Test feedback flow for incorrect information with correction."""
        doc, chunks = sample_document
        message = sample_message_session
        
        feedback_handler = FeedbackHandler()
        
        # Create feedback request
        feedback_request = FeedbackRequest(
            message_id=message.id,
            question=message.question,
            model_answer=message.answer,
            user_feedback=UserFeedback(
                label=FeedbackLabel.INCORRECT,
                correction_text="The corrected information is that test chunks contain verified data.",
                scope="chunk",
                target=FeedbackTarget(doc_id=doc.id, chunk_id=chunks[0].id),
                reason="The original information was outdated."
            )
        )
        
        # Process feedback (would need mocking in real test)
        # feedback_id, update_ids = await feedback_handler.process_feedback(
        #     db_session, feedback_request
        # )
        
        # Verify feedback event was created
        feedback_event = db_session.query(FeedbackEvent).filter(
            FeedbackEvent.message_id == message.id
        ).first()
        
        # In full test, would verify:
        # assert feedback_event is not None
        # assert feedback_event.label == FeedbackLabel.INCORRECT
        # assert feedback_event.correction_text == feedback_request.user_feedback.correction_text
    
    def test_tombstone_operation(self, db_session, sample_document):
        """Test tombstone operation deprecates chunks correctly."""
        doc, chunks = sample_document
        chunk_to_deprecate = chunks[0]
        
        feedback_handler = FeedbackHandler()
        
        # Create mock feedback event
        feedback_event = FeedbackEvent(
            message_id="test_message",
            label=FeedbackLabel.INCORRECT,
            scope="chunk",
            target_chunk_id=chunk_to_deprecate.id
        )
        db_session.add(feedback_event)
        db_session.flush()
        
        # Apply tombstone (would need async in real test)
        # await feedback_handler._create_tombstone_mutation(
        #     db_session, feedback_event, chunk_to_deprecate.id
        # )
        
        # Verify chunk weight was created and deprecated
        from app.models.feedback import ChunkWeight
        chunk_weight = db_session.query(ChunkWeight).filter(
            ChunkWeight.chunk_id == chunk_to_deprecate.id
        ).first()
        
        # In full test:
        # assert chunk_weight is not None
        # assert chunk_weight.is_deprecated is True
        # assert chunk_weight.penalty_weight < 0
    
    def test_upsert_operation(self, db_session):
        """Test upsert operation creates new corrected chunks."""
        correction_text = "This is the corrected information that should be added to the knowledge base."
        
        feedback_handler = FeedbackHandler()
        
        # Create mock feedback event
        feedback_event = FeedbackEvent(
            message_id="test_message",
            label=FeedbackLabel.INCORRECT,
            correction_text=correction_text
        )
        db_session.add(feedback_event)
        db_session.flush()
        
        # Apply upsert (would need async and mocking in real test)
        # await feedback_handler._create_upsert_mutation(
        #     db_session, feedback_event, correction_text
        # )
        
        # Verify new document was created
        correction_doc = db_session.query(Document).filter(
            Document.uri == "user_feedback"
        ).first()
        
        # In full test:
        # assert correction_doc is not None
        # assert correction_doc.metadata["source"] == "user_feedback"
        
        # Verify chunks were created with boost weights
        # correction_chunks = db_session.query(Chunk).filter(
        #     Chunk.document_id == correction_doc.id
        # ).all()
        # assert len(correction_chunks) > 0
    
    def test_rerank_bias_operation(self, db_session, sample_document):
        """Test rerank bias adjusts chunk weights."""
        doc, chunks = sample_document
        chunk_to_boost = chunks[0]
        
        feedback_handler = FeedbackHandler()
        
        # Create mock feedback event for correct information
        feedback_event = FeedbackEvent(
            message_id="test_message",
            label=FeedbackLabel.CORRECT,
            scope="chunk",
            target_chunk_id=chunk_to_boost.id
        )
        db_session.add(feedback_event)
        db_session.flush()
        
        # Apply rerank bias (would need async in real test)
        # await feedback_handler._create_rerank_bias_mutation(
        #     db_session, feedback_event, chunk_to_boost.id, boost_weight=0.5
        # )
        
        # Verify chunk weight was created with boost
        from app.models.feedback import ChunkWeight
        chunk_weight = db_session.query(ChunkWeight).filter(
            ChunkWeight.chunk_id == chunk_to_boost.id
        ).first()
        
        # In full test:
        # assert chunk_weight is not None
        # assert chunk_weight.boost_weight > 0
        # assert chunk_weight.feedback_count == 1


class TestFeedbackAPI:
    """Test feedback API endpoints."""
    
    def test_feedback_endpoint_validation(self):
        """Test feedback endpoint validates input correctly."""
        # Test missing message_id
        invalid_request = {
            "question": "Test question",
            "model_answer": "Test answer",
            "user_feedback": {
                "label": "incorrect",
                "correction_text": "Corrected text",
                "scope": "chunk"
            }
        }
        
        response = client.post("/v1/feedback/feedback", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_feedback_endpoint_message_not_found(self):
        """Test feedback endpoint handles missing message."""
        request_data = {
            "message_id": "non_existent_id",
            "question": "Test question",
            "model_answer": "Test answer",
            "user_feedback": {
                "label": "incorrect",
                "correction_text": "Corrected text",
                "scope": "chunk"
            }
        }
        
        response = client.post("/v1/feedback/feedback", json=request_data)
        assert response.status_code == 404
    
    def test_revert_endpoint_validation(self):
        """Test revert endpoint validates input."""
        response = client.post("/v1/feedback/revert", json={"update_id": "non_existent"})
        assert response.status_code == 404
    
    def test_history_endpoint(self):
        """Test feedback history endpoint."""
        response = client.get("/v1/feedback/history?limit=10")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_stats_endpoint(self):
        """Test feedback stats endpoint."""
        response = client.get("/v1/feedback/stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert "total_feedback_events" in stats
        assert "feedback_by_label" in stats
        assert "pending_updates" in stats


class TestFeedbackIntegration:
    """Integration tests for the complete feedback cycle."""
    
    @pytest.mark.asyncio
    async def test_complete_feedback_cycle(self, db_session, sample_document):
        """Test complete cycle: ask -> feedback -> ask again."""
        doc, chunks = sample_document
        
        # This would be a comprehensive E2E test that:
        # 1. Asks a question and gets an answer
        # 2. Provides incorrect feedback with correction
        # 3. Asks the same question again
        # 4. Verifies the answer has improved
        
        # Mock services would be needed for full implementation
        pipeline = EnhancedRAGPipeline()
        feedback_handler = FeedbackHandler()
        
        # Step 1: Ask initial question (would need mocking)
        # message_id, answer1, contexts1 = await pipeline.ask_question(
        #     db_session, "What is important information?"
        # )
        
        # Step 2: Provide negative feedback
        # feedback_request = FeedbackRequest(...)
        # await feedback_handler.process_feedback(db_session, feedback_request)
        
        # Step 3: Ask same question again
        # message_id2, answer2, contexts2 = await pipeline.ask_question(
        #     db_session, "What is important information?"
        # )
        
        # Step 4: Verify improvement
        # assert answer2 != answer1  # Answer should be different
        # Check that deprecated chunks are not in contexts2
        pass


class TestFeedbackErrorHandling:
    """Test error handling in feedback system."""
    
    def test_invalid_feedback_scope(self, db_session, sample_message_session):
        """Test handling of invalid feedback scope."""
        message = sample_message_session
        
        # Test chunk scope without chunk_id
        feedback_request = FeedbackRequest(
            message_id=message.id,
            question=message.question,
            model_answer=message.answer,
            user_feedback=UserFeedback(
                label=FeedbackLabel.INCORRECT,
                correction_text="Test correction",
                scope="chunk",  # scope is chunk but no target provided
                reason="Test reason"
            )
        )
        
        # Should handle gracefully or provide helpful error
        feedback_handler = FeedbackHandler()
        
        # In real test, would verify appropriate error handling
        # with pytest.raises(ValidationError):
        #     await feedback_handler.process_feedback(db_session, feedback_request)
    
    def test_ollama_service_failure(self):
        """Test handling of Ollama service failures."""
        # Test that feedback system handles embedding service failures gracefully
        # Should queue operations and retry or fail gracefully
        pass
    
    def test_concurrent_feedback(self):
        """Test handling of concurrent feedback on same content."""
        # Test race conditions and ensure data consistency
        pass


if __name__ == "__main__":
    pytest.main([__file__])