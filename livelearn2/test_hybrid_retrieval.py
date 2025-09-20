"""Test script for Hybrid Retrieval (Dense + BM25) system."""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

from sqlalchemy.orm import Session
from loguru import logger

# Import our modules
from app.database import get_db_session, init_db
from app.services.hybrid_rag_pipeline import HybridRAGPipeline
from app.services.bm25_search import BM25Search
from app.services.mock_embeddings import MockEmbeddings


class HybridRetrievalTester:
    """Test suite for hybrid retrieval system."""
    
    def __init__(self):
        self.db_session = None
        self.pipeline = None
        
        # Test data for different query types
        self.test_queries = [
            # Factual queries (should benefit from BM25)
            "ИИН 123456789012",
            "закон о государственных услугах",
            "справка о несудимости",
            "регистрация ИП",
            "паспорт гражданина",
            
            # Semantic queries (should benefit from dense)
            "как получить документы онлайн",
            "процедура оформления справки",
            "требования к регистрации",
            "электронные государственные услуги",
            "подача заявления через портал"
        ]
        
        # Sample documents for testing
        self.test_documents = [
            {
                "text": """
                Справка о несудимости выдается гражданам Республики Казахстан по их заявлению. 
                Для получения справки необходимо подать заявление через портал egov.kz или лично в ЦОН.
                Срок выдачи справки составляет 5 рабочих дней.
                Документы: удостоверение личности (паспорт), ИИН обязателен.
                """,
                "metadata": {"type": "справка", "category": "документы"}
            },
            {
                "text": """
                Регистрация индивидуального предпринимателя (ИП) осуществляется в соответствии 
                с законом "О государственной регистрации юридических лиц и индивидуальных предпринимателей".
                Подача документов возможна онлайн через портал электронного правительства egov.kz.
                Необходимые документы: заявление, копия паспорта, справка о несудимости.
                """,
                "metadata": {"type": "регистрация", "category": "бизнес"}
            },
            {
                "text": """
                Электронные государственные услуги предоставляются через единый портал egov.kz.
                Для доступа к услугам необходима электронная цифровая подпись (ЭЦП).
                Граждане могут получить справки, подать заявления и отслеживать статус обращений онлайн.
                Время работы портала: круглосуточно, 7 дней в неделю.
                """,
                "metadata": {"type": "услуги", "category": "электронные"}
            },
            {
                "text": """
                Паспорт гражданина Республики Казахстан является основным документом, удостоверяющим личность.
                Выдается гражданам с 16 лет. Содержит ИИН (индивидуальный идентификационный номер).
                Срок действия паспорта: 10 лет. Замена производится в ЦОНах или через egov.kz.
                При утере паспорта необходимо подать заявление о выдаче дубликата.
                """,
                "metadata": {"type": "документ", "category": "удостоверение"}
            }
        ]
    
    async def setup(self):
        """Initialize database and pipeline."""
        try:
            logger.info("Setting up hybrid retrieval test environment...")
            
            # Initialize database
            init_db()
            self.db_session = get_db_session()
            
            # Initialize services
            embeddings = MockEmbeddings()
            bm25 = BM25Search(k1=1.5, b=0.75, epsilon=0.25)
            
            # Initialize pipeline
            self.pipeline = HybridRAGPipeline(
                embeddings_service=embeddings,
                bm25_service=bm25,
                alpha=0.6,  # 60% dense, 40% BM25
                tau_retr=0.3,  # Lower threshold for testing
                max_contexts=5
            )
            
            logger.info("Test environment setup completed")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    async def ingest_test_data(self):
        """Ingest test documents."""
        try:
            logger.info("Ingesting test documents...")
            
            for i, doc in enumerate(self.test_documents):
                doc_id, chunk_count = await self.pipeline.ingest_text(
                    db=self.db_session,
                    text=doc["text"],
                    metadata=doc["metadata"],
                    uri=f"test_doc_{i+1}",
                    rebuild_bm25=(i == len(self.test_documents) - 1)  # Rebuild only on last doc
                )
                
                logger.info(f"Ingested document {doc_id} with {chunk_count} chunks")
            
            logger.info("Test data ingestion completed")
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
    
    async def test_individual_retrievers(self):
        """Test dense and BM25 retrievers separately."""
        try:
            logger.info("\n" + "="*50)
            logger.info("TESTING INDIVIDUAL RETRIEVERS")
            logger.info("="*50)
            
            for query in self.test_queries[:5]:  # Test first 5 queries
                logger.info(f"\nQuery: {query}")
                logger.info("-" * 40)
                
                # Test dense retrieval
                dense_results = await self.pipeline.hybrid_retrieval.dense_retrieve(
                    db=self.db_session,
                    query=query,
                    top_k=3
                )
                
                logger.info(f"Dense results: {len(dense_results)}")
                for i, (chunk, score) in enumerate(dense_results[:2], 1):
                    logger.info(f"  {i}. Score: {score:.3f} | {chunk.content[:80]}...")
                
                # Test BM25 retrieval
                bm25_results = self.pipeline.hybrid_retrieval.bm25_retrieve(
                    db=self.db_session,
                    query=query,
                    top_k=3
                )
                
                logger.info(f"BM25 results: {len(bm25_results)}")
                for i, (chunk, score, terms) in enumerate(bm25_results[:2], 1):
                    logger.info(f"  {i}. Score: {score:.3f} | Terms: {terms} | {chunk.content[:80]}...")
            
        except Exception as e:
            logger.error(f"Individual retriever test failed: {e}")
            raise
    
    async def test_hybrid_retrieval(self):
        """Test hybrid retrieval system."""
        try:
            logger.info("\n" + "="*50)
            logger.info("TESTING HYBRID RETRIEVAL")
            logger.info("="*50)
            
            for query in self.test_queries:
                logger.info(f"\nQuery: {query}")
                logger.info("-" * 40)
                
                # Perform hybrid search
                hybrid_results = await self.pipeline.hybrid_retrieval.hybrid_search(
                    db=self.db_session,
                    query=query,
                    top_k=3
                )
                
                logger.info(f"Hybrid results: {len(hybrid_results)}")
                
                for i, result in enumerate(hybrid_results, 1):
                    logger.info(
                        f"  {i}. Final: {result.final_score:.3f} "
                        f"(Dense: {result.normalized_dense:.3f}, "
                        f"BM25: {result.normalized_bm25:.3f}) "
                        f"| Method: {result.retrieval_method} "
                        f"| Terms: {result.matched_terms}"
                    )
                    logger.info(f"     Content: {result.chunk.content[:100]}...")
            
        except Exception as e:
            logger.error(f"Hybrid retrieval test failed: {e}")
            raise
    
    async def test_full_rag_pipeline(self):
        """Test complete RAG pipeline with answer generation."""
        try:
            logger.info("\n" + "="*50)
            logger.info("TESTING FULL RAG PIPELINE")
            logger.info("="*50)
            
            test_questions = [
                "Как получить справку о несудимости?",
                "Что нужно для регистрации ИП?",
                "Какие документы нужны для паспорта?",
                "Как работает портал egov.kz?"
            ]
            
            for question in test_questions:
                logger.info(f"\nQuestion: {question}")
                logger.info("-" * 50)
                
                # Ask question using hybrid RAG
                response = await self.pipeline.ask(
                    question=question,
                    db=self.db_session,
                    top_k=3,
                    explain=True
                )
                
                logger.info(f"Can answer: {response['can_answer']}")
                logger.info(f"Max score: {response.get('max_score', 0):.3f}")
                logger.info(f"Contexts used: {len(response['contexts'])}")
                logger.info(f"Answer: {response['answer'][:200]}...")
                
                if response.get('retrieval_explanation'):
                    stats = response['retrieval_explanation']['retrieval_stats']
                    logger.info(f"Retrieval stats: {stats}")
            
        except Exception as e:
            logger.error(f"Full RAG pipeline test failed: {e}")
            raise
    
    async def test_score_normalization(self):
        """Test score normalization methods."""
        try:
            logger.info("\n" + "="*50)
            logger.info("TESTING SCORE NORMALIZATION")
            logger.info("="*50)
            
            query = "справка о несудимости ИИН"
            
            # Test with different normalization methods
            for norm_method in ["min_max", "z_score"]:
                logger.info(f"\nNormalization method: {norm_method}")
                logger.info("-" * 30)
                
                results = await self.pipeline.hybrid_retrieval.hybrid_search(
                    db=self.db_session,
                    query=query,
                    top_k=3,
                    normalization=norm_method
                )
                
                for i, result in enumerate(results, 1):
                    logger.info(
                        f"  {i}. Final: {result.final_score:.3f} "
                        f"(α={result.alpha:.2f} × {result.normalized_dense:.3f} + "
                        f"{1-result.alpha:.2f} × {result.normalized_bm25:.3f})"
                    )
        
        except Exception as e:
            logger.error(f"Score normalization test failed: {e}")
            raise
    
    async def test_alpha_sensitivity(self):
        """Test sensitivity to alpha parameter."""
        try:
            logger.info("\n" + "="*50)
            logger.info("TESTING ALPHA SENSITIVITY")
            logger.info("="*50)
            
            query = "регистрация ИП закон"
            alpha_values = [0.2, 0.4, 0.6, 0.8]
            
            for alpha in alpha_values:
                logger.info(f"\nAlpha = {alpha:.1f} (Dense: {alpha:.1f}, BM25: {1-alpha:.1f})")
                logger.info("-" * 40)
                
                # Create temporary pipeline with different alpha
                temp_hybrid = self.pipeline.hybrid_retrieval.__class__(
                    embeddings_service=self.pipeline.embeddings,
                    bm25_service=self.pipeline.bm25,
                    alpha=alpha
                )
                
                results = await temp_hybrid.hybrid_search(
                    db=self.db_session,
                    query=query,
                    top_k=2
                )
                
                for i, result in enumerate(results, 1):
                    logger.info(
                        f"  {i}. Final: {result.final_score:.3f} "
                        f"| Method: {result.retrieval_method} "
                        f"| Dense contrib: {alpha * result.normalized_dense:.3f} "
                        f"| BM25 contrib: {(1-alpha) * result.normalized_bm25:.3f}"
                    )
        
        except Exception as e:
            logger.error(f"Alpha sensitivity test failed: {e}")
            raise
    
    async def get_pipeline_statistics(self):
        """Get and display pipeline statistics."""
        try:
            logger.info("\n" + "="*50)
            logger.info("PIPELINE STATISTICS")
            logger.info("="*50)
            
            stats = await self.pipeline.get_pipeline_stats(self.db_session)
            
            logger.info(f"Pipeline type: {stats['pipeline_type']}")
            logger.info(f"Parameters: {stats['parameters']}")
            logger.info(f"Chunk statistics: {stats['chunk_statistics']}")
            logger.info(f"Features: {stats['features']}")
            
            # BM25 specific stats
            bm25_stats = self.pipeline.bm25.get_index_statistics()
            logger.info(f"BM25 index: {bm25_stats}")
            
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.db_session:
            self.db_session.close()
        logger.info("Test cleanup completed")
    
    async def run_all_tests(self):
        """Run complete test suite."""
        try:
            await self.setup()
            await self.ingest_test_data()
            await self.test_individual_retrievers()
            await self.test_hybrid_retrieval()
            await self.test_score_normalization()
            await self.test_alpha_sensitivity()
            await self.test_full_rag_pipeline()
            await self.get_pipeline_statistics()
            
            logger.info("\n" + "="*50)
            logger.info("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main test function."""
    logger.info("Starting Hybrid Retrieval Test Suite")
    logger.info("Testing Dense + BM25 with formula: α·z(dense) + (1-α)·z(bm25)")
    
    tester = HybridRetrievalTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

