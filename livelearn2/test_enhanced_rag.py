#!/usr/bin/env python3
"""
Скрипт для тестирования улучшенного RAG с диагностикой проблем.

Использование:
    python test_enhanced_rag.py
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

import httpx
from loguru import logger


# Конфигурация
API_BASE_URL = "http://localhost:8000"
TEST_QUERIES = [
    "Как получить справку о несудимости?",
    "Что нужно для регистрации по месту жительства?",
    "Как зарегистрировать ИП онлайн?",
    "Какие документы нужны для получения паспорта?",
    "Как подать заявление через egov.kz?",
]


async def test_api_health() -> bool:
    """Проверяет доступность API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/healthz", timeout=10.0)
            response.raise_for_status()
            
            health_data = response.json()
            logger.info(f"API статус: {health_data.get('status')}")
            
            return health_data.get('status') == 'ok'
            
    except Exception as e:
        logger.error(f"API недоступен: {e}")
        return False


async def test_debug_retrieve(query: str) -> Dict[str, Any]:
    """Тестирует debug endpoint для диагностики ретривера."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/debug/retrieve",
                params={"q": query, "k": 10, "threshold": 0.1},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Debug retrieve failed for '{query}': {e}")
        return {"error": str(e)}


async def test_enhanced_query(query: str) -> Dict[str, Any]:
    """Тестирует улучшенный RAG endpoint."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/enhanced/query",
                json={"query": query, "top_k": 4},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Enhanced query failed for '{query}': {e}")
        return {"error": str(e)}


async def test_hybrid_debug(query: str) -> Dict[str, Any]:
    """Тестирует debug endpoint для гибридного поиска."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE_URL}/api/v1/enhanced/debug/hybrid-retrieve",
                params={
                    "q": query,
                    "k": 8,
                    "alpha": 0.6,
                    "mmr_lambda": 0.15
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Hybrid debug failed for '{query}': {e}")
        return {"error": str(e)}


async def compare_rag_methods(query: str) -> Dict[str, Any]:
    """Сравнивает разные методы RAG для одного запроса."""
    logger.info(f"🔍 Сравнение методов RAG для: '{query}'")
    
    results = {"query": query}
    
    # 1. Обычный RAG
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/query",
                json={"query": query, "top_k": 4},
                timeout=30.0
            )
            if response.status_code == 200:
                data = response.json()
                results["basic_rag"] = {
                    "answer_length": len(data.get("answer", "")),
                    "contexts_found": len(data.get("citations", [])),
                    "top_scores": [c.get("score", 0) for c in data.get("citations", [])[:3]]
                }
            else:
                results["basic_rag"] = {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        results["basic_rag"] = {"error": str(e)}
    
    # 2. Enhanced RAG
    enhanced_result = await test_enhanced_query(query)
    if "error" not in enhanced_result:
        results["enhanced_rag"] = {
            "answer_length": len(enhanced_result.get("answer", "")),
            "contexts_found": len(enhanced_result.get("citations", [])),
            "top_scores": [c.get("score", 0) for c in enhanced_result.get("citations", [])[:3]]
        }
    else:
        results["enhanced_rag"] = enhanced_result
    
    # 3. Debug информация
    debug_result = await test_debug_retrieve(query)
    if "error" not in debug_result:
        results["debug_info"] = {
            "total_results": debug_result.get("diagnostics", {}).get("total_results", 0),
            "embedding_coverage": debug_result.get("diagnostics", {}).get("embedding_coverage", 0),
            "max_score": debug_result.get("diagnostics", {}).get("max_score", 0)
        }
    
    # 4. Hybrid debug
    hybrid_result = await test_hybrid_debug(query)
    if "error" not in hybrid_result:
        results["hybrid_debug"] = {
            "dense_count": hybrid_result.get("diagnostics", {}).get("dense_count", 0),
            "keyword_count": hybrid_result.get("diagnostics", {}).get("keyword_count", 0),
            "fused_count": hybrid_result.get("diagnostics", {}).get("fused_count", 0),
            "embedding_service": hybrid_result.get("diagnostics", {}).get("embedding_service_active", "unknown")
        }
    
    return results


async def get_system_stats() -> Dict[str, Any]:
    """Получает статистику системы."""
    try:
        async with httpx.AsyncClient() as client:
            # Базовая статистика
            response = await client.get(f"{API_BASE_URL}/api/v1/stats", timeout=10.0)
            basic_stats = response.json() if response.status_code == 200 else {}
            
            # Enhanced статистика
            response = await client.get(f"{API_BASE_URL}/api/v1/enhanced/stats", timeout=10.0)
            enhanced_stats = response.json() if response.status_code == 200 else {}
            
            return {
                "basic": basic_stats,
                "enhanced": enhanced_stats
            }
            
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {"error": str(e)}


async def diagnose_retrieval_issues():
    """Диагностирует проблемы с ретривером."""
    logger.info("🔧 Диагностика проблем с ретривером")
    
    # Получаем статистику системы
    stats = await get_system_stats()
    
    logger.info("📊 Статистика системы:")
    if "basic" in stats:
        basic = stats["basic"]
        logger.info(f"  - Документов: {basic.get('documents', 0)}")
        logger.info(f"  - Чанков: {basic.get('chunks', 0)}")
        logger.info(f"  - Покрытие эмбеддингами: {basic.get('embedding_coverage', 0):.2%}")
    
    if "enhanced" in stats:
        enhanced = stats["enhanced"]
        logger.info(f"  - Тип пайплайна: {enhanced.get('pipeline_type', 'unknown')}")
        logger.info(f"  - Размер чанков: {enhanced.get('chunk_size', 0)} токенов")
        logger.info(f"  - Перекрытие чанков: {enhanced.get('chunk_overlap', 0)} токенов")
        logger.info(f"  - Порог ретривера: {enhanced.get('tau_retr', 0)}")
        
        embedding_service = enhanced.get('embedding_service', {})
        logger.info(f"  - Сервис эмбеддингов: {embedding_service.get('current_service', 'unknown')}")
        logger.info(f"  - Здоровье сервиса: {embedding_service.get('service_healthy', False)}")
    
    # Тестируем поиск
    logger.info("\n🧪 Тестирование поиска:")
    
    for i, query in enumerate(TEST_QUERIES[:3], 1):  # Первые 3 запроса
        logger.info(f"\n{i}. Тестирование: '{query}'")
        
        # Debug retrieve
        debug_result = await test_debug_retrieve(query)
        if "error" not in debug_result:
            diagnostics = debug_result.get("diagnostics", {})
            results = debug_result.get("results", [])
            
            logger.info(f"   📈 Найдено результатов: {diagnostics.get('total_results', 0)}")
            logger.info(f"   📊 Макс. скор: {diagnostics.get('max_score', 0):.3f}")
            logger.info(f"   📊 Мин. скор: {diagnostics.get('min_score', 0):.3f}")
            
            if results:
                logger.info(f"   🥇 Топ результат: скор {results[0].get('score', 0):.3f}")
                preview = results[0].get('content_preview', '')[:100]
                logger.info(f"   📄 Превью: {preview}...")
            else:
                logger.warning("   ⚠️  Нет результатов!")
        else:
            logger.error(f"   ❌ Ошибка debug: {debug_result.get('error', 'unknown')}")


async def run_comprehensive_test():
    """Запускает комплексное тестирование."""
    logger.info("🚀 Запуск комплексного тестирования улучшенного RAG")
    
    # Проверяем доступность API
    if not await test_api_health():
        logger.error("❌ API недоступен. Убедитесь, что сервер запущен.")
        return
    
    # Диагностика
    await diagnose_retrieval_issues()
    
    # Сравнительное тестирование
    logger.info("\n🔬 Сравнительное тестирование методов RAG:")
    
    comparison_results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"\n{i}/{len(TEST_QUERIES)}: {query}")
        
        result = await compare_rag_methods(query)
        comparison_results.append(result)
        
        # Краткий отчет
        basic = result.get("basic_rag", {})
        enhanced = result.get("enhanced_rag", {})
        
        if "error" not in basic:
            logger.info(f"  📊 Базовый RAG: {basic.get('contexts_found', 0)} контекстов, ответ {basic.get('answer_length', 0)} символов")
        else:
            logger.warning(f"  ⚠️  Базовый RAG: {basic.get('error', 'unknown error')}")
        
        if "error" not in enhanced:
            logger.info(f"  ✨ Enhanced RAG: {enhanced.get('contexts_found', 0)} контекстов, ответ {enhanced.get('answer_length', 0)} символов")
        else:
            logger.warning(f"  ⚠️  Enhanced RAG: {enhanced.get('error', 'unknown error')}")
    
    # Сохраняем результаты
    results_file = Path("test_results.json")
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Результаты сохранены в {results_file}")
    except Exception as e:
        logger.error(f"Не удалось сохранить результаты: {e}")
    
    logger.info("\n🎉 Тестирование завершено!")
    
    # Итоговые рекомендации
    logger.info("\n💡 Рекомендации для улучшения:")
    logger.info("1. Убедитесь, что Ollama запущен с моделью nomic-embed-text для эмбеддингов")
    logger.info("2. Проверьте, что данные загружены в базу (должно быть >0 чанков с эмбеддингами)")
    logger.info("3. Если скоры низкие (<0.4), попробуйте переиндексировать с лучшими эмбеддингами")
    logger.info("4. Используйте /api/v1/enhanced/query для лучшего качества поиска")


if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=''),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    # Запуск тестирования
    try:
        asyncio.run(run_comprehensive_test())
    except KeyboardInterrupt:
        logger.info("\n🛑 Тестирование прервано пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")

