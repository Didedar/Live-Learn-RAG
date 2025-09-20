#!/usr/bin/env python3
"""
Скрипт для быстрой проверки статуса Live-Learn RAG системы.

Использование:
    python check_system.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

import httpx
from loguru import logger


# Конфигурация
OLLAMA_URL = "http://localhost:11434"
RAG_URL = "http://localhost:8000"


async def check_ollama() -> Dict[str, Any]:
    """Проверяет статус Ollama."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Проверяем доступность
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            
            return {
                "status": "ok",
                "url": OLLAMA_URL,
                "models": models,
                "model_count": len(models)
            }
            
    except Exception as e:
        return {
            "status": "error",
            "url": OLLAMA_URL,
            "error": str(e)
        }


async def check_rag_server() -> Dict[str, Any]:
    """Проверяет статус RAG сервера."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Базовая проверка
            response = await client.get(f"{RAG_URL}/healthz")
            response.raise_for_status()
            basic_health = response.json()
            
            # Детальная проверка
            response = await client.get(f"{RAG_URL}/healthz/detailed")
            response.raise_for_status()
            detailed_health = response.json()
            
            # Статистика
            response = await client.get(f"{RAG_URL}/api/v1/stats")
            response.raise_for_status()
            stats = response.json()
            
            return {
                "status": "ok",
                "url": RAG_URL,
                "basic_health": basic_health,
                "detailed_health": detailed_health,
                "stats": stats
            }
            
    except Exception as e:
        return {
            "status": "error",
            "url": RAG_URL,
            "error": str(e)
        }


async def test_rag_query() -> Dict[str, Any]:
    """Тестирует RAG запрос."""
    try:
        test_query = "Как получить паспорт?"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_URL}/api/v1/feedback/ask",
                json={
                    "question": test_query,
                    "session_id": "health_check",
                    "top_k": 3
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "status": "ok",
                "query": test_query,
                "answer_length": len(result.get("answer", "")),
                "contexts_count": len(result.get("contexts", [])),
                "has_answer": bool(result.get("answer", "").strip())
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_files() -> Dict[str, Any]:
    """Проверяет наличие файлов."""
    files_to_check = [
        "data_for_rag (1).xlsx",
        ".env",
        "rag.db",
        "app/main.py",
        "frontend/index.final.html"
    ]
    
    file_status = {}
    for file_path in files_to_check:
        path = Path(file_path)
        file_status[file_path] = {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0
        }
    
    return file_status


async def main():
    """Основная функция проверки."""
    logger.info("🔍 Проверка статуса Live-Learn RAG системы")
    logger.info("=" * 50)
    
    # Проверка файлов
    logger.info("📁 Проверка файлов...")
    file_status = check_files()
    
    for file_path, status in file_status.items():
        if status["exists"]:
            size_mb = status["size"] / (1024 * 1024)
            logger.info(f"   ✅ {file_path} ({size_mb:.1f} MB)")
        else:
            logger.error(f"   ❌ {file_path} - не найден")
    
    # Проверка Ollama
    logger.info("=" * 50)
    logger.info("🤖 Проверка Ollama...")
    ollama_status = await check_ollama()
    
    if ollama_status["status"] == "ok":
        logger.success(f"   ✅ Ollama доступен: {ollama_status['url']}")
        logger.info(f"   📦 Моделей установлено: {ollama_status['model_count']}")
        for model in ollama_status["models"]:
            logger.info(f"      • {model}")
    else:
        logger.error(f"   ❌ Ollama недоступен: {ollama_status['error']}")
    
    # Проверка RAG сервера
    logger.info("=" * 50)
    logger.info("🚀 Проверка RAG сервера...")
    rag_status = await check_rag_server()
    
    if rag_status["status"] == "ok":
        logger.success(f"   ✅ RAG сервер доступен: {rag_status['url']}")
        
        # Детали о системе
        basic = rag_status["basic_health"]
        detailed = rag_status["detailed_health"]
        stats = rag_status["stats"]
        
        logger.info(f"   📊 Статус: {basic.get('status')}")
        logger.info(f"   🔧 Версия: {basic.get('version')}")
        logger.info(f"   📚 Документов: {stats.get('documents', 0)}")
        logger.info(f"   🧩 Чанков: {stats.get('chunks', 0)}")
        logger.info(f"   🎯 Покрытие эмбеддингами: {stats.get('embedding_coverage', 0):.1%}")
        
        # Статус компонентов
        checks = detailed.get("checks", {})
        for component, check in checks.items():
            status = check.get("status", "unknown")
            if status in ["ok", "healthy"]:
                logger.info(f"   ✅ {component}: {status}")
            elif status in ["disabled", "not_configured"]:
                logger.info(f"   ⚪ {component}: {status}")
            else:
                logger.warning(f"   ⚠️ {component}: {status}")
                
    else:
        logger.error(f"   ❌ RAG сервер недоступен: {rag_status['error']}")
    
    # Тест RAG запроса
    if rag_status["status"] == "ok":
        logger.info("=" * 50)
        logger.info("🧪 Тест RAG запроса...")
        test_result = await test_rag_query()
        
        if test_result["status"] == "ok":
            logger.success(f"   ✅ Запрос обработан: {test_result['query']}")
            logger.info(f"   📝 Длина ответа: {test_result['answer_length']} символов")
            logger.info(f"   📋 Найдено контекстов: {test_result['contexts_count']}")
            
            if test_result["has_answer"]:
                logger.success("   ✅ Ответ сгенерирован")
            else:
                logger.warning("   ⚠️ Пустой ответ")
        else:
            logger.error(f"   ❌ Ошибка в тесте: {test_result['error']}")
    
    # Итоговый статус
    logger.info("=" * 50)
    
    overall_ok = (
        ollama_status["status"] == "ok" and
        rag_status["status"] == "ok" and
        file_status["data_for_rag (1).xlsx"]["exists"] and
        file_status["rag.db"]["exists"]
    )
    
    if overall_ok:
        logger.success("🎉 Система полностью работоспособна!")
        logger.info("")
        logger.info("🌐 Доступные интерфейсы:")
        logger.info(f"   • API: {RAG_URL}/docs")
        logger.info(f"   • Статус: {RAG_URL}/healthz/detailed")
        logger.info("   • Веб-интерфейс: frontend/index.final.html")
        logger.info("")
        logger.info("💡 Пример запроса:")
        logger.info(f'   curl -X POST "{RAG_URL}/api/v1/feedback/ask" \\')
        logger.info('     -H "Content-Type: application/json" \\')
        logger.info('     -d \'{"question": "Как получить справку?", "top_k": 3}\'')
        
        return 0
    else:
        logger.error("❌ Система не готова к работе")
        logger.info("")
        logger.info("🔧 Для запуска системы:")
        logger.info("   python deploy_system.py")
        logger.info("")
        logger.info("🔧 Для ручной настройки:")
        logger.info("   1. python setup_ollama.py")
        logger.info("   2. python -m app.main")
        logger.info("   3. python load_full_data.py")
        
        return 1


if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=''),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
        colorize=True
    )
    
    # Запуск
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n🛑 Проверка прервана")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Ошибка при проверке: {e}")
        sys.exit(1)
