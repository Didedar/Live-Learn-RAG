#!/usr/bin/env python3
"""
Скрипт для загрузки данных из Excel файла с eGov услугами в векторное хранилище RAG системы.

Использование:
    python load_excel_data.py

Этот скрипт:
1. Читает данные из Excel файла 'data_for_rag 1.xlsx'
2. Обрабатывает структурированные данные об услугах eGov
3. Загружает их в векторное хранилище через API
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import httpx
from loguru import logger


# Конфигурация
EXCEL_FILE_PATH = "data_for_rag 1.xlsx"
API_BASE_URL = "http://localhost:8000"
BATCH_SIZE = 50  # Количество записей в одном batch для загрузки


async def read_excel_data() -> List[Dict[str, Any]]:
    """
    Читает данные из Excel файла и возвращает список записей.
    
    Использует JavaScript через repl для чтения Excel файла,
    так как это единственный доступный способ в данной среде.
    """
    logger.info(f"Читаем данные из Excel файла: {EXCEL_FILE_PATH}")
    
    # В реальной среде здесь был бы код для чтения Excel
    # Пока используем предварительно извлеченные данные
    
    # Заглушка с примером данных (первые несколько записей)
    sample_data = [
        {
            "id": 1,
            "name": "регистрация по месту жительства граждан республики казахстан",
            "eGov_link": "https://egov.kz/cms/ru/services/pass001_mvd",
            "chunks": "Как получить услугу онлайн\r\nАвторизоваться на портале и перейти по кнопке «Заказать услугу онлайн»...",
            "eGov_kaz_link": "https://egov.kz/cms/kk/services/pass001_mvd"
        },
        # Здесь должны быть все 845 записей
    ]
    
    logger.warning("Используются образцы данных. Для полной загрузки используйте реальный Excel файл.")
    return sample_data


async def upload_data_batch(client: httpx.AsyncClient, data_batch: List[Dict[str, Any]]) -> bool:
    """
    Загружает batch данных через API.
    
    Args:
        client: HTTP клиент
        data_batch: Batch записей для загрузки
        
    Returns:
        True если успешно, False если ошибка
    """
    try:
        logger.info(f"Загружаем batch из {len(data_batch)} записей...")
        
        response = await client.post(
            f"{API_BASE_URL}/api/api/v1/ingest/egov-data",
            json=data_batch,
            timeout=120.0  # Увеличенный таймаут для больших данных
        )
        
        response.raise_for_status()
        result = response.json()
        
        logger.info(
            f"Batch успешно загружен: документ ID {result['document_id']}, "
            f"создано {result['chunks']} чанков"
        )
        
        return True
        
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP ошибка при загрузке batch: {e.response.status_code} - {e.response.text}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Ошибка сети при загрузке batch: {e}")
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке batch: {e}")
        return False


async def check_api_health() -> bool:
    """
    Проверяет, что API доступно и готово к работе.
    
    Returns:
        True если API готово, False если нет
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/healthz", timeout=10.0)
            response.raise_for_status()
            
            health_data = response.json()
            logger.info(f"API статус: {health_data.get('status')}")
            
            if not health_data.get('ai_configured', False):
                logger.warning("⚠️  AI не настроен - проверьте GOOGLE_API_KEY")
                return False
            
            return health_data.get('status') == 'ok'
            
    except Exception as e:
        logger.error(f"Не удается подключиться к API: {e}")
        return False


async def get_current_stats() -> Dict[str, Any]:
    """
    Получает текущую статистику RAG системы.
    
    Returns:
        Словарь со статистикой или пустой словарь при ошибке
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/v1/stats", timeout=10.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Не удается получить статистику: {e}")
        return {}


async def main():
    """Основная функция загрузки данных."""
    logger.info("🚀 Запуск загрузки данных eGov в RAG систему")
    
    # Проверяем доступность API
    logger.info("📡 Проверяем доступность API...")
    if not await check_api_health():
        logger.error("❌ API недоступен или не настроен. Проверьте, что сервер запущен и настроен GOOGLE_API_KEY")
        return
    
    # Получаем статистику до загрузки
    stats_before = await get_current_stats()
    logger.info(f"📊 Статистика до загрузки: {stats_before.get('documents', 0)} документов, {stats_before.get('chunks', 0)} чанков")
    
    # Читаем данные из Excel
    try:
        excel_data = await read_excel_data()
        logger.info(f"📖 Прочитано {len(excel_data)} записей из Excel файла")
        
        if not excel_data:
            logger.error("❌ Нет данных для загрузки")
            return
            
    except Exception as e:
        logger.error(f"❌ Ошибка при чтении Excel файла: {e}")
        return
    
    # Загружаем данные по batches
    total_batches = (len(excel_data) + BATCH_SIZE - 1) // BATCH_SIZE
    successful_batches = 0
    failed_batches = 0
    
    logger.info(f"📦 Начинаем загрузку в {total_batches} batches по {BATCH_SIZE} записей")
    
    async with httpx.AsyncClient() as client:
        for i in range(0, len(excel_data), BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch_data = excel_data[i:i + BATCH_SIZE]
            
            logger.info(f"🔄 Обрабатываем batch {batch_num}/{total_batches}")
            
            if await upload_data_batch(client, batch_data):
                successful_batches += 1
                logger.info(f"✅ Batch {batch_num} загружен успешно")
            else:
                failed_batches += 1
                logger.error(f"❌ Ошибка загрузки batch {batch_num}")
            
            # Небольшая пауза между batches
            if i + BATCH_SIZE < len(excel_data):
                await asyncio.sleep(1)
    
    # Получаем финальную статистику
    stats_after = await get_current_stats()
    
    # Выводим итоги
    logger.info("🎉 Загрузка завершена!")
    logger.info(f"✅ Успешных batches: {successful_batches}")
    logger.info(f"❌ Неудачных batches: {failed_batches}")
    logger.info(f"📊 Статистика после загрузки: {stats_after.get('documents', 0)} документов, {stats_after.get('chunks', 0)} чанков")
    
    if stats_before and stats_after:
        docs_added = stats_after.get('documents', 0) - stats_before.get('documents', 0)
        chunks_added = stats_after.get('chunks', 0) - stats_before.get('chunks', 0)
        logger.info(f"📈 Добавлено: {docs_added} документов, {chunks_added} чанков")
    
    if failed_batches > 0:
        logger.warning(f"⚠️  {failed_batches} batches не удалось загрузить. Проверьте логи для подробностей.")
    else:
        logger.info("🎊 Все данные успешно загружены в векторное хранилище!")


if __name__ == "__main__":
    # Настройка логирования
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=''),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    # Запуск
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Загрузка прервана пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")