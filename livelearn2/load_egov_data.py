#!/usr/bin/env python3
"""
Скрипт для загрузки eGov данных из Excel файла через специальный endpoint.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import httpx
from loguru import logger
from pathlib import Path

# Конфигурация
EXCEL_FILE = "data_for_rag (1).xlsx"
API_BASE_URL = "http://localhost:8000"


async def check_api():
    """Проверяем доступность API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/healthz", timeout=5.0)
            if response.status_code == 200:
                logger.info("✅ API доступен")
                return True
            else:
                logger.error(f"❌ API недоступен: {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к API: {e}")
        return False


async def load_excel_data():
    """Загружаем данные из Excel файла."""
    try:
        logger.info(f"📖 Читаем Excel файл: {EXCEL_FILE}")
        
        # Читаем Excel файл
        df = pd.read_excel(EXCEL_FILE)
        logger.info(f"📊 Найдено {len(df)} записей в Excel файле")
        
        # Показываем структуру данных
        logger.info(f"📋 Колонки: {list(df.columns)}")
        
        # Фильтруем только записи с непустыми данными
        df_clean = df.dropna(subset=['name'])  # Только name обязательно
        df_clean = df_clean[df_clean['name'].str.strip() != '']
        
        # Заполняем пустые chunks пустой строкой
        df_clean['chunks'] = df_clean['chunks'].fillna('')
        
        logger.info(f"🧹 После очистки осталось {len(df_clean)} записей с названиями")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"❌ Ошибка чтения Excel файла: {e}")
        return None


async def ingest_egov_data(df):
    """Загружаем eGov данные через специальный endpoint."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Конвертируем DataFrame в список словарей
            data_list = []
            
            for index, row in df.iterrows():
                record = {
                    'id': int(row.get('id', index+1)) if pd.notna(row.get('id')) else index+1,
                    'name': str(row.get('name', '')).strip(),
                    'eGov_link': str(row.get('eGov_link', '')).strip() if pd.notna(row.get('eGov_link')) else '',
                    'chunks': str(row.get('chunks', '')).strip() if pd.notna(row.get('chunks')) else '',
                    'eGov_kaz_link': str(row.get('eGov_kaz_link', '')).strip() if pd.notna(row.get('eGov_kaz_link')) else ''
                }
                
                # Убираем записи где chunks пустой или 'nan'
                if record['chunks'] and record['chunks'] != 'nan' and len(record['chunks']) > 10:
                    data_list.append(record)
            
            logger.info(f"📦 Подготовлено {len(data_list)} записей с содержательными данными")
            
            if not data_list:
                logger.error("❌ Нет записей для загрузки")
                return 0
            
            # Отправляем данные
            logger.info("🚀 Отправляем данные в API...")
            
            response = await client.post(
                f"{API_BASE_URL}/api/v1/ingest/egov-data",
                json=data_list
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Успешно загружено! Ответ: {result}")
                return len(data_list)
            else:
                logger.error(f"❌ Ошибка загрузки: {response.status_code} - {response.text}")
                return 0
            
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки eGov данных: {e}")
        return 0


async def get_stats():
    """Получаем статистику загруженных данных."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/v1/stats")
            if response.status_code == 200:
                stats = response.json()
                logger.info(f"📊 Статистика: {stats.get('total_documents', stats.get('documents', 0))} документов, {stats.get('total_chunks', stats.get('chunks', 0))} чанков")
                return stats
            else:
                logger.warning("⚠️ Не удалось получить статистику")
                return None
    except Exception as e:
        logger.warning(f"⚠️ Ошибка получения статистики: {e}")
        return None


async def main():
    """Основная функция."""
    logger.info("🚀 Начинаем загрузку eGov данных из Excel")
    
    # 1. Проверяем API
    if not await check_api():
        logger.error("❌ API недоступен. Убедитесь, что сервер запущен.")
        return
    
    # 2. Загружаем данные из Excel
    df = await load_excel_data()
    if df is None or len(df) == 0:
        logger.error("❌ Не удалось загрузить данные из Excel или файл пустой")
        return
    
    # 3. Показываем статистику до загрузки
    logger.info("📊 Статистика ДО загрузки:")
    await get_stats()
    
    # 4. Загружаем данные в систему
    success_count = await ingest_egov_data(df)
    
    # 5. Показываем статистику после загрузки
    logger.info("📊 Статистика ПОСЛЕ загрузки:")
    await get_stats()
    
    if success_count > 0:
        logger.info(f"✅ Успешно! Загружено {success_count} услуг eGov")
        logger.info("🎯 Векторное хранилище содержит только чистые изначальные данные")
    else:
        logger.error("❌ Не удалось загрузить данные")


if __name__ == "__main__":
    # Проверяем наличие pandas
    try:
        import pandas as pd
    except ImportError:
        logger.error("❌ Требуется установить pandas: pip install pandas openpyxl")
        sys.exit(1)
    
    asyncio.run(main())


