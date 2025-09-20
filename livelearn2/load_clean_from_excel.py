#!/usr/bin/env python3
"""
Скрипт для загрузки чистых данных из Excel файла data_for_rag (1).xlsx
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
        df_clean = df.dropna(subset=['name', 'chunks'])
        df_clean = df_clean[df_clean['name'].str.strip() != '']
        df_clean = df_clean[df_clean['chunks'].str.strip() != '']
        
        logger.info(f"🧹 После очистки осталось {len(df_clean)} записей с данными")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"❌ Ошибка чтения Excel файла: {e}")
        return None


async def ingest_data_to_api(df):
    """Загружаем данные в RAG систему через API."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            success_count = 0
            
            for index, row in df.iterrows():
                try:
                    # Формируем текст для ingestion
                    service_name = str(row.get('name', '')).strip()
                    service_chunks = str(row.get('chunks', '')).strip()
                    egov_link = str(row.get('eGov_link', '')).strip()
                    egov_kaz_link = str(row.get('eGov_kaz_link', '')).strip()
                    
                    if not service_name or not service_chunks:
                        logger.warning(f"⚠️ Пропускаем запись {index+1}: отсутствуют обязательные данные")
                        continue
                    
                    # Создаем полный текст документа
                    full_text = f"""Услуга: {service_name}

Описание и инструкции:
{service_chunks}"""
                    
                    if egov_link and egov_link.strip() != 'nan':
                        full_text += f"\n\nСсылка на портал eGov: {egov_link}"
                    
                    if egov_kaz_link and egov_kaz_link.strip() != 'nan':
                        full_text += f"\nСсылка на казахском языке: {egov_kaz_link}"
                    
                    # Формируем метаданные
                    metadata = {
                        "service_id": int(row.get('id', index+1)) if pd.notna(row.get('id')) else index+1,
                        "service_name": service_name,
                        "source": "egov_services",
                        "type": "government_service",
                        "language": "russian"
                    }
                    
                    if egov_link and egov_link.strip() != 'nan':
                        metadata["egov_link"] = egov_link
                    
                    if egov_kaz_link and egov_kaz_link.strip() != 'nan':
                        metadata["egov_kaz_link"] = egov_kaz_link
                    
                    # Отправляем в API
                    payload = {
                        "text": full_text,
                        "metadata": metadata,
                        "uri": f"egov_service_{metadata['service_id']}"
                    }
                    
                    response = await client.post(
                        f"{API_BASE_URL}/api/v1/ingest",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        success_count += 1
                        if success_count % 10 == 0:
                            logger.info(f"✅ Загружено {success_count}/{len(df)} услуг...")
                        else:
                            logger.debug(f"✅ Загружена услуга {index+1}: {service_name[:50]}...")
                    else:
                        logger.error(f"❌ Ошибка загрузки услуги {index+1}: {response.status_code} - {response.text}")
                
                except Exception as e:
                    logger.error(f"❌ Ошибка обработки записи {index+1}: {e}")
                    continue
            
            logger.info(f"🎉 Загрузка завершена! Успешно загружено: {success_count}/{len(df)} услуг")
            return success_count
            
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки данных в API: {e}")
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
    logger.info("🚀 Начинаем загрузку чистых данных из Excel")
    
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
    success_count = await ingest_data_to_api(df)
    
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


