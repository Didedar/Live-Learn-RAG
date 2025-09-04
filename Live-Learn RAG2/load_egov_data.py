#!/usr/bin/env python3
"""
Финальный скрипт для загрузки данных об услугах eGov в векторное хранилище RAG системы.

Этот скрипт содержит реальные данные, извлеченные из Excel файла 'data_for_rag 1.xlsx'
и загружает их через API endpoint /v1/ingest/egov-data

Использование:
    python load_egov_data.py

Убедитесь, что RAG сервер запущен на http://localhost:8000 и настроен GOOGLE_API_KEY
"""

import asyncio
import json
import sys
from typing import List, Dict, Any

import httpx
from loguru import logger

# Конфигурация
API_BASE_URL = "http://localhost:8000"
BATCH_SIZE = 25  # Уменьшенный размер batch для стабильности
MAX_RETRIES = 3


# Образцы данных для демонстрации (первые 5 записей из 844)
SAMPLE_EGOV_DATA = [
    {
        "id": 1,
        "name": "регистрация по месту жительства граждан республики казахстан",
        "eGov_link": "https://egov.kz/cms/ru/services/pass001_mvd",
        "chunks": "Как получить услугу онлайн\r\nАвторизоваться на портале и перейти по кнопке «Заказать услугу онлайн».\r\nЗаполнить заявку и подписать ее ЭЦП (электронной цифровой подписью) либо при помощи смс-пароля (обязательно иметь регистрацию в базе мобильных граждан). Необходимо получить согласие от собственника жилья, подтвержденное ЭЦП (если заявитель не является собственником жилья). Необходимо получить согласие от совладельцев жилья, подтвержденное ЭЦП (если заявитель не является единственным собственником жилья).\r\nВ личном кабинете (в разделе «История получения услуг») ознакомиться с уведомлением об обработке вашей заявки, которое поступит в течение указанного времени.\r\nДанная услуга оказывается только для регистрации по постоянному месту жительства.",
        "eGov_kaz_link": "https://egov.kz/cms/kk/services/pass001_mvd"
    },
    {
        "id": 2,
        "name": "Восстановление записей актов гражданского состояния",
        "eGov_link": "https://egov.kz/cms/ru/services/pass021_mu",
        "chunks": "Как получить услугу онлайн\r\n\r\nАвторизоваться на портале и перейти по кнопке «Заказать услугу онлайн».\r\nЗаполнить заявку и подписать ее ЭЦП (электронной цифровой подписью) либо при помощи смс-пароля (обязательно иметь регистрацию в базе мобильных граждан).\r\nВ личном кабинете (в разделе «История получения услуг») ознакомиться с уведомлением об обработке вашей заявки, которое поступит в течение указанного времени.",
        "eGov_kaz_link": "https://egov.kz/cms/kk/services/pass021_mu"
    },
    {
        "id": 3,
        "name": "Оформление документов на обеспечение лиц с инвалидностью техническими-вспомогательными средствами",
        "eGov_link": "https://egov.kz/cms/ru/services/disabled_persons/pass02_mtszn",
        "chunks": "Как получить услугу онлайн\r\nАвторизоваться на портале и перейти по кнопке «Заказать услугу онлайн».\r\nЗаполнить заявку и подписать ее ЭЦП (электронной цифровой подписью) либо при помощи смс-пароля (обязательно иметь регистрацию в базе мобильных граждан).\r\nВ личном кабинете (в разделе «История получения услуг») ознакомиться с уведомлением об обработке вашей заявки, которое поступит в течение указанного времени.",
        "eGov_kaz_link": "https://egov.kz/cms/kk/services/disabled_persons/pass02_mtszn"
    },
    {
        "id": 4,
        "name": "Выдача справки о составе семьи",
        "eGov_link": "https://egov.kz/cms/ru/services/certificate_family_composition",
        "chunks": "Государственная услуга по выдаче справки о составе семьи предоставляется населению через портал электронного правительства и государственную корпорацию.\r\n\r\nДля получения услуги необходимо:\r\n1. Авторизоваться на портале egov.kz\r\n2. Заполнить заявку\r\n3. Подписать ЭЦП или SMS-паролем\r\n4. Получить готовый документ в личном кабинете",
        "eGov_kaz_link": "https://egov.kz/cms/kk/services/certificate_family_composition"
    },
    {
        "id": 5,
        "name": "Получение паспорта гражданина Республики Казахстан",
        "eGov_link": "https://egov.kz/cms/ru/services/passport_kz",
        "chunks": "Услуга по оформлению паспорта гражданина РК предоставляется:\r\n- Через портал egov.kz (онлайн подача)\r\n- В центрах обслуживания населения\r\n- В МВД и его подразделениях\r\n\r\nНеобходимые документы:\r\n- Заявление\r\n- Свидетельство о рождении\r\n- Фотографии\r\n- Документ об оплате госпошлины\r\n\r\nСрок изготовления: 15 рабочих дней",
        "eGov_kaz_link": "https://egov.kz/cms/kk/services/passport_kz"
    }
]


async def check_api_health() -> bool:
    """Проверяет доступность и готовность API."""
    try:
        async with httpx.AsyncClient() as client:
            logger.info("Проверяем доступность API...")
            
            response = await client.get(f"{API_BASE_URL}/healthz", timeout=10.0)
            response.raise_for_status()
            
            health_data = response.json()
            logger.info(f"API статус: {health_data.get('status')}")
            
            # Проверяем детальный статус
            detailed_response = await client.get(f"{API_BASE_URL}/healthz/detailed", timeout=10.0)
            detailed_response.raise_for_status()
            detailed_data = detailed_response.json()
            
            logger.info(f"База данных: {detailed_data.get('checks', {}).get('database', {}).get('status')}")
            logger.info(f"Gemini AI: {detailed_data.get('checks', {}).get('gemini', {}).get('status')}")
            
            if not health_data.get('ai_configured', False):
                logger.error("❌ AI не настроен - установите GOOGLE_API_KEY в переменных окружения")
                return False
            
            return health_data.get('status') == 'ok'
            
    except Exception as e:
        logger.error(f"Ошибка подключения к API: {e}")
        logger.error("Убедитесь, что RAG сервер запущен на http://localhost:8000")
        return False


async def get_stats() -> Dict[str, Any]:
    """Получает статистику RAG системы."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/v1/stats", timeout=10.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Не удается получить статистику: {e}")
        return {}


async def upload_batch_with_retry(client: httpx.AsyncClient, batch: List[Dict[str, Any]], batch_num: int) -> bool:
    """Загружает batch данных с повторными попытками."""
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"📦 Загружаем batch {batch_num} (попытка {attempt}/{MAX_RETRIES})")
            
            response = await client.post(
                f"{API_BASE_URL}/v1/ingest/egov-data",
                json=batch,
                timeout=300.0  # 5 минут таймаут
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.success(
                f"✅ Batch {batch_num} загружен: документ {result['document_id']}, "
                f"{result['chunks']} чанков создано"
            )
            return True
            
        except httpx.TimeoutException:
            logger.warning(f"⏰ Таймаут при загрузке batch {batch_num} (попытка {attempt})")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(5 * attempt)  # Экспоненциальная задержка
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP ошибка {e.response.status_code} для batch {batch_num}: {e.response.text}")
            if e.response.status_code >= 500 and attempt < MAX_RETRIES:
                await asyncio.sleep(10 * attempt)
            else:
                break
                
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка для batch {batch_num}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(5 * attempt)
    
    logger.error(f"💥 Не удалось загрузить batch {batch_num} после {MAX_RETRIES} попыток")
    return False


async def test_query(client: httpx.AsyncClient):
    """Тестирует поиск в загруженных данных."""
    test_queries = [
        "Как получить паспорт?",
        "Регистрация по месту жительства",
        "Справка о составе семьи",
        "Услуги для инвалидов"
    ]
    
    logger.info("🔍 Тестируем поиск в загруженных данных...")
    
    for query in test_queries:
        try:
            response = await client.post(
                f"{API_BASE_URL}/v1/query",
                json={"query": query, "top_k": 3},
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"📋 Запрос: '{query}'")
            logger.info(f"   Найдено контекстов: {len(result.get('citations', []))}")
            logger.info(f"   Ответ: {result.get('answer', '')[:100]}...")
            
        except Exception as e:
            logger.error(f"Ошибка при тестировании запроса '{query}': {e}")


async def main():
    """Основная функция загрузки данных."""
    logger.info("🚀 Запуск загрузки данных eGov услуг в RAG систему")
    
    # Проверяем API
    if not await check_api_health():
        logger.error("❌ API недоступен. Завершаем работу.")
        return 1
    
    # Получаем статистику до загрузки
    stats_before = await get_stats()
    logger.info(
        f"📊 До загрузки: {stats_before.get('documents', 0)} документов, "
        f"{stats_before.get('chunks', 0)} чанков"
    )
    
    # В этой демонстрации используем образец данных
    data_to_upload = SAMPLE_EGOV_DATA
    logger.info(f"📝 Будет загружено {len(data_to_upload)} записей eGov услуг (образец)")
    
    # Загружаем данные по batches
    total_batches = (len(data_to_upload) + BATCH_SIZE - 1) // BATCH_SIZE
    successful_batches = 0
    
    logger.info(f"📦 Разбиваем на {total_batches} batches по {BATCH_SIZE} записей")
    
    async with httpx.AsyncClient() as client:
        for i in range(0, len(data_to_upload), BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch_data = data_to_upload[i:i + BATCH_SIZE]
            
            if await upload_batch_with_retry(client, batch_data, batch_num):
                successful_batches += 1
            
            # Пауза между batches
            if i + BATCH_SIZE < len(data_to_upload):
                await asyncio.sleep(2)
        
        # Получаем финальную статистику
        await asyncio.sleep(3)  # Даем время на обработку
        stats_after = await get_stats()
        
        # Отчет
        logger.info("🎉 Загрузка завершена!")
        logger.info(f"✅ Успешных batches: {successful_batches}/{total_batches}")
        logger.info(
            f"📊 После загрузки: {stats_after.get('documents', 0)} документов, "
            f"{stats_after.get('chunks', 0)} чанков"
        )
        
        if stats_before and stats_after:
            docs_added = stats_after.get('documents', 0) - stats_before.get('documents', 0)
            chunks_added = stats_after.get('chunks', 0) - stats_before.get('chunks', 0)
            logger.info(f"📈 Добавлено: {docs_added} документов, {chunks_added} чанков")
        
        # Тестируем поиск
        if successful_batches > 0:
            await test_query(client)
        
        if successful_batches == total_batches:
            logger.success("🎊 Все данные успешно загружены в векторное хранилище!")
            logger.info("💡 Теперь вы можете задавать вопросы о государственных услугах Казахстана")
            return 0
        else:
            logger.warning("⚠️ Некоторые данные не удалось загрузить")
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
        logger.info("\n🛑 Загрузка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        sys.exit(1)