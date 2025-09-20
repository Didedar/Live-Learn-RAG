#!/usr/bin/env python3
"""
Скрипт для настройки и проверки Ollama для работы с Live-Learn RAG системой.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import httpx
from loguru import logger

# Конфигурация
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"  # Можно изменить на llama3.2:3b для более быстрой работы
ENV_FILE = ".env"


def check_ollama_installed() -> bool:
    """Проверяет, установлен ли Ollama в системе."""
    try:
        result = subprocess.run(["ollama", "--version"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"Ollama установлен: {result.stdout.strip()}")
            return True
        else:
            logger.error("Ollama не найден в системе")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("Ollama не найден в системе")
        return False


def install_ollama_instructions():
    """Выводит инструкции по установке Ollama."""
    logger.info("=" * 60)
    logger.info("📋 ИНСТРУКЦИИ ПО УСТАНОВКЕ OLLAMA:")
    logger.info("=" * 60)
    logger.info("")
    logger.info("🍎 macOS:")
    logger.info("   brew install ollama")
    logger.info("   # или скачайте с https://ollama.ai")
    logger.info("")
    logger.info("🐧 Linux:")
    logger.info("   curl -fsSL https://ollama.ai/install.sh | sh")
    logger.info("")
    logger.info("🪟 Windows:")
    logger.info("   Скачайте установщик с https://ollama.ai")
    logger.info("")
    logger.info("После установки запустите этот скрипт снова.")
    logger.info("=" * 60)


async def check_ollama_running() -> bool:
    """Проверяет, запущен ли сервер Ollama."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            logger.info("✅ Ollama сервер запущен и доступен")
            return True
    except Exception as e:
        logger.warning(f"Ollama сервер не отвечает: {e}")
        return False


def start_ollama_server():
    """Пытается запустить сервер Ollama."""
    try:
        logger.info("🚀 Запускаем Ollama сервер...")

        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        logger.info(f"Ollama сервер запущен с PID: {process.pid}")
        logger.info("Ждем 5 секунд для запуска сервера...")

        import time
        time.sleep(5)

        return True

    except Exception as e:
        logger.error(f"Не удалось запустить Ollama сервер: {e}")
        return False


async def list_available_models() -> Dict[str, Any]:
    """Получает список доступных моделей в Ollama."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [model["name"] for model in data.get("models", [])]

            logger.info(f"📦 Доступные модели в Ollama: {models}")
            return {"models": models, "total": len(models)}

    except Exception as e:
        logger.error(f"Не удалось получить список моделей: {e}")
        return {"models": [], "total": 0}


async def pull_model(model_name: str) -> bool:
    """Скачивает модель в Ollama."""
    try:
        logger.info(f"📥 Скачиваем модель {model_name}...")
        logger.info("⏳ Это может занять несколько минут в зависимости от размера модели...")

        async with httpx.AsyncClient(timeout=600.0) as client:  # 10 минут таймаут
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/pull",
                json={"name": model_name}
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            import json
                            data = json.loads(line)
                            if "status" in data:
                                status = data["status"]
                                if "total" in data and "completed" in data:
                                    total = data["total"]
                                    completed = data["completed"]
                                    percent = (completed / total * 100) if total > 0 else 0
                                    logger.info(f"   📊 {status}: {percent:.1f}%")
                                else:
                                    logger.info(f"   📝 {status}")
                        except:
                            continue

        logger.success(f"✅ Модель {model_name} успешно скачана!")
        return True

    except Exception as e:
        logger.error(f"❌ Не удалось скачать модель {model_name}: {e}")
        return False


async def test_model(model_name: str) -> bool:
    """Тестирует работу модели."""
    try:
        logger.info(f"🧪 Тестируем модель {model_name}...")

        test_prompt = "Скажите 'Привет' если вы работаете корректно."

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": False
                }
            )
            response.raise_for_status()

            result = response.json()
            answer = result.get("response", "").strip()

            logger.info(f"📝 Ответ модели: {answer}")

            if answer:
                logger.success("✅ Модель работает корректно!")
                return True
            else:
                logger.error("❌ Модель вернула пустой ответ")
                return False

    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании модели: {e}")
        return False


def create_env_file():
    """Создает .env файл с настройками для Ollama."""
    env_content = f"""# Live-Learn RAG Configuration

# App settings
APP_NAME="Live-Learn RAG with Llama"
APP_ENV=dev
DEBUG=true

# Database
DB_PATH=./rag.db

# LLM Configuration - Using Ollama with Llama
USE_OLLAMA=true
OLLAMA_URL={OLLAMA_URL}
OLLAMA_MODEL={OLLAMA_MODEL}

# Google AI Settings (disabled when using Ollama)
# GOOGLE_API_KEY=your_google_api_key_here
# LLM_MODEL=gemini-2.0-flash-exp
# EMBEDDING_MODEL=text-embedding-004

# Alternative: Use OpenAI for embeddings (optional)
USE_OPENAI_EMBEDDINGS=false
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG settings
CHUNK_SIZE=400
CHUNK_OVERLAP=40
DEFAULT_TOP_K=6

# Feedback settings
FEEDBACK_PENALTY_WEIGHT=-0.3
FEEDBACK_BOOST_WEIGHT=0.5

# Security (optional)
# API_KEY=your_api_key_here
# API_KEY_HEADER=X-API-Key

# Performance
REQUEST_TIMEOUT=60
MAX_RETRIES=3

# LLM parameters
GEMINI_TEMPERATURE=0.1
GEMINI_MAX_TOKENS=8192
GEMINI_TOP_P=0.8
GEMINI_TOP_K=40
"""

    try:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write(env_content)
        logger.success(f"✅ Создан файл {ENV_FILE} с настройками Ollama")
        return True
    except Exception as e:
        logger.error(f"❌ Не удалось создать {ENV_FILE}: {e}")
        return False


async def main():
    """Основная функция настройки Ollama."""
    global OLLAMA_MODEL  # <-- ВАЖНО: объявляем до первого использования
    logger.info("🚀 Настройка Ollama для Live-Learn RAG системы")
    logger.info("=" * 60)

    # Шаг 1: Проверяем установку Ollama
    logger.info("📋 Шаг 1: Проверка установки Ollama")
    if not check_ollama_installed():
        install_ollama_instructions()
        return 1

    # Шаг 2: Проверяем, запущен ли сервер
    logger.info("=" * 60)
    logger.info("📋 Шаг 2: Проверка сервера Ollama")
    if not await check_ollama_running():
        logger.info("Пытаемся запустить сервер...")
        if not start_ollama_server():
            logger.error("❌ Не удалось запустить Ollama сервер")
            logger.info("💡 Попробуйте запустить вручную: ollama serve")
            return 1

        # Проверяем еще раз после запуска
        await asyncio.sleep(2)
        if not await check_ollama_running():
            logger.error("❌ Сервер все еще не отвечает")
            return 1

    # Шаг 3: Проверяем доступные модели
    logger.info("=" * 60)
    logger.info("📋 Шаг 3: Проверка доступных моделей")
    models_info = await list_available_models()

    # Шаг 4: Скачиваем нужную модель если её нет
    if OLLAMA_MODEL not in models_info["models"]:
        logger.info("=" * 60)
        logger.info(f"📋 Шаг 4: Скачивание модели {OLLAMA_MODEL}")

        logger.info("🤔 Выберите модель для скачивания:")
        logger.info(f"   1. {OLLAMA_MODEL} (рекомендуемая, ~2GB)")
        logger.info("   2. llama3.2:3b (более быстрая, ~2GB)")
        logger.info("   3. llama3.2:1b (самая быстрая, ~1GB)")

        choice = input("Введите номер (по умолчанию 1): ").strip()

        if choice == "2":
            model_to_pull = "llama3.2:3b"
        elif choice == "3":
            model_to_pull = "llama3.2:1b"
        else:
            model_to_pull = OLLAMA_MODEL

        if not await pull_model(model_to_pull):
            logger.error("❌ Не удалось скачать модель")
            return 1

        # Обновляем модель в конфигурации (уже можно — global объявлен в начале)
        OLLAMA_MODEL = model_to_pull
    else:
        logger.success(f"✅ Модель {OLLAMA_MODEL} уже доступна")

    # Шаг 5: Тестируем модель
    logger.info("=" * 60)
    logger.info("📋 Шаг 5: Тестирование модели")
    if not await test_model(OLLAMA_MODEL):
        logger.error("❌ Модель не прошла тест")
        return 1

    # Шаг 6: Создаем .env файл
    logger.info("=" * 60)
    logger.info("📋 Шаг 6: Создание конфигурации")
    if not create_env_file():
        logger.error("❌ Не удалось создать конфигурационный файл")
        return 1

    # Финальный отчет
    logger.info("=" * 60)
    logger.success("🎉 Настройка Ollama завершена успешно!")
    logger.info("")
    logger.info("📋 Что настроено:")
    logger.info(f"   • Ollama сервер: {OLLAMA_URL}")
    logger.info(f"   • Модель: {OLLAMA_MODEL}")
    logger.info(f"   • Конфигурация: {ENV_FILE}")
    logger.info("")
    logger.info("🚀 Следующие шаги:")
    logger.info("   1. Запустите RAG сервер: python -m app.main")
    logger.info("   2. Загрузите данные: python load_full_data.py")
    logger.info("   3. Откройте фронтенд в браузере")
    logger.info("")
    logger.info("💡 Полезные команды:")
    logger.info("   • Список моделей: ollama list")
    logger.info("   • Чат с моделью: ollama run " + OLLAMA_MODEL)
    logger.info("   • Остановка сервера: Ctrl+C в терминале где запущен ollama serve")

    return 0


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
        logger.info("\n🛑 Настройка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        sys.exit(1)
