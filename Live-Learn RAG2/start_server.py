#!/usr/bin/env python3
"""
Скрипт для запуска RAG сервера с проверками.
Использование: python start_server.py
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Проверка версии Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Проверка зависимостей."""
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import google.generativeai
        logger.info("✅ Core dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Проверка .env файла."""
    env_path = Path(".env")
    
    if not env_path.exists():
        logger.warning("⚠️  .env file not found")
        logger.info("Creating .env from example...")
        
        example_path = Path(".env.example")
        if example_path.exists():
            try:
                with open(example_path, 'r') as src, open(env_path, 'w') as dst:
                    dst.write(src.read())
                logger.info("✅ Created .env file from example")
                logger.warning("❗ Please edit .env and add your GOOGLE_API_KEY")
                return False
            except Exception as e:
                logger.error(f"Failed to create .env: {e}")
                return False
        else:
            logger.error("❌ .env.example not found")
            return False
    
    # Проверка Google API key
    from dotenv import load_dotenv
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key or google_key == "your_google_api_key_here":
        logger.error("❌ GOOGLE_API_KEY not configured in .env")
        logger.info("Get your key at: https://aistudio.google.com/app/apikey")
        return False
    
    logger.info("✅ Configuration looks good")
    return True

def check_project_structure():
    """Проверка структуры проекта."""
    required_paths = [
        "app/main.py",
        "app/config.py", 
        "app/database.py",
        "app/services/rag_pipeline.py",
        "app/api/v1/feedback.py"
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            logger.error(f"❌ Missing file: {path}")
            return False
    
    logger.info("✅ Project structure OK")
    return True

def run_basic_test():
    """Запуск базового теста системы."""
    logger.info("🧪 Running basic system test...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", 
             "import asyncio; from test_rag_system import test_basic_functionality; "
             "exit(0 if asyncio.run(test_basic_functionality()) else 1)"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            logger.info("✅ Basic test passed")
            return True
        else:
            logger.error("❌ Basic test failed")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Basic test timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def start_server(host="0.0.0.0", port=8000, reload=True):
    """Запуск сервера."""
    logger.info(f"🚀 Starting server at http://{host}:{port}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("👋 Server stopped")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        sys.exit(1)

def main():
    """Главная функция."""
    print("🔧 RAG System Startup Checker")
    print("=" * 40)
    
    checks_passed = 0
    total_checks = 5
    
    # Проверки
    if check_python_version():
        checks_passed += 1
    
    if check_dependencies():
        checks_passed += 1
    
    if check_project_structure():
        checks_passed += 1
    
    if check_env_file():
        checks_passed += 1
    
    if run_basic_test():
        checks_passed += 1
    
    print("\n" + "=" * 40)
    print(f"✅ Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed < total_checks:
        logger.error("❌ Some checks failed. Fix issues before starting server.")
        
        # Показываем советы по исправлению
        print("\n🔧 Quick fixes:")
        print("1. pip install -r requirements.txt")
        print("2. cp .env.example .env")
        print("3. Edit .env and add your GOOGLE_API_KEY")
        print("4. Run: python test_rag_system.py")
        
        sys.exit(1)
    
    # Запуск сервера
    print("\n🚀 All checks passed! Starting server...")
    print("\n📋 Available endpoints:")
    print("   • Health: http://localhost:8000/healthz")
    print("   • Docs: http://localhost:8000/docs")  
    print("   • Ask: POST http://localhost:8000/api/v1/feedback/ask")
    print("   • Ingest: POST http://localhost:8000/api/v1/ingest")
    print("\n💡 Test with: python test_rag_system.py (in another terminal)")
    
    try:
        start_server()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)