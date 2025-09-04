#!/usr/bin/env python3
"""Синхронное исправление базы данных без async/await."""

import os
import sys
from pathlib import Path
import shutil
from datetime import datetime

def main():
    print("🔧 Исправляем базу данных RAG системы...")

    # 1. Создаем резервную копию
    db_path = Path("rag.db")
    if db_path.exists():
        backup = f"rag_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(db_path, backup)
        print(f"📦 Создана резервная копия: {backup}")

    # 2. Исправляем модель documents.py
    print("🔧 Исправляем app/models/documents.py...")
    doc_model = Path("app/models/documents.py")
    
    if not doc_model.exists():
        print("❌ Файл app/models/documents.py не найден!")
        return False
    
    try:
        content = doc_model.read_text(encoding='utf-8')
        original_content = content
        
        # Добавляем ForeignKey импорт если его нет
        if "ForeignKey" not in content:
            content = content.replace(
                "from sqlalchemy import DateTime, Integer, JSON, String, Text, func",
                "from sqlalchemy import DateTime, Integer, JSON, String, Text, func, ForeignKey"
            )
            print("  ✅ Добавлен импорт ForeignKey")
        
        # Исправляем document_id с ForeignKey
        if 'ForeignKey("documents.id")' not in content:
            content = content.replace(
                'document_id: Mapped[int] = mapped_column(Integer, index=True)',
                'document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id"), index=True)'
            )
            print("  ✅ Добавлен ForeignKey в Chunk.document_id")
        
        if content != original_content:
            doc_model.write_text(content, encoding='utf-8')
            print("✅ Исправлен app/models/documents.py")
        else:
            print("✅ app/models/documents.py уже корректен")
            
    except Exception as e:
        print(f"❌ Ошибка при исправлении documents.py: {e}")
        return False

    # 3. Исправляем модель feedback.py
    print("🔧 Исправляем app/models/feedback.py...")
    feedback_model = Path("app/models/feedback.py")
    
    if feedback_model.exists():
        try:
            content = feedback_model.read_text(encoding='utf-8')
            original_content = content
            
            # Добавляем импорты
            if "ForeignKey" not in content:
                content = content.replace(
                    "from sqlalchemy import JSON, DateTime, Enum as SQLEnum, Float, Integer, String, Text, func",
                    "from sqlalchemy import JSON, DateTime, Enum as SQLEnum, Float, ForeignKey, Integer, String, Text, func, Boolean"
                )
                print("  ✅ Добавлены импорты ForeignKey и Boolean")
            
            # Исправляем все ForeignKey связи
            replacements = [
                ('message_id: Mapped[str] = mapped_column(String(36), index=True)',
                 'message_id: Mapped[str] = mapped_column(String(36), ForeignKey("message_sessions.id"), index=True)'),
                ('feedback_event_id: Mapped[str] = mapped_column(String(36), index=True)',
                 'feedback_event_id: Mapped[str] = mapped_column(String(36), ForeignKey("feedback_events.id"), index=True)'),
                ('chunk_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)',
                 'chunk_id: Mapped[int] = mapped_column(Integer, ForeignKey("chunks.id"), unique=True, index=True)'),
                ('is_deprecated: Mapped[bool] = mapped_column(default=False)',
                 'is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False)')
            ]
            
            changes_made = 0
            for old_text, new_text in replacements:
                if old_text in content and new_text not in content:
                    content = content.replace(old_text, new_text)
                    changes_made += 1
            
            if content != original_content:
                feedback_model.write_text(content, encoding='utf-8')
                print(f"✅ Исправлен app/models/feedback.py ({changes_made} изменений)")
            else:
                print("✅ app/models/feedback.py уже корректен")
                
        except Exception as e:
            print(f"❌ Ошибка при исправлении feedback.py: {e}")
            return False
    else:
        print("⚠️ Файл app/models/feedback.py не найден")

    # 4. Удаляем старую базу и создаем новую
    print("🗄️ Пересоздаем базу данных...")
    
    if db_path.exists():
        db_path.unlink()
        print("🗑️ Удалена старая база данных")

    try:
        # Добавляем текущую директорию в путь Python
        sys.path.insert(0, ".")
        
        # Импортируем и инициализируем базу
        from app.database import init_db
        init_db()
        print("✅ База данных пересоздана!")
        
    except Exception as e:
        print(f"❌ Ошибка при создании базы: {e}")
        print("Убедитесь что установлены зависимости: pip install -r requirements.txt")
        return False

    # 5. Тестируем базовую функциональность
    print("🧪 Тестируем связи в базе данных...")
    
    try:
        from app.database import get_db_session
        from app.models.documents import Document, Chunk
        
        with get_db_session() as db:
            # Создаем тестовый документ
            doc = Document(
                uri="test://fix_test", 
                doc_metadata={"title": "Test Document"}
            )
            db.add(doc)
            db.flush()  # Получаем ID документа
            
            # Создаем тестовый чанк с правильной связью
            chunk = Chunk(
                document_id=doc.id,  # Теперь должно работать!
                ordinal=0,
                content="Test content for database fix",
                embedding=[]
            )
            db.add(chunk)
            
            # Коммитим изменения
            db.commit()
            
            print("✅ Тест связей прошел успешно!")
            print(f"  📊 Создан документ ID: {doc.id}")
            print(f"  📊 Создан чанк ID: {chunk.id}")
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        print("База данных создана, но связи могут работать неправильно")
        return False

    # 6. Финальные инструкции
    print("\n🎉 ИСПРАВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("\n📋 Следующие шаги:")
    print("1. Убедитесь что GOOGLE_API_KEY настроен в .env файле:")
    print("   echo 'GOOGLE_API_KEY=your_key_here' >> .env")
    print("2. Запустите сервер:")
    print("   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    print("3. Протестируйте API:")
    print("   curl http://localhost:8000/healthz")
    print("4. Протестируйте ask endpoint:")
    print('   curl -X POST "http://localhost:8000/api/v1/feedback/ask" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"question":"What is AI?","session_id":"test"}\'')
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Все готово! Запускайте сервер.")
            sys.exit(0)
        else:
            print("\n❌ Исправление не завершено. Проверьте ошибки выше.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)