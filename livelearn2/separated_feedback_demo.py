"""
Демонстрация работы разделенной системы фидбэка.

Этот пример показывает, как новая архитектура предотвращает протечки
между разными намерениями пользователей.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import get_db, init_db
from app.services.separated_rag_pipeline import SeparatedRAGPipeline


async def demo_separated_feedback():
    """Демонстрация разделенной системы фидбэка."""
    
    print("🚀 Демонстрация разделенной системы фидбэка")
    print("=" * 60)
    
    # Инициализация
    init_db()
    db = next(get_db())
    pipeline = SeparatedRAGPipeline()
    
    try:
        # 1. Загружаем тестовые документы
        print("\n📚 Загружаем тестовые документы в docs_index...")
        
        await pipeline.ingest_text(
            db=db,
            text="""
            Машинное обучение — это метод анализа данных, который автоматизирует 
            построение аналитических моделей. Это раздел искусственного интеллекта, 
            основанный на идее, что системы могут учиться на данных.
            """,
            metadata={"topic": "machine_learning", "source": "educational"},
            uri="docs://ml_basics"
        )
        
        await pipeline.ingest_text(
            db=db,
            text="""
            Приготовление пасты — это простой кулинарный процесс. Нужно вскипятить 
            воду, добавить соль, затем пасту. Варить согласно инструкции на упаковке, 
            обычно 8-12 минут до состояния аль денте.
            """,
            metadata={"topic": "cooking", "source": "recipe"},
            uri="docs://pasta_recipe"
        )
        
        print("✅ Документы загружены в docs_index")
        
        # 2. Первый пользователь задает вопрос о машинном обучении
        print("\n👤 Пользователь 1 спрашивает о машинном обучении...")
        
        result_ml_1 = await pipeline.ask(
            question="Что такое машинное обучение?",
            db=db,
            user_id="user_1",
            session_id="session_1"
        )
        
        print(f"🤖 Ответ: {result_ml_1['answer'][:100]}...")
        print(f"📊 Использовано контекстов: {len(result_ml_1['contexts'])}")
        
        # 3. Пользователь 1 дает негативный фидбэк
        print("\n👎 Пользователь 1 дает негативный фидбэк...")
        
        feedback_id_1 = await pipeline.store_user_feedback(
            db=db,
            message_id=result_ml_1["message_id"],
            feedback_label="reject",
            target_doc_ids=[1],  # Документ о ML
            target_chunk_ids=[1],
            user_id="user_1",
            notes="Определение слишком поверхностное, нужно больше деталей"
        )
        
        print(f"✅ Фидбэк сохранен в feedback_store: {feedback_id_1[:8]}...")
        
        # 4. Второй пользователь задает похожий вопрос
        print("\n👤 Пользователь 2 спрашивает о машинном обучении...")
        
        result_ml_2 = await pipeline.ask(
            question="Объясни машинное обучение",
            db=db,
            user_id="user_2",
            session_id="session_2"
        )
        
        print(f"🤖 Ответ: {result_ml_2['answer'][:100]}...")
        print(f"📊 Фидбэк применен: {result_ml_2.get('feedback_applied_count', 0)} раз")
        
        # 5. Пользователь 1 задает вопрос о готовке
        print("\n👤 Пользователь 1 спрашивает о готовке...")
        
        result_cooking = await pipeline.ask(
            question="Как приготовить пасту?",
            db=db,
            user_id="user_1",
            session_id="session_1"
        )
        
        print(f"🤖 Ответ: {result_cooking['answer'][:100]}...")
        print(f"📊 Фидбэк применен: {result_cooking.get('feedback_applied_count', 0)} раз")
        
        # 6. Проверяем целостность разделения
        print("\n🔍 Проверяем целостность разделения...")
        
        stats = await pipeline.get_feedback_stats(db)
        
        print(f"📈 Статистика фидбэк системы:")
        print(f"   • Всего записей фидбэка: {stats['total_feedback_entries']}")
        print(f"   • Локальный scope: {stats['local_scope']}")
        print(f"   • Глобальный scope: {stats['global_scope']}")
        print(f"   • Целостность разделения: {stats['separation_integrity']}")
        
        # 7. Проверяем, что фидбэк не попал в контексты
        print("\n🛡️ Проверяем отсутствие протечек...")
        
        # Проверяем контексты второго ответа
        contamination_found = False
        for context in result_ml_2['contexts']:
            context_text = context.get('text', '')
            if 'слишком поверхностное' in context_text:
                contamination_found = True
                break
        
        if contamination_found:
            print("❌ ОШИБКА: Найдена протечка фидбэка в контексты!")
        else:
            print("✅ Протечек не обнаружено - фидбэк изолирован")
        
        # 8. Проверяем, что фидбэк по ML не влияет на готовку
        cooking_affected = result_cooking.get('feedback_applied_count', 0) > 0
        
        if cooking_affected:
            print("❌ ОШИБКА: Фидбэк по ML повлиял на вопрос о готовке!")
        else:
            print("✅ Гейтинг работает - фидбэк применяется только к релевантным намерениям")
        
        # 9. Демонстрируем коррекцию без контаминации
        print("\n🔧 Демонстрируем коррекцию без контаминации...")
        
        feedback_id_2 = await pipeline.store_user_feedback(
            db=db,
            message_id=result_cooking["message_id"],
            feedback_label="fix",
            target_doc_ids=[2],  # Документ о готовке
            target_chunk_ids=[2],
            user_id="user_1",
            correction_text="Важно добавить, что воду нужно солить ПОСЛЕ закипания, а не до.",
            notes="Уточнение по технике приготовления"
        )
        
        print(f"✅ Коррекция сохранена: {feedback_id_2[:8]}...")
        
        # Задаем тот же вопрос снова
        result_cooking_2 = await pipeline.ask(
            question="Как приготовить пасту?",
            db=db,
            user_id="user_1",
            session_id="session_1"
        )
        
        # Проверяем, что текст коррекции НЕ попал в контексты
        correction_leaked = False
        for context in result_cooking_2['contexts']:
            if 'ПОСЛЕ закипания' in context.get('text', ''):
                correction_leaked = True
                break
        
        if correction_leaked:
            print("❌ ОШИБКА: Текст коррекции попал в контексты!")
        else:
            print("✅ Коррекция изолирована - используется только для переранжирования")
        
        print(f"📊 Коррекционный фидбэк применен: {result_cooking_2.get('feedback_applied_count', 0)} раз")
        
        print("\n🎉 Демонстрация завершена!")
        print("\n📋 Резюме архитектуры:")
        print("   • docs_index: содержит ТОЛЬКО оригинальные документы")
        print("   • feedback_store: изолированное хранение фидбэка с intent_key")
        print("   • Жесткие фильтры: фидбэк применяется только при совпадении намерений")
        print("   • Переранжирование: фидбэк влияет на порядок, но не на содержание")
        print("   • Отсутствие протечек: тексты фидбэка никогда не попадают в контексты")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


async def check_separation_integrity():
    """Проверка целостности разделения в базе данных."""
    
    print("\n🔍 Проверка целостности разделения...")
    
    db = next(get_db())
    
    try:
        from app.models.documents import Chunk
        from app.models.intent_feedback import IntentFeedback
        
        # Проверяем docs_index
        total_chunks = db.query(Chunk).count()
        original_chunks = db.query(Chunk).filter(Chunk.source == "original").count()
        feedback_chunks = db.query(Chunk).filter(Chunk.source == "user_feedback").count()
        
        print(f"📊 Статистика docs_index:")
        print(f"   • Всего чанков: {total_chunks}")
        print(f"   • Оригинальных: {original_chunks}")
        print(f"   • Из фидбэка: {feedback_chunks}")
        
        if feedback_chunks > 0:
            print("❌ КОНТАМИНАЦИЯ ОБНАРУЖЕНА!")
        else:
            print("✅ docs_index чист от контаминации")
        
        # Проверяем feedback_store
        total_feedback = db.query(IntentFeedback).count()
        feedback_with_intent = db.query(IntentFeedback).filter(
            IntentFeedback.intent_key.isnot(None)
        ).count()
        
        print(f"📊 Статистика feedback_store:")
        print(f"   • Всего записей фидбэка: {total_feedback}")
        print(f"   • С intent_key: {feedback_with_intent}")
        
        if total_feedback == feedback_with_intent:
            print("✅ Все записи фидбэка имеют intent_key")
        else:
            print("❌ Некоторые записи фидбэка без intent_key")
        
        # Общая оценка
        integrity_score = 100
        if feedback_chunks > 0:
            integrity_score -= 50
        if total_feedback != feedback_with_intent:
            integrity_score -= 30
        
        print(f"\n🏆 Оценка целостности: {integrity_score}%")
        
    finally:
        db.close()


if __name__ == "__main__":
    print("🔧 Система разделенного фидбэка - Live-Learn RAG")
    print("Демонстрация архитектуры без протечек")
    print()
    
    # Запускаем демонстрацию
    asyncio.run(demo_separated_feedback())
    
    # Проверяем целостность
    asyncio.run(check_separation_integrity())


