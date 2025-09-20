#!/usr/bin/env python3
"""
Тестирование RAG системы на основе данных обратной связи.
Анализ ошибок и улучшение качества ответов.
"""

import asyncio
import json
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import time

from loguru import logger
from sqlalchemy.orm import Session

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from app.database import SessionLocal
from app.services.enhanced_rag_pipeline import EnhancedRAGPipeline
from app.services.rag_pipeline import EnhancedRAGPipeline as OriginalRAGPipeline
from app.services.strict_rag_pipeline import StrictRAGPipeline
from app.services.separated_rag_pipeline import SeparatedRAGPipeline


@dataclass
class TestResult:
    """Результат тестирования одного вопроса."""
    question: str
    expected_rating: int
    actual_answer: str
    expected_answer: str
    comment: str
    
    # Метрики качества
    answer_length: int
    response_time: float
    retrieval_score: float
    can_answer: bool
    contexts_count: int
    
    # Анализ ошибок
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    improvement_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.improvement_suggestions is None:
            self.improvement_suggestions = []


@dataclass
class RAGTestSuite:
    """Набор тестов для RAG системы."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    avg_response_time: float
    avg_retrieval_score: float
    
    # Анализ ошибок по типам
    error_patterns: Dict[str, int]
    low_rating_cases: List[TestResult]
    improvement_suggestions: List[str]


class RAGTester:
    """Класс для тестирования RAG системы."""
    
    def __init__(self):
        self.pipelines = {
            'enhanced': EnhancedRAGPipeline(),
            'original': OriginalRAGPipeline(),
            'strict': StrictRAGPipeline(),
            'separated': SeparatedRAGPipeline()
        }
        
    async def load_feedback_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Загрузка данных обратной связи."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Загружено {len(data)} записей обратной связи")
            return data
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return []
    
    async def test_single_question(
        self,
        pipeline_name: str,
        pipeline,
        db: Session,
        feedback_item: Dict[str, Any]
    ) -> TestResult:
        """Тестирование одного вопроса."""
        try:
            question = feedback_item['question']
            expected_rating = feedback_item['rating']
            expected_answer = feedback_item.get('correct_answer', 'N/A')
            comment = feedback_item.get('comment', '')
            
            logger.debug(f"Тестирование: {question[:50]}...")
            
            # Измеряем время ответа
            start_time = time.time()
            
            # Получаем ответ от RAG
            response = await pipeline.ask(
                question=question,
                db=db,
                top_k=4
            )
            
            response_time = time.time() - start_time
            
            # Анализируем результат
            actual_answer = response.get('answer', '')
            can_answer = response.get('can_answer', True)
            contexts = response.get('contexts', [])
            max_score = response.get('max_score', 0.0)
            
            # Определяем тип ошибки
            error_type, error_details = self._analyze_error(
                expected_rating, actual_answer, expected_answer, comment
            )
            
            # Генерируем предложения по улучшению
            suggestions = self._generate_improvement_suggestions(
                error_type, expected_rating, actual_answer, comment
            )
            
            return TestResult(
                question=question,
                expected_rating=expected_rating,
                actual_answer=actual_answer,
                expected_answer=expected_answer,
                comment=comment,
                answer_length=len(actual_answer),
                response_time=response_time,
                retrieval_score=max_score,
                can_answer=can_answer,
                contexts_count=len(contexts),
                error_type=error_type,
                error_details=error_details,
                improvement_suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Ошибка при тестировании вопроса: {e}")
            return TestResult(
                question=feedback_item.get('question', ''),
                expected_rating=feedback_item.get('rating', 0),
                actual_answer='',
                expected_answer='',
                comment='',
                answer_length=0,
                response_time=0.0,
                retrieval_score=0.0,
                can_answer=False,
                contexts_count=0,
                error_type='system_error',
                error_details=str(e)
            )
    
    def _analyze_error(
        self,
        expected_rating: int,
        actual_answer: str,
        expected_answer: str,
        comment: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Анализ типа ошибки."""
        if expected_rating >= 4:
            return None, None  # Хороший ответ
        
        # Анализируем комментарии для определения типа ошибки
        comment_lower = comment.lower()
        
        if 'неверен' in comment_lower or 'ошибочн' in comment_lower:
            return 'factual_error', 'Фактическая ошибка в ответе'
        
        if 'сумбурный' in comment_lower or 'непонятн' in comment_lower:
            return 'clarity_issue', 'Проблемы с ясностью и структурой ответа'
        
        if 'слишком много' in comment_lower and 'если' in comment_lower:
            return 'uncertainty', 'Слишком много неопределенности в ответе'
        
        if 'игнорирует' in comment_lower or 'не учитывает' in comment_lower:
            return 'incomplete_analysis', 'Неполный анализ или игнорирование важных аспектов'
        
        if len(actual_answer) < 50:
            return 'too_short', 'Слишком краткий ответ'
        
        if expected_rating <= 2:
            return 'low_quality', 'Низкое качество ответа'
        
        return 'unknown', 'Неопределенный тип ошибки'
    
    def _generate_improvement_suggestions(
        self,
        error_type: Optional[str],
        expected_rating: int,
        actual_answer: str,
        comment: str
    ) -> List[str]:
        """Генерация предложений по улучшению."""
        suggestions = []
        
        if error_type == 'factual_error':
            suggestions.extend([
                'Улучшить качество поиска релевантных документов',
                'Добавить проверку фактов через дополнительные источники',
                'Использовать более строгие критерии отбора контекста'
            ])
        
        elif error_type == 'clarity_issue':
            suggestions.extend([
                'Улучшить структурирование ответа',
                'Добавить четкие разделы и пункты',
                'Использовать более простые формулировки'
            ])
        
        elif error_type == 'uncertainty':
            suggestions.extend([
                'Снизить температуру генерации для более определенных ответов',
                'Улучшить confidence scoring',
                'Добавить фильтрацию неопределенных формулировок'
            ])
        
        elif error_type == 'incomplete_analysis':
            suggestions.extend([
                'Увеличить количество контекстов для анализа',
                'Улучшить алгоритм поиска релевантных документов',
                'Добавить проверку полноты ответа'
            ])
        
        elif error_type == 'too_short':
            suggestions.extend([
                'Увеличить минимальную длину ответа',
                'Добавить больше деталей и объяснений',
                'Использовать дополнительные контексты'
            ])
        
        if expected_rating <= 2:
            suggestions.append('Критический случай - требует особого внимания')
        
        return suggestions
    
    async def run_test_suite(
        self,
        pipeline_name: str,
        feedback_data: List[Dict[str, Any]],
        max_tests: Optional[int] = None
    ) -> RAGTestSuite:
        """Запуск полного набора тестов."""
        logger.info(f"Запуск тестирования RAG pipeline: {pipeline_name}")
        
        pipeline = self.pipelines[pipeline_name]
        test_results = []
        
        # Ограничиваем количество тестов если указано
        test_data = feedback_data[:max_tests] if max_tests else feedback_data
        
        # Создаем сессию базы данных
        db = SessionLocal()
        
        try:
            for i, feedback_item in enumerate(test_data):
                logger.info(f"Тест {i+1}/{len(test_data)}")
                
                result = await self.test_single_question(
                    pipeline_name, pipeline, db, feedback_item
                )
                test_results.append(result)
                
                # Небольшая пауза между запросами
                await asyncio.sleep(0.1)
        finally:
            db.close()
        
        # Анализируем результаты
        return self._analyze_test_results(test_results)
    
    def _analyze_test_results(self, results: List[TestResult]) -> RAGTestSuite:
        """Анализ результатов тестирования."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.expected_rating >= 4)
        failed_tests = total_tests - passed_tests
        
        # Средние метрики
        avg_response_time = sum(r.response_time for r in results) / total_tests
        avg_retrieval_score = sum(r.retrieval_score for r in results) / total_tests
        
        # Анализ ошибок
        error_patterns = {}
        low_rating_cases = []
        
        for result in results:
            if result.expected_rating <= 2:
                low_rating_cases.append(result)
            
            if result.error_type:
                error_patterns[result.error_type] = error_patterns.get(result.error_type, 0) + 1
        
        # Общие предложения по улучшению
        improvement_suggestions = self._generate_overall_suggestions(results, error_patterns)
        
        return RAGTestSuite(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            avg_response_time=avg_response_time,
            avg_retrieval_score=avg_retrieval_score,
            error_patterns=error_patterns,
            low_rating_cases=low_rating_cases,
            improvement_suggestions=improvement_suggestions
        )
    
    def _generate_overall_suggestions(
        self,
        results: List[TestResult],
        error_patterns: Dict[str, int]
    ) -> List[str]:
        """Генерация общих предложений по улучшению."""
        suggestions = []
        
        # Анализ наиболее частых ошибок
        if error_patterns:
            most_common_error = max(error_patterns.items(), key=lambda x: x[1])
            suggestions.append(f"Приоритет: исправление '{most_common_error[0]}' (встречается {most_common_error[1]} раз)")
        
        # Анализ времени ответа
        slow_responses = [r for r in results if r.response_time > 5.0]
        if slow_responses:
            suggestions.append(f"Оптимизировать скорость ответа ({len(slow_responses)} медленных запросов)")
        
        # Анализ качества поиска
        low_retrieval = [r for r in results if r.retrieval_score < 0.3]
        if low_retrieval:
            suggestions.append(f"Улучшить алгоритм поиска ({len(low_retrieval)} случаев низкого retrieval score)")
        
        # Анализ длины ответов
        short_answers = [r for r in results if r.answer_length < 100]
        if short_answers:
            suggestions.append(f"Увеличить детализацию ответов ({len(short_answers)} коротких ответов)")
        
        return suggestions
    
    def save_results(self, pipeline_name: str, test_suite: RAGTestSuite, output_dir: str = "test_results"):
        """Сохранение результатов тестирования."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Сохраняем общие результаты
        results_file = output_path / f"{pipeline_name}_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(test_suite), f, ensure_ascii=False, indent=2)
        
        # Сохраняем детальный отчет
        report_file = output_path / f"{pipeline_name}_detailed_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Отчет по тестированию RAG Pipeline: {pipeline_name} ===\n\n")
            f.write(f"Всего тестов: {test_suite.total_tests}\n")
            f.write(f"Успешных: {test_suite.passed_tests}\n")
            f.write(f"Неудачных: {test_suite.failed_tests}\n")
            f.write(f"Среднее время ответа: {test_suite.avg_response_time:.2f}с\n")
            f.write(f"Средний retrieval score: {test_suite.avg_retrieval_score:.3f}\n\n")
            
            f.write("=== Паттерны ошибок ===\n")
            for error_type, count in test_suite.error_patterns.items():
                f.write(f"{error_type}: {count} случаев\n")
            
            f.write("\n=== Предложения по улучшению ===\n")
            for suggestion in test_suite.improvement_suggestions:
                f.write(f"- {suggestion}\n")
            
            f.write(f"\n=== Критические случаи (рейтинг ≤ 2) ===\n")
            for case in test_suite.low_rating_cases[:10]:  # Первые 10 случаев
                f.write(f"\nВопрос: {case.question}\n")
                f.write(f"Ожидаемый рейтинг: {case.expected_rating}\n")
                f.write(f"Тип ошибки: {case.error_type}\n")
                f.write(f"Комментарий: {case.comment}\n")
                f.write("-" * 50 + "\n")
        
        logger.info(f"Результаты сохранены в {output_path}")


async def main():
    """Основная функция тестирования."""
    logger.info("Запуск тестирования RAG системы на основе feedback данных")
    
    tester = RAGTester()
    
    # Загружаем данные обратной связи
    feedback_data = await tester.load_feedback_data("feedback_data (1).json")
    if not feedback_data:
        logger.error("Не удалось загрузить данные обратной связи")
        return
    
    # Фильтруем только проблемные случаи (рейтинг ≤ 3) для анализа
    problematic_cases = [item for item in feedback_data if item['rating'] <= 3]
    logger.info(f"Найдено {len(problematic_cases)} проблемных случаев для анализа")
    
    # Тестируем разные pipeline
    pipelines_to_test = ['enhanced', 'original', 'strict']
    
    for pipeline_name in pipelines_to_test:
        logger.info(f"\n{'='*50}")
        logger.info(f"Тестирование pipeline: {pipeline_name}")
        logger.info(f"{'='*50}")
        
        try:
            # Запускаем тесты (ограничиваем 20 случаями для быстрого анализа)
            test_suite = await tester.run_test_suite(
                pipeline_name=pipeline_name,
                feedback_data=problematic_cases,
                max_tests=20
            )
            
            # Сохраняем результаты
            tester.save_results(pipeline_name, test_suite)
            
            # Выводим краткий отчет
            logger.info(f"Результаты для {pipeline_name}:")
            logger.info(f"  Всего тестов: {test_suite.total_tests}")
            logger.info(f"  Успешных: {test_suite.passed_tests}")
            logger.info(f"  Среднее время: {test_suite.avg_response_time:.2f}с")
            logger.info(f"  Средний retrieval score: {test_suite.avg_retrieval_score:.3f}")
            logger.info(f"  Основные ошибки: {list(test_suite.error_patterns.keys())}")
            
        except Exception as e:
            logger.error(f"Ошибка при тестировании {pipeline_name}: {e}")
    
    logger.info("\nТестирование завершено. Результаты сохранены в папке test_results/")


if __name__ == "__main__":
    asyncio.run(main())
