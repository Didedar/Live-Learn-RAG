#!/usr/bin/env python3
"""
Тестирование улучшений RAG системы.
Сравнение оригинальной и улучшенной версий на проблемных случаях.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

from loguru import logger

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from app.database import SessionLocal
from app.services.enhanced_rag_pipeline import EnhancedRAGPipeline
from app.services.improved_rag_pipeline import ImprovedRAGPipeline


class RAGComparison:
    """Класс для сравнения RAG систем."""
    
    def __init__(self):
        self.original_pipeline = EnhancedRAGPipeline()
        self.improved_pipeline = ImprovedRAGPipeline()
    
    async def test_single_question(self, db, question: str) -> Dict[str, Any]:
        """Тестирование одного вопроса на обеих системах."""
        
        logger.info(f"Тестирование: {question[:80]}...")
        
        # Тестируем оригинальную систему
        start_time = time.time()
        original_result = await self.original_pipeline.ask(
            question=question,
            db=db,
            top_k=4
        )
        original_time = time.time() - start_time
        
        # Тестируем улучшенную систему
        start_time = time.time()
        improved_result = await self.improved_pipeline.ask(
            question=question,
            db=db,
            top_k=4
        )
        improved_time = time.time() - start_time
        
        return {
            "question": question,
            "original": {
                "answer": original_result["answer"],
                "can_answer": original_result.get("can_answer", True),
                "contexts_count": len(original_result.get("contexts", [])),
                "max_score": original_result.get("max_score", 0.0),
                "response_time": original_time,
                "method": original_result.get("retrieval_method", "original"),
                "answer_length": len(original_result["answer"])
            },
            "improved": {
                "answer": improved_result["answer"],
                "can_answer": improved_result.get("can_answer", True),
                "confidence": improved_result.get("confidence", 0.0),
                "contexts_count": len(improved_result.get("contexts", [])),
                "max_score": improved_result.get("max_score", 0.0),
                "response_time": improved_time,
                "method": improved_result.get("retrieval_method", "improved"),
                "answer_length": len(improved_result["answer"])
            }
        }
    
    def analyze_comparison(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ сравнения двух ответов."""
        
        original = result["original"]
        improved = result["improved"]
        
        analysis = {
            "improvements": [],
            "concerns": [],
            "metrics": {}
        }
        
        # Анализ языковой консистентности
        if self._has_english_words(original["answer"]) and not self._has_english_words(improved["answer"]):
            analysis["improvements"].append("Исправлена языковая проблема (убран английский)")
        
        # Анализ уверенности
        if improved.get("confidence", 0) > 0.6:
            analysis["improvements"].append(f"Высокая уверенность: {improved['confidence']:.2f}")
        elif improved.get("confidence", 0) < 0.3:
            analysis["concerns"].append(f"Низкая уверенность: {improved['confidence']:.2f}")
        
        # Анализ отказов от ответа
        original_refuses = self._is_refusal(original["answer"])
        improved_refuses = self._is_refusal(improved["answer"])
        
        if original_refuses and not improved_refuses:
            analysis["improvements"].append("Улучшенная система дала ответ там, где оригинальная отказалась")
        elif not original_refuses and improved_refuses:
            analysis["improvements"].append("Улучшенная система корректно отказалась от неточного ответа")
        
        # Анализ качества ответа
        if improved["answer_length"] > original["answer_length"] * 1.5:
            analysis["improvements"].append("Более подробный ответ")
        elif improved["answer_length"] < original["answer_length"] * 0.5:
            analysis["concerns"].append("Слишком краткий ответ")
        
        # Анализ производительности
        if improved["response_time"] < original["response_time"]:
            analysis["improvements"].append(f"Быстрее на {original['response_time'] - improved['response_time']:.1f}с")
        elif improved["response_time"] > original["response_time"] * 1.5:
            analysis["concerns"].append("Значительно медленнее")
        
        # Метрики
        analysis["metrics"] = {
            "confidence_score": improved.get("confidence", 0.0),
            "speed_improvement": original["response_time"] - improved["response_time"],
            "answer_length_ratio": improved["answer_length"] / max(original["answer_length"], 1),
            "retrieval_score_improvement": improved["max_score"] - original["max_score"]
        }
        
        return analysis
    
    def _has_english_words(self, text: str) -> bool:
        """Проверка на английские слова."""
        english_words = ['unfortunately', 'i don\'t know', 'the', 'and', 'based on', 'context']
        return any(word in text.lower() for word in english_words)
    
    def _is_refusal(self, answer: str) -> bool:
        """Проверка, является ли ответ отказом."""
        refusal_phrases = [
            'у меня нет информации',
            'не могу ответить',
            'недостаточно информации',
            'unfortunately',
            'i don\'t know'
        ]
        return any(phrase in answer.lower() for phrase in refusal_phrases)
    
    async def run_comparison_tests(self, questions: List[str], max_tests: int = 10) -> List[Dict[str, Any]]:
        """Запуск сравнительных тестов."""
        
        results = []
        db = SessionLocal()
        
        try:
            for i, question in enumerate(questions[:max_tests]):
                logger.info(f"Тест {i+1}/{min(len(questions), max_tests)}")
                
                # Тестируем вопрос
                result = await self.test_single_question(db, question)
                
                # Анализируем результат
                analysis = self.analyze_comparison(result)
                result["analysis"] = analysis
                
                results.append(result)
                
                # Пауза между тестами
                await asyncio.sleep(0.5)
                
        finally:
            db.close()
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Генерация итогового отчета."""
        
        total_tests = len(results)
        if total_tests == 0:
            return {"error": "No test results"}
        
        # Счетчики улучшений
        language_fixes = 0
        confidence_improvements = 0
        better_answers = 0
        speed_improvements = 0
        
        total_confidence = 0
        total_speed_diff = 0
        
        for result in results:
            analysis = result["analysis"]
            
            # Подсчет улучшений
            if any("языковая" in imp for imp in analysis["improvements"]):
                language_fixes += 1
            
            if any("уверенность" in imp for imp in analysis["improvements"]):
                confidence_improvements += 1
            
            if any("подробный" in imp or "ответ" in imp for imp in analysis["improvements"]):
                better_answers += 1
            
            if analysis["metrics"]["speed_improvement"] > 0:
                speed_improvements += 1
            
            total_confidence += analysis["metrics"]["confidence_score"]
            total_speed_diff += analysis["metrics"]["speed_improvement"]
        
        return {
            "summary": {
                "total_tests": total_tests,
                "language_fixes": language_fixes,
                "confidence_improvements": confidence_improvements,
                "better_answers": better_answers,
                "speed_improvements": speed_improvements,
                "avg_confidence": total_confidence / total_tests,
                "avg_speed_improvement": total_speed_diff / total_tests
            },
            "success_rates": {
                "language_consistency": (language_fixes / total_tests) * 100,
                "confidence_scoring": (confidence_improvements / total_tests) * 100,
                "answer_quality": (better_answers / total_tests) * 100,
                "performance": (speed_improvements / total_tests) * 100
            },
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Генерация рекомендаций на основе результатов."""
        
        recommendations = []
        
        # Анализ проблем
        low_confidence_cases = [r for r in results if r["analysis"]["metrics"]["confidence_score"] < 0.4]
        slow_cases = [r for r in results if r["analysis"]["metrics"]["speed_improvement"] < -2.0]
        short_answers = [r for r in results if r["analysis"]["metrics"]["answer_length_ratio"] < 0.5]
        
        if len(low_confidence_cases) > len(results) * 0.3:
            recommendations.append("Необходимо улучшить confidence scoring - слишком много случаев низкой уверенности")
        
        if len(slow_cases) > len(results) * 0.2:
            recommendations.append("Оптимизировать производительность - улучшенная система работает медленнее")
        
        if len(short_answers) > len(results) * 0.3:
            recommendations.append("Увеличить детализацию ответов - многие ответы стали слишком краткими")
        
        # Положительные аспекты
        language_fixes = sum(1 for r in results if any("языковая" in imp for imp in r["analysis"]["improvements"]))
        if language_fixes > 0:
            recommendations.append(f"✅ Языковые проблемы исправлены в {language_fixes} случаях")
        
        return recommendations


async def main():
    """Основная функция тестирования."""
    
    logger.info("Запуск сравнительного тестирования RAG систем")
    
    # Проблемные вопросы из feedback данных
    test_questions = [
        "Может ли ИП в РК иметь счёт в иностранном банке без уведомления Нацбанка?",
        "Нужно ли в Казахстане регистрировать брак в ЗАГСе?",
        "Какой штраф за езду без техосмотра в Казахстане?",
        "Сколько статей в Конституции РК?",
        "Что такое закон?",
        "В 2024 году казахстанский банк «НурБанк» выдал кредит в тенге резиденту Кыргызстана под залог недвижимости в Астане. Какие нормы коллизионного права РК определяют применимое право?",
        "В 2021 году казахстанская фермерская компания «АгроБереке» арендовала 5000 га земли в Костанайской области. Какие законы определяют приоритет при коллизии между Земельным кодексом и градостроительным законодательством?",
        "В 2020 году акционерное общество «Алмалы-Транзит» заключило с АО «НК «ҚТЖ» концессионное соглашение. Какие санкции могут быть применены к участникам при нарушении?",
        "В 2022 году казахстанский предприниматель Айбек Сулейменов, наследуя после смерти отца дом в Алматы по завещанию 2018 года, обнаружил проблемы. Какие нормы коллизионного права РК определяют применимое право к дееспособности?"
    ]
    
    # Создаем тестер
    comparator = RAGComparison()
    
    # Запускаем тесты
    logger.info(f"Тестирование {len(test_questions)} проблемных вопросов...")
    results = await comparator.run_comparison_tests(test_questions, max_tests=9)
    
    # Генерируем отчет
    summary = comparator.generate_summary_report(results)
    
    # Сохраняем результаты
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "detailed_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "summary_report.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Выводим краткий отчет
    logger.info("\n" + "="*60)
    logger.info("ИТОГОВЫЙ ОТЧЕТ СРАВНЕНИЯ RAG СИСТЕМ")
    logger.info("="*60)
    
    summary_data = summary["summary"]
    success_rates = summary["success_rates"]
    
    logger.info(f"Всего тестов: {summary_data['total_tests']}")
    logger.info(f"Исправлены языковые проблемы: {summary_data['language_fixes']} ({success_rates['language_consistency']:.1f}%)")
    logger.info(f"Улучшения в уверенности: {summary_data['confidence_improvements']} ({success_rates['confidence_scoring']:.1f}%)")
    logger.info(f"Улучшения качества ответов: {summary_data['better_answers']} ({success_rates['answer_quality']:.1f}%)")
    logger.info(f"Улучшения производительности: {summary_data['speed_improvements']} ({success_rates['performance']:.1f}%)")
    logger.info(f"Средняя уверенность: {summary_data['avg_confidence']:.2f}")
    logger.info(f"Среднее улучшение скорости: {summary_data['avg_speed_improvement']:.1f}с")
    
    logger.info("\nРекомендации:")
    for rec in summary["recommendations"]:
        logger.info(f"  • {rec}")
    
    logger.info(f"\nДетальные результаты сохранены в {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())

