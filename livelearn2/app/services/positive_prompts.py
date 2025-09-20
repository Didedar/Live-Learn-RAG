"""Positive prompts for RAG system - focusing on helpful responses."""

def build_positive_rag_prompt(query: str, contexts: list, language: str = "ru") -> str:
    """
    Build a positive, helpful RAG prompt that encourages comprehensive answers.
    
    Args:
        query: User question
        contexts: List of context dictionaries
        language: Response language ("ru" for Russian, "kz" for Kazakh)
    
    Returns:
        Formatted prompt string
    """
    
    # Language-specific instructions
    if language == "kz":
        lang_instruction = "Жауапты ТҰРАҚТЫ қазақ тілінде беріңіз"
        helpful_instruction = "Барлық қолжетімді ақпаратты пайдалана отырып, толық және пайдалы жауап беріңіз"
    else:
        lang_instruction = "Отвечай ТОЛЬКО на русском языке"
        helpful_instruction = "Предоставь максимально полную и полезную информацию, используя все доступные данные"
    
    # Format contexts
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        content = ctx.get('content', ctx.get('text', ''))
        score = ctx.get('score', 0.0)
        doc_id = ctx.get('doc_id', 'unknown')
        chunk_id = ctx.get('chunk_id', 'unknown')
        
        context_parts.append(
            f"[{i}] (документ {doc_id}, фрагмент {chunk_id}, релевантность {score:.3f})\n{content}"
        )
    
    context_text = "\n\n".join(context_parts)
    
    # Build positive system prompt
    system_prompt = f"""Ты - эксперт-консультант по государственным услугам Республики Казахстан. Твоя задача - максимально помочь пользователю.

ПРИНЦИПЫ РАБОТЫ:
• {helpful_instruction}
• {lang_instruction}
• Используй весь предоставленный контекст для формирования исчерпывающего ответа
• Структурируй информацию логично и понятно
• Указывай конкретные источники в формате [номер]
• Предоставляй практические рекомендации и следующие шаги
• Если есть альтернативные варианты или подходы, представь их все
• Делай разумные выводы на основе имеющейся информации
• Будь уверенным и конкретным в формулировках

СТИЛЬ ОТВЕТА:
• Начинай с прямого ответа на вопрос
• Предоставляй детали и контекст
• Завершай практическими рекомендациями
• Используй профессиональный, но доступный язык"""

    user_prompt = f"""КОНТЕКСТ:
{context_text}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}

Предоставь максимально полный и полезный ответ, используя всю доступную информацию из контекста."""
    
    return f"System: {system_prompt}\n\nUser: {user_prompt}"


def build_positive_ollama_prompt(query: str, contexts: list, language: str = "ru") -> str:
    """
    Build positive prompt specifically optimized for Ollama/Llama models.
    
    Args:
        query: User question
        contexts: List of context dictionaries
        language: Response language
    
    Returns:
        Formatted prompt for Ollama
    """
    
    # Language instruction
    if language == "kz":
        lang_instruction = "Жауапты ТҰРАҚТЫ қазақ тілінде беріңіз"
    else:
        lang_instruction = "Отвечай ТОЛЬКО на русском языке"
    
    # Format contexts with metadata
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        content = ctx.get('content', ctx.get('text', ''))
        score = ctx.get('score', 0.0)
        
        # Add metadata if available
        metadata_info = ""
        if 'doc_id' in ctx:
            metadata_info = f" (документ {ctx['doc_id']}"
            if 'chunk_id' in ctx:
                metadata_info += f", фрагмент {ctx['chunk_id']}"
            metadata_info += ")"
        
        context_parts.append(
            f"[{i}] релевантность {score:.3f}{metadata_info}\n{content}"
        )
    
    context_text = "\n\n".join(context_parts)
    
    # Optimized prompt for Llama
    system_prompt = f"""Ты - профессиональный консультант по государственным услугам Казахстана.

ЗАДАЧА: Предоставить максимально полезный и исчерпывающий ответ на вопрос пользователя.

ПРАВИЛА:
• {lang_instruction}
• Используй ВСЮ предоставленную информацию из контекста
• Структурируй ответ логично: основная информация → детали → практические советы
• Ссылайся на источники в формате [номер]
• Будь конкретным и уверенным
• Предоставляй практические рекомендации
• Если есть несколько вариантов решения, представь их все

ФОРМАТ ОТВЕТА:
1. Прямой ответ на вопрос
2. Подробная информация из контекста
3. Практические рекомендации и следующие шаги"""

    user_prompt = f"""ИНФОРМАЦИЯ ДЛЯ ОТВЕТА:
{context_text}

ВОПРОС: {query}

Ответ (полный и практичный):"""
    
    return f"System: {system_prompt}\n\nUser: {user_prompt}"


def get_positive_fallback_messages(language: str = "ru") -> dict:
    """
    Get positive fallback messages for different scenarios.
    
    Args:
        language: Response language
    
    Returns:
        Dictionary with fallback messages
    """
    
    if language == "kz":
        return {
            "no_context": "Қолжетімді ақпарат негізінде жалпы ұсыныстар бере аламын. Толық консультация алу үшін ХҚК немесе egov.kz порталына жүгінуіңізді ұсынамын.",
            "low_confidence": "Барлық қолжетімді мәліметтерді пайдалана отырып, сізге көмектесуге тырысамын. Қосымша ақпарат алу үшін ХҚК маманымен кеңесуіңізді ұсынамын.",
            "technical_error": "Техникалық қиындыққа қарамастан, сізге максималды пайдалы ақпарат беруге тырысамын. egov.kz порталынан немесе ХҚК-дан қосымша консультация алуыңызды ұсынамын.",
            "processing_error": "Сұрағыңызға көмектесуге тырысамын. Нақты мәліметтер алу үшін ХҚК немесе egov.kz порталына жүгіну ұсынылады."
        }
    else:
        return {
            "no_context": "На основе имеющейся информации предоставлю общие рекомендации. Для получения подробной консультации рекомендую обратиться в ЦОН или на портал egov.kz.",
            "low_confidence": "Используя все доступные данные, постараюсь максимально помочь с вашим вопросом. Для получения дополнительной информации рекомендую проконсультироваться со специалистом ЦОНа.",
            "technical_error": "Несмотря на технические сложности, предоставлю максимально полезную информацию. Рекомендую также получить консультацию на портале egov.kz или в ЦОНе.",
            "processing_error": "Постараюсь помочь с вашим вопросом. Для получения точных данных рекомендую обратиться в ЦОН или на egov.kz."
        }


def enhance_answer_positivity(answer: str, language: str = "ru") -> str:
    """
    Enhance answer to be more positive and helpful.
    
    Args:
        answer: Original answer
        language: Response language
    
    Returns:
        Enhanced positive answer
    """
    
    if not answer or not answer.strip():
        fallback_messages = get_positive_fallback_messages(language)
        return fallback_messages["processing_error"]
    
    # Remove negative patterns
    negative_patterns = [
        "не знаю", "не могу", "недостаточно информации", "нет информации",
        "извините", "к сожалению", "не удалось",
        "don't know", "cannot", "insufficient", "unfortunately", "sorry"
    ]
    
    answer_lower = answer.lower()
    for pattern in negative_patterns:
        if pattern in answer_lower and len(answer) < 100:  # Only for short negative answers
            fallback_messages = get_positive_fallback_messages(language)
            return fallback_messages["low_confidence"]
    
    # Enhance positive aspects
    if language == "ru":
        positive_starters = [
            "На основе предоставленной информации",
            "Согласно имеющимся данным",
            "Рассматривая доступную информацию"
        ]
        
        # Add positive starter if answer doesn't have one
        if not any(starter.lower() in answer_lower[:50] for starter in positive_starters):
            if not answer.startswith(("Да", "Нет", "Согласно", "На основе", "В соответствии")):
                answer = f"На основе имеющейся информации: {answer}"
    
    return answer


def build_confidence_enhancing_prompt(query: str, contexts: list, language: str = "ru") -> str:
    """
    Build a prompt that enhances confidence in responses while maintaining accuracy.
    
    Args:
        query: User question
        contexts: List of contexts
        language: Response language
    
    Returns:
        Confidence-enhancing prompt
    """
    
    base_prompt = build_positive_rag_prompt(query, contexts, language)
    
    # Add confidence-building instructions
    confidence_instruction = """
    
ДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ ДЛЯ УВЕРЕННЫХ ОТВЕТОВ:
• Формулируй утверждения уверенно, используя активный залог
• Избегай слов сомнения: "возможно", "может быть", "вероятно"
• Используй конкретные факты и цифры из контекста
• Структурируй ответ от общего к частному
• Завершай практическими действиями, которые может предпринять пользователь
• Если информация частичная, четко укажи что известно точно, а что требует уточнения"""
    
    return base_prompt + confidence_instruction

