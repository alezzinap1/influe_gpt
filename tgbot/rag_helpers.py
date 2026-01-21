"""Вспомогательные функции для работы с RAG в боте."""
from typing import List, Optional
from exceptions import LLMError, ChromaDBError, ValidationError
from utils.cache import get_cache


def _handle_rag_error(e: Exception, context: str = "RAG") -> Exception:
    """
    Обрабатывает ошибки RAG pipeline и преобразует их в типизированные исключения.
    
    Args:
        e: Исходное исключение
        context: Контекст операции (для сообщения об ошибке)
    
    Returns:
        Типизированное исключение (LLMError, ChromaDBError или ValidationError)
    """
    if isinstance(e, (ValidationError, LLMError, ChromaDBError)):
        return e
    
    error_msg = f"Ошибка {context}: {e}"
    
    # Специальная обработка для torch ошибок
    if "torch" in str(e).lower() or "tensor" in str(e).lower():
        error_msg = (
            "Ошибка: проблема с библиотекой torch. "
            "Попробуйте переустановить: pip install --upgrade torch"
        )
        return LLMError(error_msg)
    
    # Обработка ChromaDB ошибок
    if "chroma" in str(e).lower() or "vectorstore" in str(e).lower():
        return ChromaDBError(f"Ошибка ChromaDB: {e}")
    
    # Обработка импортных ошибок
    if isinstance(e, (ImportError, AttributeError, OSError)):
        return LLMError(error_msg)
    
    # По умолчанию - LLMError
    return LLMError(error_msg)


def execute_rag_query(
    question: str,
    channel: str | None = None,
    channels: List[str] | None = None,
    backend: str = "deepseek"
) -> str:
    """
    Выполняет RAG запрос с обработкой ошибок и кэшированием.
    
    Args:
        question: Вопрос пользователя
        channel: Имя канала (для одиночного запроса)
        channels: Список каналов (для мультиканального запроса)
        backend: LLM backend для использования
    
    Returns:
        Ответ от RAG системы
    
    Raises:
        ValidationError: Если валидация не прошла
        LLMError: Если ошибка в LLM или RAG pipeline
        ChromaDBError: Если ошибка в ChromaDB
    """
    from rag.pipeline import RAGPipeline
    from config_telegram import normalize_channel_name
    import logging
    
    logger = logging.getLogger(__name__)
    cache = get_cache()
    
    # Пытаемся получить ответ из кэша
    cached_answer = cache.get(question, channel, channels)
    if cached_answer:
        logger.info(f"[RAG] Ответ получен из кэша для вопроса: {question[:50]}...")
        return cached_answer
    
    try:
        rag = RAGPipeline(backend=backend)
        answer = None
        
        # Мультиканальный запрос
        if channels and len(channels) > 0:
            normalized_channels = [normalize_channel_name(ch) for ch in channels if normalize_channel_name(ch)]
            
            if len(normalized_channels) < 2:
                raise ValidationError("Нужно минимум 2 канала для мультиканального поиска")
            
            answer = rag.query(question, channels=normalized_channels)
        
        # Одиночный запрос
        elif channel:
            answer = rag.query(question, channel=channel)
        
        # Запрос по всем каналам
        else:
            answer = rag.query(question)
        
        # Сохраняем ответ в кэш
        if answer:
            cache.set(question, answer, channel, channels)
        
        return answer
        
    except ValidationError:
        raise
    except Exception as e:
        error = _handle_rag_error(e, context="RAG pipeline")
        raise error

