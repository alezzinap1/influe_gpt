"""Иерархия исключений для RAG системы."""


class RAGError(Exception):
    """Базовое исключение для RAG системы."""
    pass


class ConfigurationError(RAGError):
    """Ошибки конфигурации."""
    pass


class ChromaDBError(RAGError):
    """Ошибки ChromaDB."""
    pass


class LLMError(RAGError):
    """Ошибки LLM."""
    pass


class TelegramError(RAGError):
    """Ошибки Telegram API."""
    pass


class ValidationError(RAGError):
    """Ошибки валидации данных."""
    pass



