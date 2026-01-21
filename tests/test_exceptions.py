"""Тесты для иерархии исключений."""
import pytest
from exceptions import (
    RAGError,
    ConfigurationError,
    ChromaDBError,
    LLMError,
    TelegramError,
    ValidationError
)


def test_rag_error_inheritance():
    """Тест наследования базового исключения."""
    assert issubclass(ConfigurationError, RAGError)
    assert issubclass(ChromaDBError, RAGError)
    assert issubclass(LLMError, RAGError)
    assert issubclass(TelegramError, RAGError)
    assert issubclass(ValidationError, RAGError)


def test_configuration_error():
    """Тест исключения конфигурации."""
    error = ConfigurationError("Missing API key")
    assert str(error) == "Missing API key"
    assert isinstance(error, RAGError)


def test_chromadb_error():
    """Тест исключения ChromaDB."""
    error = ChromaDBError("Connection failed")
    assert str(error) == "Connection failed"
    assert isinstance(error, RAGError)


def test_llm_error():
    """Тест исключения LLM."""
    error = LLMError("API timeout")
    assert str(error) == "API timeout"
    assert isinstance(error, RAGError)


def test_validation_error():
    """Тест исключения валидации."""
    error = ValidationError("Invalid channel name")
    assert str(error) == "Invalid channel name"
    assert isinstance(error, RAGError)



