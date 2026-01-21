"""Тесты для настроек."""
import os
import pytest
from pathlib import Path
from exceptions import ConfigurationError


def test_settings_import():
    """Тест импорта настроек."""
    from settings import (
        CHROMA_DIR,
        EMBED_MODEL,
        TELEGRAM_BOT_TOKEN,
        _settings
    )
    assert CHROMA_DIR is not None
    assert EMBED_MODEL is not None
    assert isinstance(TELEGRAM_BOT_TOKEN, str)


def test_settings_validation_for_bot():
    """Тест валидации настроек для бота."""
    from settings import _settings
    
    if hasattr(_settings, 'validate_for_bot'):
        # Если токен не установлен, должно быть исключение
        original_token = _settings.telegram_bot_token
        try:
            _settings.telegram_bot_token = ""
            with pytest.raises(ConfigurationError):
                _settings.validate_for_bot()
        finally:
            _settings.telegram_bot_token = original_token


def test_settings_defaults():
    """Тест значений по умолчанию."""
    from settings import (
        TOP_K,
        SUMM_K,
        CHUNKS_K,
        MAX_CTX_CHARS,
        MAX_CHUNK_CHARS
    )
    assert TOP_K >= 1
    assert SUMM_K >= 1
    assert CHUNKS_K >= 1
    assert MAX_CTX_CHARS >= 1000
    assert MAX_CHUNK_CHARS >= 100



