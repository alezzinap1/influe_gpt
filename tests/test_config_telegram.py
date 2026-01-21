"""Тесты для функций работы с Telegram конфигурацией."""
import pytest
from config_telegram import normalize_channel_name, raw_parquet_path


def test_normalize_channel_name_simple():
    """Тест нормализации простого имени канала."""
    assert normalize_channel_name("channelname") == "channelname"


def test_normalize_channel_name_with_at():
    """Тест нормализации имени с @."""
    assert normalize_channel_name("@channelname") == "channelname"


def test_normalize_channel_name_with_url():
    """Тест нормализации URL канала."""
    assert normalize_channel_name("https://t.me/channelname") == "channelname"
    assert normalize_channel_name("http://t.me/channelname") == "channelname"


def test_normalize_channel_name_mixed():
    """Тест нормализации смешанного формата."""
    assert normalize_channel_name("@https://t.me/channelname") == "channelname"


def test_normalize_channel_name_empty():
    """Тест нормализации пустой строки."""
    assert normalize_channel_name("") == ""
    assert normalize_channel_name("   ") == ""


def test_normalize_channel_name_with_spaces():
    """Тест нормализации с пробелами."""
    assert normalize_channel_name("  channelname  ") == "channelname"


def test_raw_parquet_path():
    """Тест генерации пути к parquet файлу."""
    path = raw_parquet_path("test_channel")
    assert path.name == "test_channel.parquet"
    assert path.suffix == ".parquet"


def test_raw_parquet_path_with_slash():
    """Тест генерации пути с заменой слешей."""
    path = raw_parquet_path("test/channel")
    assert "test_channel" in path.name or "_" in path.name



