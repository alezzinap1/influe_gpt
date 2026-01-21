"""Тесты для функций работы с источниками."""
import pytest
from utils.source_links import filter_sources_strict, format_source_reference_html


def test_filter_sources_strict_empty():
    """Тест фильтрации пустого списка."""
    result = filter_sources_strict([], max_sources=5)
    assert result == []


def test_filter_sources_strict_single():
    """Тест фильтрации одного источника."""
    chunks = [
        (0.9, "text", {
            "channel": "test_channel",
            "msg_ids": "123",
            "is_forwarded": False
        })
    ]
    result = filter_sources_strict(chunks, max_sources=5)
    assert len(result) == 1
    assert result[0][0] == "test_channel"
    assert result[0][1] == 123
    assert result[0][2] == False


def test_filter_sources_strict_forwarded():
    """Тест фильтрации пересланного сообщения."""
    chunks = [
        (0.9, "text", {
            "channel": "test_channel",
            "msg_ids": "456",
            "is_forwarded": True,
            "forwarded_from": "source_channel"
        })
    ]
    result = filter_sources_strict(chunks, max_sources=5)
    assert len(result) == 1
    assert result[0][0] == "test_channel"
    assert result[0][1] == 456
    assert result[0][2] == True
    assert result[0][3] == "source_channel"


def test_filter_sources_strict_max_sources():
    """Тест ограничения количества источников."""
    chunks = [
        (0.9, f"text{i}", {
            "channel": "test_channel",
            "msg_ids": str(i),
            "is_forwarded": False
        })
        for i in range(10)
    ]
    result = filter_sources_strict(chunks, max_sources=3)
    assert len(result) == 3


def test_filter_sources_strict_duplicates():
    """Тест удаления дубликатов."""
    chunks = [
        (0.9, "text1", {
            "channel": "test_channel",
            "msg_ids": "123",
            "is_forwarded": False
        }),
        (0.8, "text2", {
            "channel": "test_channel",
            "msg_ids": "123",
            "is_forwarded": False
        })
    ]
    result = filter_sources_strict(chunks, max_sources=5)
    assert len(result) == 1


def test_format_source_reference_html():
    """Тест форматирования HTML ссылки на источник."""
    result = format_source_reference_html("test_channel", 123, 1)
    assert "test_channel" in result
    assert "123" in result
    assert "href" in result.lower()


def test_format_source_reference_html_forwarded():
    """Тест форматирования пересланного сообщения."""
    result = format_source_reference_html(
        "test_channel", 456, 2,
        is_forwarded=True,
        forwarded_from="source_channel"
    )
    assert "test_channel" in result
    assert "456" in result
    assert "Пересылка" in result or "пересылка" in result.lower()



