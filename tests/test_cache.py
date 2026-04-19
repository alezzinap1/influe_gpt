"""Тесты кэша ответов RAG."""
import pytest

from utils.cache import Cache, MemoryCacheBackend


@pytest.fixture
def rag_cache():
    return Cache(MemoryCacheBackend(default_ttl=3600), enabled=True, default_ttl=3600)


def test_cache_key_separates_backend_single_channel(rag_cache):
    rag_cache.set("What is X?", "from-deepseek", channel="mychannel", backend="deepseek")
    assert rag_cache.get("What is X?", channel="mychannel", backend="deepseek") == "from-deepseek"
    assert rag_cache.get("What is X?", channel="mychannel", backend="openai") is None


def test_cache_key_separates_backend_multi_channel(rag_cache):
    chans = ["a", "b"]
    rag_cache.set("q", "ans-ds", channels=chans, backend="deepseek")
    assert rag_cache.get("q", channels=chans, backend="deepseek") == "ans-ds"
    assert rag_cache.get("q", channels=chans, backend="gemini") is None


def test_cache_default_backend_is_deepseek(rag_cache):
    rag_cache.set("hello", "v1", channel="c")
    assert rag_cache.get("hello", channel="c") == "v1"
    assert rag_cache.get("hello", channel="c", backend="deepseek") == "v1"


def test_cache_delete_respects_backend(rag_cache):
    rag_cache.set("x", "a1", channel="c", backend="deepseek")
    rag_cache.delete("x", channel="c", backend="deepseek")
    assert rag_cache.get("x", channel="c", backend="deepseek") is None
