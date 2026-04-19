"""Лёгкие тесты утилит RAG pipeline (без загрузки Chroma/моделей)."""
import hashlib
from typing import List


def _documents_fingerprint(docs: List[str]) -> str:
    """Копия логики из rag.pipeline._documents_fingerprint для регрессии без тяжёлых импортов."""
    h = hashlib.md5()
    sep = b"|"
    for i, doc in enumerate(docs):
        h.update(str(i).encode("ascii", errors="ignore"))
        h.update(sep)
        h.update(doc.encode("utf-8", errors="replace"))
    return h.hexdigest()


def test_documents_fingerprint_stable():
    assert _documents_fingerprint(["hello", "world"]) == _documents_fingerprint(["hello", "world"])


def test_documents_fingerprint_order_matters():
    assert _documents_fingerprint(["a", "b"]) != _documents_fingerprint(["b", "a"])


def test_documents_fingerprint_matches_pipeline_module():
    """Синхронизация с rag.pipeline (если модуль импортируется в окружении CI)."""
    try:
        from rag.pipeline import _documents_fingerprint as fp_mod
    except ImportError:
        import pytest

        pytest.skip("rag.pipeline недоступен (зависимости)")
    docs = ["один", "два", "три"]
    assert fp_mod(docs) == _documents_fingerprint(docs)
