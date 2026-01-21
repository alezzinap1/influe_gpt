# test_improvements.py
"""Тест новых функций chunking и keyword extraction."""

from utils.keywords import extract_keywords, keywords_to_string, string_to_keywords
from utils.chunker import semantic_chunk
import time

# Тест 1: Keyword extraction
print("=" * 60)
print("Тест 1: Keyword extraction")
print("=" * 60)
text = "Python это язык программирования. Python используется для машинного обучения и веб-разработки. Python очень популярен."
start = time.time()
keywords = extract_keywords(text, top_k=5)
elapsed = time.time() - start
print(f"Текст: {text[:80]}...")
print(f"Keywords: {keywords}")
print(f"Время: {elapsed:.4f}s")
print(f"Keywords string: {keywords_to_string(keywords)}")
print(f"Keywords from string: {string_to_keywords(keywords_to_string(keywords))}")

# Тест 2: Semantic chunking
print("\n" + "=" * 60)
print("Тест 2: Semantic chunking")
print("=" * 60)
long_text = " ".join([f"Это предложение номер {i}. Оно содержит важную информацию о проекте." for i in range(30)])
print(f"Исходный текст: {len(long_text)} символов")
start = time.time()
chunks = semantic_chunk(long_text, max_chars=200, min_chars=100)
elapsed = time.time() - start
print(f"Чанков создано: {len(chunks)}")
print(f"Время: {elapsed:.4f}s")
for i, chunk in enumerate(chunks[:3]):
    print(f"  Чанк {i+1}: {len(chunk)} символов - {chunk[:60]}...")

# Тест 3: Короткий текст (не должен использовать semantic chunking)
print("\n" + "=" * 60)
print("Тест 3: Короткий текст (без semantic chunking)")
print("=" * 60)
short_text = "Это короткое сообщение."
print(f"Текст: {short_text} ({len(short_text)} символов)")
chunks = semantic_chunk(short_text, max_chars=200, min_chars=100)
print(f"Чанков: {len(chunks)} (должен быть 1)")

print("\n" + "=" * 60)
print("Все тесты завершены!")
print("=" * 60)

