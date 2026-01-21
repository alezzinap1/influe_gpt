# keyword_extractor.py
"""
Модуль для извлечения ключевых слов из текста.
Использует легковесный подход для минимальных затрат скорости.
"""
import re
from typing import List
from collections import Counter

# Расширенный стоп-лист для русского и английского
STOPWORDS = {
    # Русский базовый + extended
    'и', 'в', 'во', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'о', 'об', 'а', 'но', 'или', 'да', 'нет', 'не', 'то', 
    'что', 'как', 'это', 'все', 'он', 'она', 'они', 'его', 'ее', 'их', 'мы', 'вы', 'ты', 'я', 'бы', 'же', 'ли', 
    'было', 'будет', 'есть', 'был', 'была', 'были', 'только', 'ещё', 'уже', 'если', 'когда', 'за', 'под', 'над', 
    'при', 'со', 'без', 'через', 'у', 'про', 'ни', 'из-за', 'после', 'перед', 'между', 'чтобы', 'также', 'вот', 
    'там', 'здесь', 'тут', 'сейчас', 'потом', 'потому', 'поэтому', 'однако', 'хотя', 'ведь', 'даже', 'очень', 
    'всё', 'весь', 'вся', 'какой', 'который', 'этот', 'тот', 'такой', 'сам', 'самый', 'другой', 'каждый', 'любой', 
    'никакой', 'несколько', 'много', 'больше', 'меньше', 'еще', 'ещё', 'вот', 'ну', 'давай', 'ага', 'ща', 'короче',
    
    # Английский extended
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'while', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
}


def extract_keywords_simple(text: str, top_k: int = 5) -> List[str]:
    """
    Быстрое извлечение ключевых слов через TF-IDF подход.
    Время: ~0.001s на текст до 1000 символов.
    
    Args:
        text: Текст для обработки
        top_k: Количество ключевых слов
        
    Returns:
        Список ключевых слов (lowercase, без стоп-слов)
    """
    if not text or len(text.strip()) < 10:
        return []
    
    # Токенизация: слова длиной >= 3 символа (русские и английские)
    words = re.findall(r'\b[а-яёa-z]{3,}\b', text.lower())
    
    # Фильтрация стоп-слов
    words = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    
    if not words:
        return []
    
    # Подсчет частоты
    word_freq = Counter(words)
    
    # Простой IDF-подобный вес: частота / длина слова (длинные слова важнее)
    scored_words = [
        (word, freq * (1 + len(word) * 0.1))
        for word, freq in word_freq.items()
    ]
    
    # Сортировка и выбор top_k
    scored_words.sort(key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in scored_words[:top_k]]
    
    return keywords


def extract_keywords_keybert(text: str, top_k: int = 5) -> List[str]:
    """
    Извлечение ключевых слов через KeyBERT (более точное, но медленнее).
    Использовать для длинных текстов (>500 символов).
    Время: ~0.01-0.02s на текст.
    
    Args:
        text: Текст для обработки
        top_k: Количество ключевых слов
        
    Returns:
        Список ключевых слов
    """
    try:
        from keybert import KeyBERT
        # Ленивая инициализация модели (легковесная)
        if not hasattr(extract_keywords_keybert, '_model'):
            extract_keywords_keybert._model = KeyBERT('paraphrase-MiniLM-L6-v2')
        
        keywords = extract_keywords_keybert._model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            top_n=top_k,
            use_mmr=True,
            diversity=0.5
        )
        return [kw[0].lower() for kw in keywords if kw[1] > 0.2]  # фильтр по confidence
    except ImportError:
        # Fallback на простой метод
        return extract_keywords_simple(text, top_k)


def extract_keywords(text: str, top_k: int = 5, use_keybert: bool = False) -> List[str]:
    """
    Универсальная функция извлечения ключевых слов.
    
    Args:
        text: Текст для обработки
        top_k: Количество ключевых слов
        use_keybert: Использовать KeyBERT (для длинных текстов)
        
    Returns:
        Список ключевых слов
    """
    if use_keybert and len(text) > 500:
        return extract_keywords_keybert(text, top_k)
    else:
        return extract_keywords_simple(text, top_k)


def keywords_to_string(keywords: List[str]) -> str:
    """Конвертирует список keywords в строку для хранения в metadata."""
    return ",".join(keywords)


def string_to_keywords(keywords_str: str) -> List[str]:
    """Конвертирует строку keywords обратно в список."""
    if not keywords_str:
        return []
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]







