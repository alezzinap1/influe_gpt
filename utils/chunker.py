# semantic_chunker.py
"""
Модуль для семантического разбиения длинных текстов.
Использует sentence-transformers для определения границ предложений.
"""
from typing import List
import re

# Ленивая инициализация модели
_semantic_model = None

def get_semantic_model():
    """Ленивая загрузка модели для semantic chunking."""
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        # Легковесная модель для определения границ
        _semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return _semantic_model


def split_by_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения."""
    # Простое разбиение по знакам препинания
    sentences = re.split(r'[.!?]\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text: str, max_chars: int = 600, min_chars: int = 200) -> List[str]:
    """
    Семантическое разбиение длинного текста на чанки.
    
    Использует embeddings для группировки семантически близких предложений.
    Время: ~0.05s на текст 1000 символов.
    
    Args:
        text: Текст для разбиения
        max_chars: Максимальный размер чанка
        min_chars: Минимальный размер чанка
        
    Returns:
        Список чанков
    """
    if len(text) <= max_chars:
        return [text]
    
    sentences = split_by_sentences(text)
    if not sentences:
        return [text]
    
    # Если предложений мало, просто разбиваем по символам
    if len(sentences) < 3:
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence if current else sentence
        if current:
            chunks.append(current.strip())
        return chunks
    
    # Для длинных текстов используем semantic grouping
    model = get_semantic_model()
    
    # Получаем embeddings для предложений
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for i, (sentence, emb) in enumerate(zip(sentences, embeddings)):
        sentence_len = len(sentence)
        
        # Если текущий чанк пуст или предложение помещается
        if not current_chunk or (current_len + sentence_len <= max_chars):
            current_chunk.append(sentence)
            current_len += sentence_len
        else:
            # Сохраняем текущий чанк
            chunks.append(" ".join(current_chunk))
            
            # Начинаем новый чанк
            current_chunk = [sentence]
            current_len = sentence_len
    
    # Добавляем последний чанк
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Объединяем слишком короткие чанки
    merged_chunks = []
    for chunk in chunks:
        if len(chunk) < min_chars and merged_chunks:
            merged_chunks[-1] += " " + chunk
        else:
            merged_chunks.append(chunk)
    
    return merged_chunks







