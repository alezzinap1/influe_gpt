# telegram_ingest.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any
import logging

import pandas as pd

from vectorstore.chromadb_store import ChromaStore
from utils.keywords import extract_keywords, keywords_to_string
from utils.chunker import semantic_chunk

logger = logging.getLogger(__name__)


@dataclass
class TgChunk:
    text: str
    metadata: Dict[str, Any]


def _year_quarter(dt: datetime):
    q = (dt.month - 1) // 3 + 1
    return dt.year, q


def _create_chunk_metadata(row, text: str, is_forwarded: bool = False, forwarded_from: str = None) -> Dict[str, Any]:
    """
    Создает метаданные для чанка.
    ChromaDB не принимает None значения, поэтому они либо удаляются, либо заменяются на пустые строки.
    
    Args:
        row: Строка DataFrame с данными сообщения
        text: Текст чанка
        is_forwarded: Является ли сообщение пересылкой
        forwarded_from: Откуда переслано (если пересылка)
    
    Returns:
        Словарь с метаданными (без None значений)
    """
    
    year, quarter = _year_quarter(row.date)
    keywords = extract_keywords(text, top_k=5, use_keybert=False)
    
    # Создаем базовые метаданные с проверкой на None/NaN
    import pandas as pd
    import math
    
    # Вспомогательная функция для безопасного получения значений
    def safe_value(val, default=""):
        if val is None:
            return default
        if isinstance(val, float) and math.isnan(val):
            return default
        if pd.isna(val) if hasattr(pd, 'isna') else False:
            return default
        return val
    
    # Создаем метаданные с безопасными значениями
    channel_val = safe_value(getattr(row, 'channel', None), "unknown")
    msg_id_val = safe_value(getattr(row, 'msg_id', None), "")
    date_val = getattr(row, 'date', None)
    date_str = date_val.isoformat() if date_val and not pd.isna(date_val) else ""
    
    meta = {
        "source": "telegram",
        "channel": str(channel_val),
        "msg_ids": str(msg_id_val),
        "msg_dates": date_str,
        "start_ts": date_str,
        "end_ts": date_str,
        "year": int(year) if year and not pd.isna(year) else 0,
        "quarter": int(quarter) if quarter and not pd.isna(quarter) else 0,
        "doc_type": "tg_post_chunk",
        "keywords": keywords_to_string(keywords) or "",
        "is_forwarded": bool(is_forwarded) if is_forwarded else False,
    }
    
    # Добавляем forwarded_from только если он не None (ChromaDB не принимает None)
    if forwarded_from:
        meta["forwarded_from"] = str(forwarded_from)
    
    return meta


def build_tg_chunks(channel: str, min_chars: int = 50, max_chars: int = 3000) -> List[TgChunk]:
    """
    Создает чанки по одному сообщению (не объединяет несколько сообщений).
    Фильтрует короткие сообщения и обрабатывает пересылки.
    
    Args:
        channel: Имя канала
        min_chars: Минимальная длина сообщения для индексации (по умолчанию 50)
        max_chars: Максимальная длина чанка (для длинных сообщений применяется semantic chunking)
    """
    from tg_channels.summaries import _load_raw_channel_df
    
    logger.info(f"[INGEST] Начало построения чанков для канала: {channel}")
    logger.info(f"[INGEST] Минимальная длина сообщения: {min_chars} символов")
    logger.info(f"[INGEST] Максимальный размер чанка: {max_chars} символов")
    
    df = _load_raw_channel_df(channel)
    logger.info(f"[INGEST] Загружено сообщений из БД: {len(df)}")
    
    # Сортируем по времени
    df = df.sort_values("date")
    
    # Фильтруем короткие сообщения
    df = df[df["text"].str.len() >= min_chars]
    logger.info(f"[INGEST] После фильтрации (>= {min_chars} символов): {len(df)} сообщений")
    
    chunks: List[TgChunk] = []
    long_messages_count = 0
    semantic_chunks_count = 0
    forwarded_messages_count = 0
    
    for row in df.itertuples():
        txt = (row.text or "").strip()
        if not txt or len(txt) < min_chars:
            continue
        
        # Обработка пересылок
        is_forwarded = getattr(row, 'is_forwarded', False)
        forwarded_from = getattr(row, 'forwarded_from_channel', None)
        
        if is_forwarded and forwarded_from:
            # Пересылка: помечаем как ссылку на другой пост
            # Формируем текст с указанием источника
            text_with_source = f"[Пересылка из {forwarded_from}]\n{txt}"
            forwarded_messages_count += 1
        else:
            text_with_source = txt
        
        # Проверяем длину сообщения
        if len(text_with_source) > 500:
            long_messages_count += 1
            logger.debug(f"[INGEST] Длинное сообщение ({len(text_with_source)} символов), применяем semantic chunking")
            # Разбиваем длинное сообщение семантически
            semantic_parts = semantic_chunk(text_with_source, max_chars=max_chars, min_chars=200)
            semantic_chunks_count += len(semantic_parts)
            
            for part_idx, part in enumerate(semantic_parts):
                if len(part) > max_chars:
                    # Если часть все еще длинная, разбиваем по символам
                    for i in range(0, len(part), max_chars):
                        sub_part = part[i:i+max_chars]
                        if sub_part.strip():
                            meta = _create_chunk_metadata(row, sub_part, is_forwarded, forwarded_from)
                            chunks.append(TgChunk(text=sub_part.strip(), metadata=meta))
                else:
                    meta = _create_chunk_metadata(row, part, is_forwarded, forwarded_from)
                    chunks.append(TgChunk(text=part, metadata=meta))
        else:
            # Обычное сообщение (не длинное)
            meta = _create_chunk_metadata(row, text_with_source, is_forwarded, forwarded_from)
            chunks.append(TgChunk(text=text_with_source, metadata=meta))
    
    logger.info(f"[INGEST] Построение чанков завершено:")
    logger.info(f"[INGEST]   - Всего чанков: {len(chunks)}")
    logger.info(f"[INGEST]   - Длинных сообщений (>500 символов): {long_messages_count}")
    logger.info(f"[INGEST]   - Семантических частей: {semantic_chunks_count}")
    logger.info(f"[INGEST]   - Пересылок: {forwarded_messages_count}")
    return chunks


def ingest_tg_channel(channel: str, max_chars: int = 600, min_chars: int = 50, delete_from_db: bool = False) -> None:
    """
    Индексирует канал в ChromaDB.
    
    Args:
        channel: Имя канала
        max_chars: Максимальная длина чанка
        min_chars: Минимальная длина сообщения для индексации
        delete_from_db: Если True, удаляет канал из SQLite БД перед индексацией (по умолчанию False)
    """
    import time
    start_time = time.time()
    
    logger.info(f"[INGEST] ========== Начало индексации канала: {channel} ==========")
    logger.info(f"[INGEST] Параметры: max_chars={max_chars}, min_chars={min_chars}, delete_from_db={delete_from_db}")
    
    chunks = build_tg_chunks(channel, min_chars=min_chars, max_chars=max_chars)
    logger.info(f"[INGEST] Построено чанков: {len(chunks)}")

    if not chunks:
        logger.warning(f"[INGEST] Нет чанков для индексации, пропускаем")
        return

    logger.info(f"[INGEST] Подключение к ChromaDB...")
    store = ChromaStore()

    # Удаляем старые чанки этого канала из ChromaDB (но не из SQLite БД)
    logger.info(f"[INGEST] Удаление старых чанков канала из ChromaDB...")
    old_count = store.count_chunks_for_channel(channel)
    
    # Удаляем ВСЕ данные канала (чанки, саммари, author_report)
    store.collection.delete(where={"channel": channel})
    
    logger.info(f"[INGEST] Удалено старых чанков из ChromaDB: {old_count}")

    # Формируем payload в том же формате, что loader/splitter
    logger.info(f"[INGEST] Подготовка payload для добавления в ChromaDB...")
    payload = []
    for idx, ch in enumerate(chunks):
        meta = ch.metadata.copy()
        # Гарантированно уникальный id
        meta["chunk_id"] = f"tg_{channel}_{idx}"
        payload.append({"text": ch.text, "metadata": meta})
        if (idx + 1) % 100 == 0:
            logger.info(f"[INGEST] Подготовлено {idx + 1}/{len(chunks)} чанков...")

    logger.info(f"[INGEST] Добавление {len(payload)} чанков в ChromaDB...")
    store.add_chunks(payload)
    
    elapsed = time.time() - start_time
    new_count = store.count_chunks_for_channel(channel)
    logger.info(f"[INGEST] ========== Индексация завершена ==========")
    logger.info(f"[INGEST] Время выполнения: {elapsed:.2f}s")
    logger.info(f"[INGEST] Чанков в ChromaDB для {channel}: {new_count}")
    print(f"✅ tg-ingest done: {len(chunks)} chunks за {elapsed:.2f}s")


def reindex_tg_channel(channel: str, max_chars: int = 600, min_chars: int = 50) -> None:
    """
    Переиндексирует канал: удаляет только из ChromaDB и индексирует заново.
    Не удаляет данные из SQLite БД и не удаляет файлы.
    
    Args:
        channel: Имя канала
        max_chars: Максимальная длина чанка
        min_chars: Минимальная длина сообщения для индексации
    """
    logger.info(f"[REINDEX] ========== Переиндексация канала: {channel} ==========")
    logger.info(f"[REINDEX] Данные из SQLite БД и файлы сохраняются")
    
    # Используем обычную индексацию, которая удаляет только из ChromaDB
    ingest_tg_channel(channel, max_chars=max_chars, min_chars=min_chars, delete_from_db=False)
    
    logger.info(f"[REINDEX] ========== Переиндексация завершена ==========")
