"""Утилиты для генерации ссылок на источники сообщений Telegram."""
from typing import List, Dict, Any, Optional
import html
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


def format_source_reference_html(channel: str, msg_id: int, index: int, is_forwarded: bool = False, forwarded_from: str = None) -> str:
    """
    Форматирует кликабельную ссылку на источник в HTML формате [N].
    Использует HTML parse mode для кликабельных ссылок.
    
    Args:
        channel: Имя канала (без @)
        msg_id: ID сообщения
        index: Номер ссылки (1, 2, 3...)
        is_forwarded: Является ли сообщение пересылкой
        forwarded_from: Откуда переслано (если пересылка)
        
    Returns:
        HTML строка вида '<a href="t.me/channel/123">[1]</a>' или '<a href="t.me/channel/123">[1] (пересылка из @source)</a>'
    """
    link = f"https://t.me/{channel}/{msg_id}"
    label = f"[{index}]"
    if is_forwarded and forwarded_from:
        label = f"[{index}] (пересылка из @{forwarded_from})"
    return f'<a href="{html.escape(link)}">{html.escape(label)}</a>'


def format_source_reference_plain(channel: str, msg_id: int, index: int) -> str:
    """
    Форматирует ссылку на источник в простом текстовом формате [N] url.
    Используется как fallback если HTML не поддерживается.
    
    Args:
        channel: Имя канала (без @)
        msg_id: ID сообщения
        index: Номер ссылки (1, 2, 3...)
        
    Returns:
        Строка вида "[1] t.me/channel/123"
    """
    link = f"t.me/{channel}/{msg_id}"
    return f"[{index}] {link}"


def extract_source_links_from_metadata(meta: Dict[str, Any]) -> List[tuple[str, int]]:
    """
    Извлекает список (channel, msg_id) из метаданных чанка.
    
    Args:
        meta: Метаданные чанка
        
    Returns:
        Список кортежей (channel, msg_id)
    """
    channel = meta.get("channel", "")
    msg_ids_str = meta.get("msg_ids", "")
    
    if not channel or not msg_ids_str:
        return []
    
    try:
        msg_ids = [int(mid.strip()) for mid in msg_ids_str.split(",") if mid.strip()]
        return [(channel, msg_id) for msg_id in msg_ids]
    except (ValueError, AttributeError):
        return []


def format_sources_section(sources: List[tuple], use_html: bool = True) -> str:
    """
    Форматирует секцию источников для добавления в конец ответа.
    
    Args:
        sources: Список кортежей (channel, msg_id) или (channel, msg_id, is_forwarded, forwarded_from)
        use_html: Использовать HTML формат для кликабельных ссылок
        
    Returns:
        Отформатированная строка с источниками
    """
    if not sources:
        return ""
    
    seen = set()
    unique_sources = []
    for source in sources:
        if len(source) >= 2:
            channel, msg_id = source[0], source[1]
            key = (channel, msg_id)
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
    
    if use_html:
        # Форматируем все источники в одной строчке
        source_links = []
        for idx, source in enumerate(unique_sources[:10], 1):
            if len(source) == 4:
                channel, msg_id, is_forwarded, forwarded_from = source
                source_links.append(format_source_reference_html(channel, msg_id, idx, is_forwarded, forwarded_from))
            else:
                channel, msg_id = source[0], source[1]
                source_links.append(format_source_reference_html(channel, msg_id, idx))
        
        sources_text = " ".join(source_links)
        if len(unique_sources) > 10:
            sources_text += f" ... и еще {len(unique_sources) - 10} источников"
        
        return f"\n\nОтвет основан на постах: {sources_text}"
    else:
        # Форматируем все источники в одной строчке (plain text)
        source_links = []
        for idx, source in enumerate(unique_sources[:10], 1):
            channel, msg_id = source[0], source[1]
            source_links.append(format_source_reference_plain(channel, msg_id, idx))
        
        sources_text = " ".join(source_links)
        if len(unique_sources) > 10:
            sources_text += f" ... и еще {len(unique_sources) - 10} источников"
        
        return f"\n\nОтвет основан на постах: {sources_text}"


def filter_sources_strict(
    scored_chunks: List[tuple],
    max_sources: int = 7
) -> List[tuple]:
    """
    Фильтрация источников: извлекает msg_id из каждого чанка.
    Так как 1 чанк = 1 сообщение, просто берем msg_id из метаданных.
    
    Returns:
        Список кортежей (channel, msg_id, is_forwarded, forwarded_from)
    """
    if not scored_chunks:
        return []
    
    all_sources = []
    
    for score, doc, meta in scored_chunks[:max_sources]:
        channel = meta.get("channel", "")
        msg_ids_str = meta.get("msg_ids", "")
        
        if not channel or not msg_ids_str:
            continue
        
        try:
            msg_id = int(msg_ids_str.strip())
            is_forwarded = meta.get("is_forwarded", False)
            forwarded_from = meta.get("forwarded_from")
            all_sources.append((channel, msg_id, is_forwarded, forwarded_from))
            
            if len(all_sources) >= max_sources:
                break
                
        except (ValueError, AttributeError) as e:
            logger.warning(f"[SOURCE] Ошибка при обработке чанка: {e}")
            continue
    
    seen = set()
    unique_sources = []
    for source in all_sources:
        if source not in seen:
            seen.add(source)
            unique_sources.append(source)
    
    return unique_sources[:max_sources]

