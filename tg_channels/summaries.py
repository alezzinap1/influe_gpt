# telegram_summaries.py

from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd
import re
import asyncio
import threading
import time
import logging

from config_telegram import RAW_DIR
from rag.llm_backends import DeepSeekBackend, GeminiBackend, OpenAIBackend
from settings import SUMMARY_BACKEND
from tg_channels.prompts import (
    YEAR_SUMMARY_PROMPT,
    QUARTER_SUMMARY_PROMPT,
    AUTHOR_QUARTER_PROMPT,
    AUTHOR_YEAR_PROMPT,
    AUTHOR_REPORT_PROMPT,
    AUTHOR_REPORT_IDENTITY_STYLE_PROMPT,
    AUTHOR_REPORT_EVOLUTION_STRENGTHS_PROMPT,
    YEAR_EVOLUTION_PARAGRAPH_PROMPT,
)

logger = logging.getLogger(__name__)

from settings import LLM_SEMAPHORE_LIMIT

# максимально 8 одновременных LLM-запроса (общий лимит на все задачи)
_LLM_SEMAPHORE_LIMIT = LLM_SEMAPHORE_LIMIT
# Хранилище семафоров для каждого event loop (по id loop)
_LLM_SEMAPHORES = {}
# Блокировка для thread-safe доступа к словарю семафоров
_semaphore_lock = threading.Lock()

def _get_semaphore():
    """Получить семафор для текущего event loop (thread-safe)."""
    # В async контексте всегда должен быть running loop
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    
    # Проверяем без блокировки для быстрого пути
    if loop_id in _LLM_SEMAPHORES:
        return _LLM_SEMAPHORES[loop_id]
    
    # Создаем семафор с блокировкой для thread-safety
    with _semaphore_lock:
        # Double-check после получения блокировки
        if loop_id not in _LLM_SEMAPHORES:
            _LLM_SEMAPHORES[loop_id] = asyncio.Semaphore(_LLM_SEMAPHORE_LIMIT)
    
    return _LLM_SEMAPHORES[loop_id]


def _get_summary_llm():
    """
    Получить LLM бэкенд для создания саммари на основе настройки SUMMARY_BACKEND.
    
    Returns:
        LLMBackend: Экземпляр выбранного бэкенда (DeepSeekBackend, GeminiBackend или OpenAIBackend)
    """
    backend = SUMMARY_BACKEND.lower()
    
    if backend == "gemini":
        return GeminiBackend(model="gemini-3-flash-preview")
    elif backend == "openai":
        return OpenAIBackend()
    elif backend == "deepseek":
        return DeepSeekBackend(model="deepseek-chat")
    else:
        logger.warning(f"Неизвестный бэкенд саммари: {backend}, используем DeepSeek по умолчанию")
        return DeepSeekBackend(model="deepseek-chat")

SHORT_TEXT_MIN_LEN = 10  # жёсткий порог длины

URL_RE = re.compile(r"https?://\S+")
EMOJI_RE = re.compile(r"[\u2600-\u27BF\u1F300-\u1F6FF]+")


async def _run_in_thread(func, *args, **kwargs):
    """Запустить синхронную функцию в отдельном потоке под семафором."""
    semaphore = _get_semaphore()
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _looks_like_noise(text: str) -> bool:
    t = text.strip().lower()

    if not t:
        return True

    if len(t) <= 3 and not re.search(r"[a-zа-я0-9]", t):
        return True

    noise_tokens = {
        "ok",
        "ок",
        "++",
        "+",
        "да",
        "нет",
        "ага",
        "тест",
        "test",
        "upd",
        "update",
    }
    if t in noise_tokens:
        return True

    if URL_RE.fullmatch(t):
        return True

    if not re.search(r"[a-zа-я0-9]", t):
        return True

    return False


def clean_raw_messages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ("is_service", "service", "action"):
        if col in df.columns:
            df = df[
                ~df[col]
                .astype(str)
                .str.contains("True|joined|left|pinned", case=False, na=False)
            ]

    df["text"] = df.get("text", "").fillna("").astype(str)

    df = df[df["text"].str.len() >= SHORT_TEXT_MIN_LEN]

    df = df[~df["text"].apply(_looks_like_noise)]

    df = df.drop_duplicates(subset=["text"])

    def _too_link_heavy(t: str) -> bool:
        links = len(URL_RE.findall(t))
        return links >= 2 and len(t) < 3 * 80

    df = df[~df["text"].apply(_too_link_heavy)]

    return df


@dataclass
class QuarterSummary:
    channel: str
    year: int
    quarter: int
    summary_text: str


@dataclass
class PeriodSummary:
    channel: str
    year: int
    summary_text: str


def _load_raw_channel_df(channel: str) -> pd.DataFrame:
    safe = channel.lstrip("@").replace("/", "_")
    path = RAW_DIR / f"{safe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"raw parquet not found: {path}")
    df = pd.read_parquet(path)
    df = clean_raw_messages(df)
    return df


def _group_messages_by_year(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    df = df.copy()
    df["year"] = df["date"].dt.year
    groups: Dict[int, pd.DataFrame] = {}
    for year, g in df.groupby("year"):
        groups[int(year)] = g.sort_values("date")
    return groups


def _group_messages_by_year_quarter(
    df: pd.DataFrame,
) -> Dict[Tuple[int, int], pd.DataFrame]:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["quarter"] = ((df["date"].dt.month - 1) // 3 + 1).astype(int)
    groups: Dict[Tuple[int, int], pd.DataFrame] = {}
    for (year, quarter), g in df.groupby(["year", "quarter"]):
        groups[(int(year), int(quarter))] = g.sort_values("date")
    return groups


def _smart_select_messages(texts: list[str], max_chars: int = 9600) -> list[str]:
    """
    Улучшенная умная выборка сообщений для контекста саммари.
    
    Стратегия:
    1. Гарантированное покрытие всего периода (начало, середина, конец)
    2. Гибридный подход: длинные сообщения + равномерная выборка
    3. Ограничение на размер одного сообщения для предотвращения доминирования
    4. Эффективное использование лимита (до 95-98%)
    5. Сохранение хронологического порядка
    
    Args:
        texts: Список текстов сообщений в хронологическом порядке
        max_chars: Максимальная длина контекста (по умолчанию 9600, +20% от 8000)
    
    Returns:
        Отобранные сообщения в оригинальном порядке
    """
    if not texts:
        return []
    
    total_chars = sum(len(t) for t in texts)
    if total_chars <= max_chars:
        return texts
    
    n = len(texts)
    if n == 0:
        return []
    
    # Ограничение на максимальный размер одного сообщения (1/4 от лимита)
    # Предотвращает ситуацию, когда одно очень длинное сообщение занимает большую часть лимита
    max_single_msg = max_chars // 4  # ~2400 символов для лимита 9600
    
    # Разделяем на 3 части по времени для гарантированного покрытия периода
    part_size = max(1, n // 3)
    parts = [
        (0, texts[:part_size]),  # Начало периода
        (1, texts[part_size:2*part_size]),  # Середина периода
        (2, texts[2*part_size:])  # Конец периода
    ]
    
    # Бюджет на каждую часть: 30% на длинные, 20% на равномерные = 50% на часть
    # Остальные 50% будут использованы для финального заполнения
    budget_per_part_long = int(max_chars * 0.30)  # 30% на длинные из каждой части
    budget_per_part_uniform = int(max_chars * 0.20)  # 20% на равномерные из каждой части
    
    selected_indices = set()
    selected_with_indices = []  # (original_index, text)
    chars_used = 0
    
    # Фаза 1: Выборка из каждой части (длинные + равномерные)
    for part_idx, part_texts in parts:
        if not part_texts:
            continue
        
        # Создаем список с индексами для этой части
        if part_idx == 0:
            start_idx = 0
        elif part_idx == 1:
            start_idx = part_size
        else:  # part_idx == 2
            start_idx = 2 * part_size
        part_with_indices = [(start_idx + i, text) for i, text in enumerate(part_texts)]
        
        # 1.1. Выбираем длинные сообщения из этой части (но с ограничением на размер)
        part_sorted_by_length = sorted(part_with_indices, key=lambda x: len(x[1]), reverse=True)
        
        part_long_chars = 0
        for orig_idx, msg in part_sorted_by_length:
            if orig_idx in selected_indices:
                continue
            
            msg_len = len(msg)
            # Обрезаем слишком длинные сообщения
            if msg_len > max_single_msg:
                msg = msg[:max_single_msg]
                msg_len = max_single_msg
            
            # Проверяем, влезает ли в общий лимит и бюджет части
            if (chars_used + msg_len <= max_chars * 0.95 and  # Используем до 95% лимита
                part_long_chars + msg_len <= budget_per_part_long):
                selected_with_indices.append((orig_idx, msg))
                selected_indices.add(orig_idx)
                chars_used += msg_len
                part_long_chars += msg_len
        
        # 1.2. Равномерная выборка из этой части (гарантируем покрытие)
        part_uniform_chars = 0
        remaining_in_part = [(idx, text) for idx, text in part_with_indices 
                            if idx not in selected_indices]
        
        if remaining_in_part:
            # Берем равномерно распределенные сообщения
            step = max(1, len(remaining_in_part) // max(1, (budget_per_part_uniform // 150)))
            for i in range(0, len(remaining_in_part), step):
                orig_idx, msg = remaining_in_part[i]
                msg_len = len(msg)
                
                if (chars_used + msg_len <= max_chars * 0.95 and
                    part_uniform_chars + msg_len <= budget_per_part_uniform):
                    selected_with_indices.append((orig_idx, msg))
                    selected_indices.add(orig_idx)
                    chars_used += msg_len
                    part_uniform_chars += msg_len
                else:
                    # Пробуем добавить обрезанное, если осталось место
                    remaining = min(max_chars * 0.95 - chars_used, 
                                  budget_per_part_uniform - part_uniform_chars)
                    if remaining > 100:  # Минимум 100 символов
                        trimmed_msg = msg[:int(remaining)]
                        selected_with_indices.append((orig_idx, trimmed_msg))
                        selected_indices.add(orig_idx)
                        chars_used += len(trimmed_msg)
                        part_uniform_chars += len(trimmed_msg)
                    break
    
    # Фаза 2: Заполнение оставшегося места короткими сообщениями
    # Используем оставшиеся 5-10% лимита для увеличения разнообразия
    remaining_budget = int(max_chars * 0.98) - chars_used  # Стремимся к 98% использования
    
    if remaining_budget > 200:  # Если осталось достаточно места
        all_remaining = [(i, text) for i, text in enumerate(texts) 
                        if i not in selected_indices]
        
        if all_remaining:
            # Сортируем по длине (сначала короткие, потом средние)
            # Это позволяет добавить больше разнообразия
            all_remaining_sorted = sorted(all_remaining, key=lambda x: len(x[1]))
            
            for orig_idx, msg in all_remaining_sorted:
                msg_len = len(msg)
                if chars_used + msg_len <= max_chars * 0.98:
                    selected_with_indices.append((orig_idx, msg))
                    selected_indices.add(orig_idx)
                    chars_used += msg_len
                else:
                    # Пробуем добавить обрезанное
                    remaining = int(max_chars * 0.98) - chars_used
                    if remaining > 100:
                        trimmed_msg = msg[:remaining]
                        selected_with_indices.append((orig_idx, trimmed_msg))
                        chars_used += len(trimmed_msg)
                    break
    
    # Восстанавливаем хронологический порядок
    selected_with_indices.sort(key=lambda x: x[0])
    return [msg for _, msg in selected_with_indices]


def _build_year_prompt(channel: str, year: int, texts: list[str]) -> tuple[str, int, int, int]:
    """
    Строит промпт для годового саммари с умной выборкой сообщений.
    Возвращает (prompt, original_count, selected_count, context_chars)
    """
    original_count = len(texts)
    selected_texts = _smart_select_messages(texts, max_chars=9600)  # +20% лимит
    selected_count = len(selected_texts)
    joined = "\n\n".join(selected_texts)
    context_chars = len(joined)
    
    prompt = YEAR_SUMMARY_PROMPT.format(channel=channel, year=year, messages=joined)
    return prompt, original_count, selected_count, context_chars


def _build_quarter_prompt(
    channel: str, year: int, quarter: int, texts: list[str]
) -> tuple[str, int, int, int]:
    """
    Строит промпт для квартального саммари с умной выборкой сообщений.
    Возвращает (prompt, original_count, selected_count, context_chars)
    """
    original_count = len(texts)
    selected_texts = _smart_select_messages(texts, max_chars=9600)  # +20% лимит
    selected_count = len(selected_texts)
    joined = "\n\n".join(selected_texts)
    context_chars = len(joined)
    
    prompt = QUARTER_SUMMARY_PROMPT.format(
        channel=channel, year=year, quarter=quarter, messages=joined
    )
    return prompt, original_count, selected_count, context_chars


def build_quarter_summaries(channel: str) -> list[QuarterSummary]:
    df = _load_raw_channel_df(channel)
    groups = _group_messages_by_year_quarter(df)
    llm = _get_summary_llm()

    summaries: list[QuarterSummary] = []
    total_start = time.time()
    
    logger.info(f"[SUMMARY] Начало создания квартальных саммари для {channel}")
    
    for (year, quarter), g in sorted(groups.items()):
        texts = [str(t).strip() for t in g["text"].tolist() if str(t).strip()]
        if not texts:
            continue
        
        start_time = time.time()
        prompt, orig_count, sel_count, ctx_chars = _build_quarter_prompt(channel, year, quarter, texts)
        
        logger.info(f"[SUMMARY] {channel} {year}Q{quarter}: {orig_count}→{sel_count} msg, {ctx_chars:,} chars")
        
        summary = llm.generate(prompt)
        elapsed = time.time() - start_time
        
        logger.info(f"[SUMMARY] {channel} {year}Q{quarter}: готово за {elapsed:.1f}s")
        
        summaries.append(
            QuarterSummary(
                channel=channel,
                year=year,
                quarter=quarter,
                summary_text=summary,
            )
        )
    
    total_elapsed = time.time() - total_start
    logger.info(f"[SUMMARY] Всего создано {len(summaries)} квартальных саммари за {total_elapsed:.1f}s")
    
    return summaries


def build_year_summaries(channel: str) -> List[PeriodSummary]:
    df = _load_raw_channel_df(channel)
    groups = _group_messages_by_year(df)
    llm = _get_summary_llm()

    summaries: List[PeriodSummary] = []
    total_start = time.time()
    
    logger.info(f"[SUMMARY] Начало создания годовых саммари для {channel}")
    
    for year, g in sorted(groups.items()):
        texts = [str(t).strip() for t in g["text"].tolist() if str(t).strip()]
        if not texts:
            continue
        
        start_time = time.time()
        prompt, orig_count, sel_count, ctx_chars = _build_year_prompt(channel, year, texts)
        
        logger.info(f"[SUMMARY] {channel} {year}: {orig_count}→{sel_count} msg, {ctx_chars:,} chars")
        
        summary = llm.generate(prompt)
        elapsed = time.time() - start_time
        
        logger.info(f"[SUMMARY] {channel} {year}: готово за {elapsed:.1f}s")
        
        summaries.append(
            PeriodSummary(channel=channel, year=year, summary_text=summary)
        )
    
    total_elapsed = time.time() - total_start
    logger.info(f"[SUMMARY] Всего создано {len(summaries)} годовых саммари за {total_elapsed:.1f}s")
    
    return summaries


def save_quarter_summaries_parquet(
    channel: str, summaries: list[QuarterSummary]
) -> Path:
    if not summaries:
        raise ValueError("No quarter summaries to save")

    rows = [
        {
            "channel": s.channel,
            "year": s.year,
            "quarter": s.quarter,
            "summary_text": s.summary_text,
        }
        for s in summaries
    ]
    df = pd.DataFrame(rows)
    safe = channel.lstrip("@").replace("/", "_")
    out_dir = Path("data") / "processed" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries_quarter.parquet"
    df.to_parquet(path, index=False)
    print(f"💾 saved quarter summaries to {path}")
    return path


def save_year_summaries_parquet(
    channel: str, summaries: List[PeriodSummary]
) -> Path:
    if not summaries:
        raise ValueError("No summaries to save")

    rows = [
        {"channel": s.channel, "year": s.year, "summary_text": s.summary_text}
        for s in summaries
    ]
    df = pd.DataFrame(rows)
    safe = channel.lstrip("@").replace("/", "_")
    out_dir = Path("data") / "processed" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries_year.parquet"
    df.to_parquet(path, index=False)
    print(f"💾 saved year summaries to {path}")
    return path


def load_year_summaries(channel: str) -> Dict[int, str]:
    safe = channel.lstrip("@").replace("/", "_")
    path = Path("data") / "processed" / safe / "summaries_year.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[int, str] = {}
    for row in df.itertuples():
        out[int(row.year)] = str(row.summary_text)
    return out


def load_quarter_summaries(channel: str) -> Dict[Tuple[int, int], str]:
    safe = channel.lstrip("@").replace("/", "_")
    path = Path("data") / "processed" / safe / "summaries_quarter.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[Tuple[int, int], str] = {}
    for row in df.itertuples():
        out[(int(row.year), int(row.quarter))] = str(row.summary_text)
    return out


def build_all_summaries_for_channel(channel: str):
    year_summ = build_year_summaries(channel)
    if year_summ:
        save_year_summaries_parquet(channel, year_summ)
    q_summ = build_quarter_summaries(channel)
    if q_summ:
        save_quarter_summaries_parquet(channel, q_summ)


# ===== АВТОРСКИЕ (СТИЛИСТИЧЕСКИЕ) САММАРИ =====


def _build_author_quarter_prompt(
    channel: str, year: int, quarter: int, texts: list[str]
) -> tuple[str, int, int, int]:
    """
    Строит промпт для авторского квартального саммари с умной выборкой сообщений.
    Возвращает (prompt, original_count, selected_count, context_chars)
    """
    original_count = len(texts)
    selected_texts = _smart_select_messages(texts, max_chars=9600)  # +20% лимит
    selected_count = len(selected_texts)
    joined = "\n\n".join(selected_texts)
    context_chars = len(joined)
    
    prompt = AUTHOR_QUARTER_PROMPT.format(
        channel=channel, year=year, quarter=quarter, messages=joined
    )
    return prompt, original_count, selected_count, context_chars


def _build_author_year_prompt(
    channel: str, year: int, quarter_summaries: list[str]
) -> tuple[str, int, int, int]:
    """
    Строит промпт для авторского годового саммари с умной выборкой.
    Возвращает (prompt, original_count, selected_count, context_chars)
    """
    original_count = len(quarter_summaries)
    selected_texts = _smart_select_messages(quarter_summaries, max_chars=9600)  # +20% лимит
    selected_count = len(selected_texts)
    joined = "\n\n".join(selected_texts)
    context_chars = len(joined)
    
    prompt = AUTHOR_YEAR_PROMPT.format(
        channel=channel, year=year, quarter_summaries=joined
    )
    return prompt, original_count, selected_count, context_chars


def build_author_quarter_summaries(channel: str) -> list[QuarterSummary]:
    df = _load_raw_channel_df(channel)
    groups = _group_messages_by_year_quarter(df)
    llm = _get_summary_llm()

    summaries: list[QuarterSummary] = []
    total_start = time.time()
    
    logger.info(f"[SUMMARY] Начало создания авторских квартальных саммари для {channel}")
    
    for (year, quarter), g in sorted(groups.items()):
        texts = [str(t).strip() for t in g["text"].tolist() if str(t).strip()]
        if not texts:
            continue
        
        start_time = time.time()
        prompt, orig_count, sel_count, ctx_chars = _build_author_quarter_prompt(channel, year, quarter, texts)
        
        logger.info(f"[SUMMARY] {channel} {year}Q{quarter} (author): {orig_count}→{sel_count} msg, {ctx_chars:,} chars")
        
        summary = llm.generate(prompt)
        elapsed = time.time() - start_time
        
        logger.info(f"[SUMMARY] {channel} {year}Q{quarter} (author): готово за {elapsed:.1f}s")
        
        summaries.append(
            QuarterSummary(
                channel=channel,
                year=year,
                quarter=quarter,
                summary_text=summary,
            )
        )
    
    total_elapsed = time.time() - total_start
    logger.info(f"[SUMMARY] Всего создано {len(summaries)} авторских квартальных саммари за {total_elapsed:.1f}s")
    
    return summaries


def save_author_quarter_summaries_parquet(
    channel: str, summaries: list[QuarterSummary]
) -> Path:
    if not summaries:
        raise ValueError("No author quarter summaries to save")

    rows = [
        {
            "channel": s.channel,
            "year": s.year,
            "quarter": s.quarter,
            "summary_text": s.summary_text,
        }
        for s in summaries
    ]
    df = pd.DataFrame(rows)
    safe = channel.lstrip("@").replace("/", "_")
    out_dir = Path("data") / "processed" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries_quarter_author.parquet"
    df.to_parquet(path, index=False)
    print(f"💾 saved author quarter summaries to {path}")
    return path


def save_author_year_summaries_parquet_from_quarters(channel: str) -> Path:
    safe = channel.lstrip("@").replace("/", "_")
    q_path = Path("data") / "processed" / safe / "summaries_quarter_author.parquet"
    if not q_path.exists():
        raise FileNotFoundError(f"author quarter parquet not found: {q_path}")

    df_q = pd.read_parquet(q_path)
    llm = _get_summary_llm()
    rows: list[dict] = []

    total_start = time.time()
    logger.info(f"[SUMMARY] Начало создания авторских годовых саммари для {channel}")
    
    for year, g in df_q.groupby("year"):
        quarter_summaries = [
            str(t).strip()
            for t in g.sort_values("quarter")["summary_text"].tolist()
            if str(t).strip()
        ]
        if not quarter_summaries:
            continue
        
        start_time = time.time()
        prompt, orig_count, sel_count, ctx_chars = _build_author_year_prompt(channel, int(year), quarter_summaries)
        
        logger.info(f"[SUMMARY] {channel} {year} (author year): {orig_count}→{sel_count} q-summaries, {ctx_chars:,} chars")
        
        summary = llm.generate(prompt)
        elapsed = time.time() - start_time
        
        logger.info(f"[SUMMARY] {channel} {year} (author year): готово за {elapsed:.1f}s")
        rows.append(
            {
                "channel": channel,
                "year": int(year),
                "summary_text": summary,
            }
        )

    df_y = pd.DataFrame(rows)
    out_dir = Path("data") / "processed" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summaries_year_author.parquet"
    df_y.to_parquet(path, index=False)
    
    total_elapsed = time.time() - total_start
    logger.info(f"[SUMMARY] Всего создано {len(rows)} авторских годовых саммари за {total_elapsed:.1f}s")
    print(f"💾 saved author year summaries to {path}")
    return path


def load_author_year_summaries(channel: str) -> Dict[int, str]:
    safe = channel.lstrip("@").replace("/", "_")
    path = Path("data") / "processed" / safe / "summaries_year_author.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[int, str] = {}
    for row in df.itertuples():
        out[int(row.year)] = str(row.summary_text)
    return out


def load_author_quarter_summaries(
    channel: str,
) -> Dict[Tuple[int, int], str]:
    safe = channel.lstrip("@").replace("/", "_")
    path = Path("data") / "processed" / safe / "summaries_quarter_author.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: Dict[Tuple[int, int], str] = {}
    for row in df.itertuples():
        out[(int(row.year), int(row.quarter))] = str(row.summary_text)
    return out


def build_author_report(channel: str) -> str:
    part1 = build_author_report_identity_style(channel)
    part2 = build_author_report_evolution_strengths(channel)
    return part1 + "\n\n---\n\n" + part2


def save_author_report(channel: str, report: str) -> Path:
    safe = channel.lstrip("@").replace("/", "_")
    out_dir = Path("data") / "processed" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "author_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"💾 saved author report to {path}")
    return path


def build_author_report_identity_style(channel: str) -> str:
    year_summ = load_year_summaries(channel)
    author_year = load_author_year_summaries(channel)
    author_quarter = load_author_quarter_summaries(channel)

    if not (year_summ or author_year or author_quarter):
        raise RuntimeError("Нет данных для author_identity_style")

    parts: list[str] = []

    years = sorted(
        set(year_summ.keys())
        | set(author_year.keys())
        | {y for (y, _) in author_quarter.keys()},
        reverse=True,
    )

    for year in years:
        if year in author_year:
            parts.append(
                f"[Стиль] Год {year}:\n{author_year[year].strip()}"
            )

    for (y, q), txt in author_quarter.items():
        if y in years and txt.strip():
            parts.append(f"[Стиль] {y} Q{q}:\n{txt.strip()}")

    raw = "\n\n".join(parts)
    max_chars = 6000
    if len(raw) > max_chars:
        raw = raw[:max_chars]

    prompt = AUTHOR_REPORT_IDENTITY_STYLE_PROMPT.format(
        channel=channel, summaries=raw
    )
    llm = _get_summary_llm()
    return llm.generate(prompt).strip()


def build_author_report_evolution_strengths(channel: str) -> str:
    year_summ = load_year_summaries(channel)
    author_year = load_author_year_summaries(channel)

    if not (year_summ or author_year):
        raise RuntimeError("Нет данных для author_evolution_strengths")

    parts: list[str] = []
    years = sorted(set(year_summ.keys()) | set(author_year.keys()), reverse=True)

    for year in years:
        if year in year_summ:
            parts.append(f"[Рынок] {year}:\n{year_summ[year].strip()}")
        if year in author_year:
            parts.append(f"[Стиль] {year}:\n{author_year[year].strip()}")

    raw = "\n\n".join(parts)
    max_chars = 10000
    if len(raw) > max_chars:
        raw = raw[:max_chars]

    prompt = AUTHOR_REPORT_EVOLUTION_STRENGTHS_PROMPT.format(
        channel=channel, summaries=raw
    )
    llm = _get_summary_llm()
    return llm.generate(prompt).strip()


def _build_year_evolution_paragraph(channel: str, year: int) -> str:
    year_summ_all = load_year_summaries(channel)
    author_year_all = load_author_year_summaries(channel)

    year_summ = year_summ_all.get(year)
    author_year = author_year_all.get(year)

    if not (year_summ or author_year):
        return ""

    parts = []
    if year_summ:
        parts.append(f"[Рынок] {year}:\n{year_summ.strip()}")
    if author_year:
        parts.append(f"[Стиль] {year}:\n{author_year.strip()}")

    raw = "\n\n".join(parts)
    max_chars = 2000
    if len(raw) > max_chars:
        raw = raw[:max_chars]

    prompt = YEAR_EVOLUTION_PARAGRAPH_PROMPT.format(
        channel=channel, year=year, summaries=raw
    )
    llm = _get_summary_llm()
    return llm.generate(prompt).strip()


def build_author_report_evolution_strengths_full(channel: str) -> str:
    year_summ = load_year_summaries(channel)
    author_year = load_author_year_summaries(channel)

    if not (year_summ or author_year):
        raise RuntimeError("Нет данных для author_evolution_strengths")

    years = sorted(set(year_summ.keys()) | set(author_year.keys()))
    evolution_blocks: list[str] = []

    for year in years:
        para = _build_year_evolution_paragraph(channel, year)
        if para:
            evolution_blocks.append(f"- **{year}:** {para}")

    evolution_text = "\n".join(evolution_blocks)

    raw = evolution_text
    max_chars = 4000
    if len(raw) > max_chars:
        raw = raw[:max_chars]

    prompt = AUTHOR_REPORT_EVOLUTION_STRENGTHS_PROMPT.format(
        channel=channel, summaries=raw
    )
    llm = _get_summary_llm()
    summary = llm.generate(prompt).strip()
    return evolution_text + "\n\n" + summary


# ===== НОВЫЕ ВСПОМОГАТЕЛИ ДЛЯ ПАРАЛЛЕЛИЗМА =====


def _summarize_quarter_market(
    channel: str, year: int, quarter: int, texts: list[str]
) -> QuarterSummary:
    llm = _get_summary_llm()
    start_time = time.time()
    prompt, orig_count, sel_count, ctx_chars = _build_quarter_prompt(channel, year, quarter, texts)
    
    logger.info(f"[SUMMARY] {channel} {year}Q{quarter}: {orig_count}→{sel_count} msg, {ctx_chars:,} chars")
    
    summary = llm.generate(prompt)
    elapsed = time.time() - start_time
    
    logger.info(f"[SUMMARY] {channel} {year}Q{quarter}: готово за {elapsed:.1f}s")
    
    return QuarterSummary(
        channel=channel, year=year, quarter=quarter, summary_text=summary
    )


def _summarize_quarter_author(
    channel: str, year: int, quarter: int, texts: list[str]
) -> QuarterSummary:
    llm = _get_summary_llm()
    start_time = time.time()
    prompt, orig_count, sel_count, ctx_chars = _build_author_quarter_prompt(channel, year, quarter, texts)
    
    logger.info(f"[SUMMARY] {channel} {year}Q{quarter} (author): {orig_count}→{sel_count} msg, {ctx_chars:,} chars")
    
    summary = llm.generate(prompt)
    elapsed = time.time() - start_time
    
    logger.info(f"[SUMMARY] {channel} {year}Q{quarter} (author): готово за {elapsed:.1f}s")
    
    return QuarterSummary(
        channel=channel, year=year, quarter=quarter, summary_text=summary
    )


async def _summarize_quarter_market_async(
    channel: str, year: int, quarter: int, texts: list[str]
) -> QuarterSummary:
    return await _run_in_thread(
        _summarize_quarter_market, channel, year, quarter, texts
    )


async def _summarize_quarter_author_async(
    channel: str, year: int, quarter: int, texts: list[str]
) -> QuarterSummary:
    return await _run_in_thread(
        _summarize_quarter_author, channel, year, quarter, texts
    )


async def _phase_quarters_async(
    channel: str,
) -> Tuple[list[QuarterSummary], list[QuarterSummary]]:
    """
    Фаза 1: считаем все квартальные саммари (рыночные и авторские) в 3 потока.
    """
    total_start = time.time()
    df = _load_raw_channel_df(channel)
    groups = _group_messages_by_year_quarter(df)

    market_tasks: list[asyncio.Task] = []
    author_tasks: list[asyncio.Task] = []

    for (year, quarter), g in sorted(groups.items()):
        texts = [str(t).strip() for t in g["text"].tolist() if str(t).strip()]
        if not texts:
            continue
        market_tasks.append(
            asyncio.create_task(
                _summarize_quarter_market_async(channel, year, quarter, texts)
            )
        )
        author_tasks.append(
            asyncio.create_task(
                _summarize_quarter_author_async(channel, year, quarter, texts)
            )
        )

    market_summaries: list[QuarterSummary] = []
    author_summaries: list[QuarterSummary] = []

    if market_tasks:
        logger.info(f"[SUMMARY] Запуск {len(market_tasks)} рыночных квартальных саммари...")
        market_summaries = list(await asyncio.gather(*market_tasks))
    if author_tasks:
        logger.info(f"[SUMMARY] Запуск {len(author_tasks)} авторских квартальных саммари...")
        author_summaries = list(await asyncio.gather(*author_tasks))

    total_elapsed = time.time() - total_start
    logger.info(f"[SUMMARY] Квартальные саммари: {len(market_summaries)} market + {len(author_summaries)} author за {total_elapsed:.1f}s")

    return market_summaries, author_summaries


async def _phase_years_and_report_async(channel: str) -> None:
    """
    Фаза 2: после того, как квартальные parquet сохранены.
    Считаем годовые (рыночные и авторские) и финальный отчёт.
    """

    def _build_years_sync() -> list[PeriodSummary]:
        return build_year_summaries(channel)

    year_summ = await _run_in_thread(_build_years_sync)
    if year_summ:
        save_year_summaries_parquet(channel, year_summ)

    await _run_in_thread(save_author_year_summaries_parquet_from_quarters, channel)

    def _build_report_sync() -> str:
        return build_author_report(channel)

    report = await _run_in_thread(_build_report_sync)
    save_author_report(channel, report)


async def build_all_summaries_and_report_async(channel: str) -> None:
    """
    Двухфазный асинхронный пайплайн:
    1) все квартальные (market + author) в 3 потока;
    2) после этого годовые + author_year + финальный отчёт.
    """
    steps = 5
    step = 1

    print(
        f"[{step}/{steps}] Квартальные (рыночные + авторские) саммари..."
    )
    market_quarters, author_quarters = await _phase_quarters_async(channel)

    if market_quarters:
        save_quarter_summaries_parquet(channel, market_quarters)
    if author_quarters:
        save_author_quarter_summaries_parquet(channel, author_quarters)

    step = 4
    print(f"[{step}/{steps}] Авторские годовые саммари и отчёт...")
    await _phase_years_and_report_async(channel)

    print("✅ tg-build-summaries (async): все саммари и отчёт готовы.")
