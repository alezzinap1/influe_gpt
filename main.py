import argparse
import sys
import time
import asyncio
import logging
import os

# Настройка кодировки UTF-8 для Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Для старых версий Python
        os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, '.')

# Настройка логирования для саммари
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

from vectorstore.chromadb_store import ChromaStore
from rag.pipeline import RAGPipeline

from telethon import TelegramClient
from telethon.tl.types import Message
import pandas as pd

from config_telegram import raw_parquet_path, TG_API_ID, TG_API_HASH, TG_SESSION_NAME
from tg_channels.schema import TgMessage

from tg_channels.ingest import ingest_tg_channel

from tg_channels.summaries import load_year_summaries, load_quarter_summaries
from tg_channels.summaries import build_all_summaries_and_report_async

from typing import List
from settings import MAX_MSGS, LOG_STEP


async def _fetch_channel_messages(channel: str) -> List[TgMessage]:
    if not TG_API_ID or not TG_API_HASH:
        raise RuntimeError("Заполни TG_API_ID и TG_API_HASH в config_telegram.py")

    from config_telegram import normalize_channel_name
    
    client = TelegramClient(TG_SESSION_NAME, TG_API_ID, TG_API_HASH)
    chan = normalize_channel_name(channel)

    messages: List[TgMessage] = []
    BATCH_SIZE = 1000  # Размер батча для обработки в памяти

    async with client:
        # новые → старые
        async for msg in client.iter_messages(chan):
            if not isinstance(msg, Message):
                continue

            text = msg.message or ""
            if not text and not msg.media:
                continue

            # Определяем пересылку
            is_forwarded = False
            forwarded_from_channel = None
            forwarded_from_msg_id = None
            
            if hasattr(msg, 'fwd_from') and msg.fwd_from:
                is_forwarded = True
                # Пытаемся получить информацию об источнике пересылки
                if hasattr(msg.fwd_from, 'from_id'):
                    from_id = msg.fwd_from.from_id
                    # from_id может быть PeerChannel, PeerUser и т.д.
                    if hasattr(from_id, 'channel_id'):
                        # Это канал, но нам нужен username
                        # В telethon обычно нужно делать дополнительный запрос для получения username
                        # Пока сохраняем только ID
                        forwarded_from_channel = f"channel_{from_id.channel_id}"
                    elif hasattr(from_id, 'user_id'):
                        forwarded_from_channel = f"user_{from_id.user_id}"
                
                # Пытаемся получить оригинальный msg_id
                if hasattr(msg.fwd_from, 'channel_post'):
                    forwarded_from_msg_id = msg.fwd_from.channel_post

            m = TgMessage(
                channel=chan,
                msg_id=msg.id,
                date=msg.date,
                edit_date=msg.edit_date,
                sender_id=getattr(msg.sender, "id", None),
                text=text,
                views=msg.views,
                forwards=msg.forwards,
                reply_to_msg_id=getattr(
                    getattr(msg, "reply_to", None), "reply_to_msg_id", None
                ),
                has_media=msg.media is not None,
                is_forwarded=is_forwarded,
                forwarded_from_channel=forwarded_from_channel,
                forwarded_from_msg_id=forwarded_from_msg_id,
            )
            messages.append(m)

            n = len(messages)
            # Логируем прогресс
            if n % LOG_STEP == 0:
                print(f"[tg-sync] {chan}: скачано {n} сообщений")
            
            # Батчевая обработка: ограничиваем размер в памяти
            # На практике это не критично, так как мы все равно возвращаем весь список
            # Но это позволяет контролировать использование памяти
            if n >= MAX_MSGS:
                print(f"[tg-sync] {chan}: достигнут лимит {MAX_MSGS} сообщений, останавливаемся")
                break

    return messages

def cmd_tg_sync(args):
    """
    Полная первоначальная выгрузка истории канала в data/raw/<channel>.parquet
    Батчевая обработка для оптимизации памяти.
    """
    from config_telegram import normalize_channel_name
    
    channel = normalize_channel_name(args.channel)
    if not channel:
        print(f"Ошибка: некорректное имя канала: {args.channel}")
        return
    print(f"▶ tg-sync: fetching history for {channel}")

    msgs: List[TgMessage] = asyncio.run(_fetch_channel_messages(channel))
    print(f"✅ fetched {len(msgs)} messages")

    if not msgs:
        return

    # Сохраняем ID последнего сообщения (самое новое, с максимальным msg_id)
    last_msg_id = max(m.msg_id for m in msgs) if msgs else None

    # Батчевая обработка: конвертируем и сохраняем батчами для экономии памяти
    BATCH_SIZE = 5000
    path = raw_parquet_path(channel)
    
    # Если сообщений немного, сохраняем сразу
    if len(msgs) <= BATCH_SIZE:
        df = pd.DataFrame(
            [
                {
                    "channel": m.channel,
                    "msg_id": m.msg_id,
                    "date": m.date,
                    "edit_date": m.edit_date,
                    "sender_id": m.sender_id,
                    "text": m.text,
                    "views": m.views,
                    "forwards": m.forwards,
                    "reply_to_msg_id": m.reply_to_msg_id,
                    "has_media": m.has_media,
                    "is_forwarded": m.is_forwarded,
                    "forwarded_from_channel": m.forwarded_from_channel,
                    "forwarded_from_msg_id": m.forwarded_from_msg_id,
                }
                for m in msgs
            ]
        )
        df.to_parquet(path, index=False)
        print(f"💾 saved to {path}")
    else:
        # Для больших объемов: обрабатываем батчами и объединяем
        dfs = []
        for i in range(0, len(msgs), BATCH_SIZE):
            batch = msgs[i:i + BATCH_SIZE]
            batch_df = pd.DataFrame(
                [
                    {
                        "channel": m.channel,
                        "msg_id": m.msg_id,
                        "date": m.date,
                        "edit_date": m.edit_date,
                        "sender_id": m.sender_id,
                        "text": m.text,
                        "views": m.views,
                        "forwards": m.forwards,
                        "reply_to_msg_id": m.reply_to_msg_id,
                        "has_media": m.has_media,
                        "is_forwarded": m.is_forwarded,
                        "forwarded_from_channel": m.forwarded_from_channel,
                        "forwarded_from_msg_id": m.forwarded_from_msg_id,
                    }
                    for m in batch
                ]
            )
            dfs.append(batch_df)
            print(f"[tg-sync] Обработан батч {i//BATCH_SIZE + 1} ({len(batch)} сообщений)")
        
        # Объединяем все батчи
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(path, index=False)
        print(f"💾 saved {len(msgs)} messages to {path}")
    
    # Сохраняем last_synced_msg_id в БД
    if last_msg_id:
        from tgbot.bot_db import set_last_synced_msg_id
        set_last_synced_msg_id(channel, last_msg_id)
        print(f"💾 saved last_synced_msg_id: {last_msg_id}")



async def _fetch_channel_messages_incremental(channel: str, last_msg_id: int | None = None) -> List[TgMessage]:
    """
    Инкрементальная загрузка новых сообщений канала.
    Загружает только сообщения с msg_id > last_msg_id.
    """
    if not TG_API_ID or not TG_API_HASH:
        raise RuntimeError("Заполни TG_API_ID и TG_API_HASH в config_telegram.py")

    from config_telegram import normalize_channel_name
    
    client = TelegramClient(TG_SESSION_NAME, TG_API_ID, TG_API_HASH)
    chan = normalize_channel_name(channel)

    messages: List[TgMessage] = []

    async with client:
        # новые → старые
        async for msg in client.iter_messages(chan):
            if not isinstance(msg, Message):
                continue
            
            # Если есть last_msg_id, останавливаемся когда достигли его
            if last_msg_id is not None and msg.id <= last_msg_id:
                break

            text = msg.message or ""
            if not text and not msg.media:
                continue

            # Определяем пересылку
            is_forwarded = False
            forwarded_from_channel = None
            forwarded_from_msg_id = None
            
            if hasattr(msg, 'fwd_from') and msg.fwd_from:
                is_forwarded = True
                # Пытаемся получить информацию об источнике пересылки
                if hasattr(msg.fwd_from, 'from_id'):
                    from_id = msg.fwd_from.from_id
                    # from_id может быть PeerChannel, PeerUser и т.д.
                    if hasattr(from_id, 'channel_id'):
                        # Это канал, но нам нужен username
                        # В telethon обычно нужно делать дополнительный запрос для получения username
                        # Пока сохраняем только ID
                        forwarded_from_channel = f"channel_{from_id.channel_id}"
                    elif hasattr(from_id, 'user_id'):
                        forwarded_from_channel = f"user_{from_id.user_id}"
                
                # Пытаемся получить оригинальный msg_id
                if hasattr(msg.fwd_from, 'channel_post'):
                    forwarded_from_msg_id = msg.fwd_from.channel_post

            m = TgMessage(
                channel=chan,
                msg_id=msg.id,
                date=msg.date,
                edit_date=msg.edit_date,
                sender_id=getattr(msg.sender, "id", None),
                text=text,
                views=msg.views,
                forwards=msg.forwards,
                reply_to_msg_id=getattr(
                    getattr(msg, "reply_to", None), "reply_to_msg_id", None
                ),
                has_media=msg.media is not None,
                is_forwarded=is_forwarded,
                forwarded_from_channel=forwarded_from_channel,
                forwarded_from_msg_id=forwarded_from_msg_id,
            )
            messages.append(m)

            n = len(messages)
            # Логируем прогресс
            if n % LOG_STEP == 0:
                print(f"[tg-update] {chan}: скачано {n} новых сообщений")

    return messages


def cmd_tg_update(args):
    """
    Инкрементальное обновление истории канала.
    Загружает только новые сообщения с момента последней синхронизации.
    """
    from config_telegram import normalize_channel_name, raw_parquet_path
    from tgbot.bot_db import get_last_synced_msg_id, set_last_synced_msg_id
    
    channel = normalize_channel_name(args.channel)
    if not channel:
        print(f"Ошибка: некорректное имя канала: {args.channel}")
        return
    
    # Получаем ID последнего синхронизированного сообщения
    last_msg_id = get_last_synced_msg_id(channel)
    
    if last_msg_id is None:
        print(f"⚠️  Канал {channel} еще не синхронизирован. Используйте 'tg-sync' для первой синхронизации.")
        return
    
    print(f"▶ tg-update: fetching new messages for {channel} (после msg_id={last_msg_id})")
    
    # Загружаем только новые сообщения
    new_msgs: List[TgMessage] = asyncio.run(_fetch_channel_messages_incremental(channel, last_msg_id))
    print(f"✅ fetched {len(new_msgs)} новых сообщений")

    if not new_msgs:
        print("ℹ️  Новых сообщений нет.")
        return

    # Загружаем существующие данные
    path = raw_parquet_path(channel)
    if path.exists():
        existing_df = pd.read_parquet(path)
        print(f"📂 Загружено {len(existing_df)} существующих сообщений")
    else:
        existing_df = pd.DataFrame()
        print("📂 Существующих данных нет, создаем новый файл")

    # Добавляем новые сообщения
    new_df = pd.DataFrame(
        [
            {
                "channel": m.channel,
                "msg_id": m.msg_id,
                "date": m.date,
                "edit_date": m.edit_date,
                "sender_id": m.sender_id,
                "text": m.text,
                "views": m.views,
                "forwards": m.forwards,
                "reply_to_msg_id": m.reply_to_msg_id,
                "has_media": m.has_media,
                "is_forwarded": m.is_forwarded,
                "forwarded_from_channel": m.forwarded_from_channel,
                "forwarded_from_msg_id": m.forwarded_from_msg_id,
            }
            for m in new_msgs
        ]
    )
    
    # Объединяем и удаляем дубликаты (на случай если msg_id уже есть)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["msg_id"], keep="last")
    combined_df = combined_df.sort_values("msg_id", ascending=False)  # новые сверху
    
    # Сохраняем
    combined_df.to_parquet(path, index=False)
    print(f"💾 saved {len(combined_df)} total messages to {path}")
    
    # Обновляем last_synced_msg_id
    if new_msgs:
        new_last_msg_id = max(m.msg_id for m in new_msgs)
        set_last_synced_msg_id(channel, new_last_msg_id)
        print(f"💾 updated last_synced_msg_id: {new_last_msg_id}")


def cmd_tg_ingest(args):
    from config_telegram import normalize_channel_name
    
    channel = normalize_channel_name(args.channel)
    if not channel:
        print(f"Ошибка: некорректное имя канала: {args.channel}")
        return
    print(f"▶ tg-ingest: {channel}")
    ingest_tg_channel(channel)


def cmd_tg_reindex(args):
    """Переиндексирует канал без удаления из БД и файлов."""
    from config_telegram import normalize_channel_name
    from tg_channels.ingest import reindex_tg_channel
    
    channel = normalize_channel_name(args.channel)
    if not channel:
        print(f"Ошибка: некорректное имя канала: {args.channel}")
        return
    print(f"▶ tg-reindex: {channel} (данные из БД и файлы сохраняются)")
    reindex_tg_channel(channel)


def cmd_tg_delete(args):
    """Удаляет канал из БД, ChromaDB и файлов."""
    from config_telegram import normalize_channel_name
    from tgbot.bot_tasks import delete_channel_completely
    
    channel = normalize_channel_name(args.channel)
    if not channel:
        print(f"Ошибка: некорректное имя канала: {args.channel}")
        return
    
    print(f">> tg-delete: {channel}")
    print(f"[!] Внимание: операция необратима!")
    
    result = delete_channel_completely(channel)
    
    if result.get("success"):
        stats = result.get("chromadb_stats", {})
        print(f"[OK] {result.get('message', 'Канал удален')}")
        print(f"   Чанков удалено: {stats.get('chunks_deleted', 0)}")
        print(f"   Саммари удалено: {stats.get('summaries_deleted', 0)}")
        print(f"   Файлы: {'[OK]' if result.get('files_deleted') else '[NOT FOUND]'}")
    else:
        print(f"[ERROR] {result.get('message', 'Ошибка при удалении')}")
        if result.get("errors"):
            print(f"   Ошибки: {', '.join(result['errors'])}")


def cmd_tg_query(args):
    from config_telegram import normalize_channel_name
    
    channel = normalize_channel_name(args.channel)
    if not channel:
        print(f"Ошибка: некорректное имя канала: {args.channel}")
        return
    question = args.question

    print(f"▶ tg-query: channel={channel}, q={question!r}")
    rag = RAGPipeline(backend=args.backend)

    # грузим годовые и квартальные саммари
    year_summaries = load_year_summaries(channel)
    quarter_summaries = load_quarter_summaries(channel)

    start = time.time()
    ans = rag.query(
        question,
        source="telegram",
        channel=channel,
        extra_year_summaries=year_summaries,
        extra_quarter_summaries=quarter_summaries,
    )

    print(f"Total time: {time.time() - start:.1f}s")
    print(f"Канал: {channel}")
    print(f"Вопрос: {question}")
    print(f"Ответ:\n{ans}")

# ====== ЕДИНЫЙ CLI С SUBCOMMANDS ======

def build_cli():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- tg-* команды ----
    p_tg_sync = sub.add_parser("tg-sync", help="download Telegram channel history")
    p_tg_sync.add_argument("channel", help="@channelname")
    p_tg_sync.set_defaults(func=cmd_tg_sync)

    p_tg_update = sub.add_parser("tg-update", help="update Telegram channel history")
    p_tg_update.add_argument("channel", help="@channelname")
    p_tg_update.set_defaults(func=cmd_tg_update)

    p_tg_ingest = sub.add_parser("tg-ingest", help="ingest Telegram channel into vector index")
    p_tg_ingest.add_argument("channel", help="@channelname")
    p_tg_ingest.set_defaults(func=cmd_tg_ingest)

    p_tg_reindex = sub.add_parser("tg-reindex", help="reindex Telegram channel (keeps DB and files, only updates ChromaDB)")
    p_tg_reindex.add_argument("channel", help="@channelname")
    p_tg_reindex.set_defaults(func=cmd_tg_reindex)

    p_tg_delete = sub.add_parser("tg-delete", help="delete Telegram channel completely (removes from DB, ChromaDB and files)")
    p_tg_delete.add_argument("channel", help="@channelname")
    p_tg_delete.set_defaults(func=cmd_tg_delete)

    p_tg_query = sub.add_parser("tg-query", help="query Telegram channel archive")
    p_tg_query.add_argument("channel", help="@channelname")
    p_tg_query.add_argument("question", help="Вопрос к архивариусу канала")
    p_tg_query.add_argument(
        "--backend",
        choices=["local", "deepseek", "openai"],
        default="deepseek",
        help="LLM backend for query",
    )
    p_tg_query.set_defaults(func=cmd_tg_query)

    p_tg_build_sum = sub.add_parser("tg-build-summaries", help="build yearly summaries for Telegram channel")
    p_tg_build_sum.add_argument("channel", help="@channelname")
    p_tg_build_sum.set_defaults(func=cmd_tg_build_summaries)

    # ---- НОВЫЕ команды ----
    p_tg_index_summ = sub.add_parser("tg-index-summaries", help="индексировать саммари в Chroma")
    p_tg_index_summ.add_argument("channel", help="@channelname")
    p_tg_index_summ.set_defaults(func=cmd_tg_index_summaries)

    p_status = sub.add_parser("status", help="status of vector index")
    p_status.set_defaults(func=cmd_status)

    return parser


def cmd_tg_build_summaries(args):
    from config_telegram import normalize_channel_name
    
    channel = normalize_channel_name(args.channel)
    if not channel:
        print(f"Ошибка: некорректное имя канала: {args.channel}")
        return

    print(f"tg-build-summaries: {channel}")
    asyncio.run(build_all_summaries_and_report_async(channel))


def cmd_tg_index_summaries(args):
    from pathlib import Path
    import pandas as pd
    from config_telegram import normalize_channel_name

    channel = normalize_channel_name(args.channel)
    if not channel:
        print(f"Ошибка: некорректное имя канала: {args.channel}")
        return
    store = ChromaStore()

    # Удаляем старые саммари и author_report для этого канала перед индексацией
    # Это предотвращает дубликаты при пересборке
    print(f"🗑️  Удаление старых саммари для {channel}...")
    try:
        # Удаляем саммари (summary)
        store.collection.delete(
            where={
                "$and": [
                    {"channel": channel},
                    {"type": "summary"}
                ]
            }
        )
        # Удаляем author_report
        store.collection.delete(
            where={
                "$and": [
                    {"channel": channel},
                    {"type": "author_report"}
                ]
            }
        )
        print(f"✅ Старые саммари удалены")
    except Exception as e:
        print(f"⚠️  Ошибка при удалении старых саммари: {e}")
        # Продолжаем индексацию даже если удаление не удалось

    # Годовые + квартальные (все имеют summary_text)
    for path_str, doc_id in [
        (f"data/processed/{channel}/summaries_year.parquet", f"{channel}_year_summaries"),
        (f"data/processed/{channel}/summaries_quarter.parquet", f"{channel}_quarter_summaries"),
        (f"data/processed/{channel}/summaries_year_author.parquet", f"{channel}_year_author_summaries"),
        (f"data/processed/{channel}/summaries_quarter_author.parquet", f"{channel}_quarter_author_summaries"),
    ]:
        path = Path(path_str)
        if not path.exists():
            print(f"⚠️  {path} не найден, пропускаем")
            continue

        df = pd.read_parquet(path)
        chunks = [
            {
                "text": str(row["summary_text"]),  # ← точно summary_text
                "metadata": {
                    "type": "summary",
                    "channel": channel,
                    "doc_id": doc_id,
                    "period": row.get("period", row.get("year", ""))
                }
            }
            for _, row in df.iterrows()
        ]
        store.add_chunks(chunks)
        print(f"✅ Загружено {len(chunks)} саммари из {path.name}")

    # author_report.md
    report_path = Path(f"data/processed/{channel}/author_report.md")
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = [{"text": text[:4000], "metadata": {"type": "author_report", "channel": channel}}]
        store.add_chunks(chunks)
        print("✅ Загружен author_report.md")

    print(f"✅ Индексированы саммари для {channel}. Total: {store.count()}")


def cmd_status(args):
    store = ChromaStore()
    print(f"Total chunks: {store.count()}")
    # Превью последних 3 чанков
    results = store.collection.peek(limit=3)
    if results['metadatas']:
        for i, meta in enumerate(results['metadatas']):
            print(f"  {i+1}. {meta.get('channel', '?')} | {meta.get('type', 'chunk')}")


if __name__ == "__main__":
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)
