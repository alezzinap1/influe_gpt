#!/usr/bin/env python3
"""
Скрипт для массовой переиндексации всех каналов.
Переиндексирует только ChromaDB, не удаляет данные из SQLite БД и файлы.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tgbot.bot_db import _get_conn
from tg_channels.ingest import reindex_tg_channel
from config_telegram import normalize_channel_name, RAW_DIR

def reindex_all_channels():
    """Переиндексирует все каналы из БД."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT tg_username FROM channels")
    channels = [row["tg_username"] for row in cur.fetchall()]
    conn.close()
    
    print(f"Найдено каналов для переиндексации: {len(channels)}")
    
    if not channels:
        print("Нет каналов для переиндексации")
        return
    
    success_count = 0
    error_count = 0
    
    for idx, channel in enumerate(channels, 1):
        print(f"\n[{idx}/{len(channels)}] Переиндексация канала: {channel}")
        try:
            reindex_tg_channel(channel, max_chars=600, min_chars=50)
            print(f"✅ {channel} переиндексирован")
            success_count += 1
        except Exception as e:
            print(f"❌ Ошибка при переиндексации {channel}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Переиндексация завершена:")
    print(f"   Успешно: {success_count}")
    print(f"   Ошибок: {error_count}")
    print(f"   Всего: {len(channels)}")


def reindex_all_from_files():
    """Переиндексирует все каналы из parquet файлов."""
    parquet_files = list(RAW_DIR.glob("*.parquet"))
    
    print(f"Найдено parquet файлов: {len(parquet_files)}")
    
    if not parquet_files:
        print("Нет parquet файлов для переиндексации")
        return
    
    success_count = 0
    error_count = 0
    
    for idx, parquet_file in enumerate(parquet_files, 1):
        # Имя канала = имя файла без расширения
        channel = parquet_file.stem
        print(f"\n[{idx}/{len(parquet_files)}] Переиндексация канала: {channel}")
        try:
            reindex_tg_channel(channel, max_chars=600, min_chars=50)
            print(f"✅ {channel} переиндексирован")
            success_count += 1
        except Exception as e:
            print(f"❌ Ошибка при переиндексации {channel}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Переиндексация завершена:")
    print(f"   Успешно: {success_count}")
    print(f"   Ошибок: {error_count}")
    print(f"   Всего: {len(parquet_files)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--from-files":
        reindex_all_from_files()
    else:
        reindex_all_channels()

