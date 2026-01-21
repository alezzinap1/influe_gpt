#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для синхронизации статуса каналов между SQLite и ChromaDB.
Использование:
    python sync_channel_status.py                    # синхронизировать все каналы
    python sync_channel_status.py bluedepp           # синхронизировать конкретный канал
    python sync_channel_status.py --reset           # сбросить все флаги
"""

import sys
import os
# Устанавливаем UTF-8 для вывода в Windows
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
from tgbot.bot_db import (
    sync_channel_status_with_chromadb,
    sync_all_channels_status,
    reset_all_channels_status,
    get_channel,
)

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--reset":
            result = reset_all_channels_status()
            print(result["message"])
            return
        
        channel = sys.argv[1].lstrip("@")
        print(f"Синхронизация канала: {channel}")
        result = sync_channel_status_with_chromadb(channel)
        
        if result.get("error"):
            print(f"Ошибка: {result['error']}")
            return
        
        print(f"\nРезультат синхронизации:")
        print(f"  Канал: @{result['channel']}")
        print(f"  Чанков в ChromaDB: {result.get('chunk_count', 0)}")
        print(f"  Есть чанки: {'YES' if result.get('has_chunks_in_chromadb') else 'NO'}")
        print(f"  Есть саммари: {'YES' if result.get('has_summaries_in_chromadb') else 'NO'}")
        print(f"  Статус в SQLite (до): has_chunks={result.get('has_chunks_in_sqlite')}")
        
        # Проверяем обновленный статус
        row = get_channel(channel)
        if row:
            print(f"  Статус в SQLite (после): has_chunks={bool(row['has_chunks'])}, has_summaries={bool(row['has_summaries'])}")
    else:
        print("Синхронизация всех каналов...")
        results = sync_all_channels_status()
        
        print(f"\nСинхронизировано каналов: {len(results)}")
        for result in results:
            if result.get("error"):
                print(f"  ERROR @{result.get('channel', 'unknown')}: {result['error']}")
            else:
                status = "OK" if result.get("has_chunks_in_chromadb") else "WAIT"
                print(f"  [{status}] @{result.get('channel', 'unknown')}: {result.get('chunk_count', 0)} chunks")

if __name__ == "__main__":
    main()

