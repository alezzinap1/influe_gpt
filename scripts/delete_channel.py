#!/usr/bin/env python3
"""Удаление канала через терминал"""
import sys
sys.path.insert(0, '.')

from tgbot.bot_tasks import delete_channel_completely
from config_telegram import normalize_channel_name

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python delete_channel.py <channel_name>")
        print("Пример: python delete_channel.py DoCryptoBred")
        print("Или: python delete_channel.py https://t.me/DoCryptoBred")
        sys.exit(1)
    
    channel_input = sys.argv[1]
    channel = normalize_channel_name(channel_input)
    
    if not channel:
        print(f"❌ Некорректное имя канала: {channel_input}")
        sys.exit(1)
    
    print(f"🗑️  Удаление канала @{channel}...")
    result = delete_channel_completely(channel)
    
    if result.get("success"):
        stats = result.get("chromadb_stats", {})
        print(f"✅ {result.get('message', 'Канал удален')}")
        print(f"   Чанков удалено: {stats.get('chunks_deleted', 0)}")
        print(f"   Саммари удалено: {stats.get('summaries_deleted', 0)}")
        print(f"   Файлы: {'✅' if result.get('files_deleted') else '❌ (не найдены)'}")
    else:
        print(f"❌ {result.get('message', 'Ошибка при удалении')}")
        if result.get("errors"):
            print(f"   Ошибки: {', '.join(result['errors'])}")
        sys.exit(1)