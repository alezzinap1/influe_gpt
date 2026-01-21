#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скрипт для нормализации имен каналов в БД и удаления дубликатов."""
import sys
import io
sys.path.insert(0, '.')

# Устанавливаем UTF-8 для вывода в Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from config_telegram import normalize_channel_name
from tgbot.bot_db import _get_conn
import sqlite3

def fix_channels_db():
    """Нормализует все имена каналов в БД и удаляет дубликаты."""
    conn = _get_conn()
    cur = conn.cursor()
    
    # Получаем все каналы
    cur.execute("SELECT id, tg_username FROM channels")
    channels = cur.fetchall()
    
    print(f"Найдено каналов: {len(channels)}")
    
    # Сначала находим дубликаты и обрабатываем их
    print("\n🔍 Поиск и обработка дубликатов...")
    name_to_channels = {}
    for channel in channels:
        channel_id = channel["id"]
        old_name = channel["tg_username"]
        normalized = normalize_channel_name(old_name)
        
        if not normalized:
            print(f"⚠️  Пропускаем некорректный канал ID={channel_id}: {old_name}")
            continue
        
        if normalized not in name_to_channels:
            name_to_channels[normalized] = []
        name_to_channels[normalized].append((channel_id, old_name))
    
    # Обрабатываем дубликаты
    duplicates_removed = 0
    for normalized, channel_list in name_to_channels.items():
        if len(channel_list) > 1:
            print(f"⚠️  Дубликаты для '{normalized}': {len(channel_list)} записей")
            # Сортируем: сначала нормализованные, потом ненормализованные
            # Оставляем первую нормализованную запись, или первую если все ненормализованные
            channel_list.sort(key=lambda x: (normalize_channel_name(x[1]) != normalized, x[0]))
            keep_id, keep_name = channel_list[0]
            delete_list = channel_list[1:]
            
            print(f"   → Оставляем ID={keep_id} ('{keep_name}'), удаляем {len(delete_list)} дубликатов")
            
            # Если оставляемая запись не нормализована, обновляем её
            if normalize_channel_name(keep_name) != normalized:
                cur.execute(
                    "UPDATE channels SET tg_username = ? WHERE id = ?",
                    (normalized, keep_id)
                )
                print(f"   → Обновлено имя на нормализованное: '{normalized}'")
            
            # Удаляем связи пользователей с дубликатами и переносим их на оставляемый канал
            for delete_id, _ in delete_list:
                # Получаем все связи пользователей с удаляемым каналом
                cur.execute("SELECT user_id FROM user_channels WHERE channel_id = ?", (delete_id,))
                user_links = cur.fetchall()
                
                # Переносим связи пользователей (только если у них еще нет связи с keep_id)
                for user_link in user_links:
                    user_id = user_link["user_id"]
                    # Проверяем, есть ли уже связь с keep_id
                    cur.execute(
                        "SELECT 1 FROM user_channels WHERE user_id = ? AND channel_id = ?",
                        (user_id, keep_id)
                    )
                    if not cur.fetchone():
                        # Связи нет, переносим
                        cur.execute(
                            "UPDATE user_channels SET channel_id = ? WHERE user_id = ? AND channel_id = ?",
                            (keep_id, user_id, delete_id)
                        )
                    else:
                        # Связь уже есть, просто удаляем старую
                        cur.execute(
                            "DELETE FROM user_channels WHERE user_id = ? AND channel_id = ?",
                            (user_id, delete_id)
                        )
                
                # Удаляем дубликат канала
                cur.execute("DELETE FROM channels WHERE id = ?", (delete_id,))
                duplicates_removed += 1
            
            conn.commit()
    
    if duplicates_removed > 0:
        print(f"✅ Удалено {duplicates_removed} дубликатов")
    else:
        print("✅ Дубликатов не найдено")
    
    # Теперь обновляем оставшиеся ненормализованные имена
    print("\n🔄 Нормализация оставшихся каналов...")
    cur.execute("SELECT id, tg_username FROM channels")
    remaining_channels = cur.fetchall()
    
    updates = []
    for channel in remaining_channels:
        channel_id = channel["id"]
        old_name = channel["tg_username"]
        normalized = normalize_channel_name(old_name)
        
        if not normalized:
            continue
        
        if old_name != normalized:
            print(f"📝 Нормализация: '{old_name}' -> '{normalized}'")
            updates.append((normalized, channel_id))
        else:
            print(f"✅ Уже нормализован: {normalized}")
    
    # Обновляем имена каналов
    if updates:
        print(f"\n🔄 Обновление {len(updates)} каналов...")
        for normalized, channel_id in updates:
            cur.execute(
                "UPDATE channels SET tg_username = ? WHERE id = ?",
                (normalized, channel_id)
            )
        conn.commit()
        print("✅ Обновление завершено")
    
    # Ищем дубликаты (каналы с одинаковым нормализованным именем)
    print("\n🔍 Поиск дубликатов...")
    cur.execute("""
        SELECT tg_username, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
        FROM channels
        GROUP BY tg_username
        HAVING cnt > 1
    """)
    duplicate_groups = cur.fetchall()
    
    if duplicate_groups:
        print(f"⚠️  Найдено {len(duplicate_groups)} групп дубликатов:")
        for group in duplicate_groups:
            name = group["tg_username"]
            ids = [int(id_str) for id_str in group["ids"].split(",")]
            print(f"   Канал '{name}': {len(ids)} записей (ID: {ids})")
            
            # Оставляем первую запись, остальные удаляем
            keep_id = ids[0]
            delete_ids = ids[1:]
            
            print(f"   → Оставляем ID={keep_id}, удаляем {delete_ids}")
            
            # Удаляем связи пользователей с дубликатами
            for delete_id in delete_ids:
                cur.execute("DELETE FROM user_channels WHERE channel_id = ?", (delete_id,))
            
            # Удаляем дубликаты каналов
            placeholders = ",".join("?" * len(delete_ids))
            cur.execute(f"DELETE FROM channels WHERE id IN ({placeholders})", delete_ids)
            
            duplicates.append((name, keep_id, delete_ids))
        
        conn.commit()
        print(f"✅ Удалено {sum(len(d[2]) for d in duplicates)} дубликатов")
    else:
        print("✅ Дубликатов не найдено")
    
    conn.close()
    print("\n✅ Нормализация БД завершена!")

if __name__ == "__main__":
    try:
        fix_channels_db()
    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

