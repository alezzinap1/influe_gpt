# tgbot/bot_db.py

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Используем адаптер БД только если явно указан PostgreSQL И он доступен
_use_adapter = False
try:
    from tgbot.db_adapter import get_db_adapter
    from settings import DB_TYPE
    # Используем адаптер только если явно указан PostgreSQL И он доступен
    if DB_TYPE == "postgresql":
        # Проверяем доступность PostgreSQL при инициализации
        # Адаптер сам проверит доступность и отключит engine, если PostgreSQL недоступен
        try:
            adapter = get_db_adapter()
            # Если engine не None и db_type все еще postgresql, значит PostgreSQL доступен
            if adapter.engine is not None and adapter.db_type == "postgresql":
                _use_adapter = True
                logger.info("[BOT_DB] PostgreSQL доступен, используется адаптер БД")
            else:
                # Адаптер уже переключился на SQLite fallback
                _use_adapter = False
                logger.info("[BOT_DB] PostgreSQL недоступен, адаптер отключен. Используется SQLite.")
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "could not connect" in error_msg.lower():
                logger.info("[BOT_DB] PostgreSQL недоступен при инициализации, отключаем адаптер. Используется SQLite.")
            else:
                logger.warning(f"[BOT_DB] Ошибка проверки PostgreSQL: {e}, отключаем адаптер. Используется SQLite.")
            _use_adapter = False
except (ImportError, AttributeError) as e:
    # По умолчанию используем SQLite
    logger.debug(f"[BOT_DB] Не удалось импортировать адаптер: {e}, используем SQLite")
    pass

DB_PATH = Path("data/bot.db")


def _get_conn():
    """Получает соединение с БД (fallback для обратной совместимости)."""
    # Всегда возвращаем SQLite соединение для обратной совместимости
    # Адаптер используется напрямую в функциях, которые его поддерживают
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Инициализирует БД, создавая таблицы если их нет."""
    if _use_adapter:
        try:
            adapter = get_db_adapter()
            adapter.execute_script(
                """
                CREATE TABLE IF NOT EXISTS channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tg_username TEXT UNIQUE NOT NULL,
                    first_synced_at TEXT,
                    last_updated_at TEXT,
                    has_chunks INTEGER DEFAULT 0,
                    has_summaries INTEGER DEFAULT 0,
                    summaries_indexed INTEGER DEFAULT 0,
                    last_summary_built_at TEXT,
                    last_synced_msg_id INTEGER DEFAULT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tg_id INTEGER UNIQUE NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_channels (
                    user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    last_mode TEXT DEFAULT 'chunks',
                    PRIMARY KEY (user_id, channel_id),
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    FOREIGN KEY(channel_id) REFERENCES channels(id)
                );
                """
            )
            
            # Миграция: добавляем столбец last_synced_msg_id если его нет
            try:
                adapter.execute("SELECT last_synced_msg_id FROM channels LIMIT 1", {})
                logger.debug("Столбец last_synced_msg_id уже существует")
            except Exception:
                try:
                    adapter.execute_write("ALTER TABLE channels ADD COLUMN last_synced_msg_id INTEGER DEFAULT NULL", {})
                    logger.info("Добавлен столбец last_synced_msg_id в таблицу channels")
                except Exception as e:
                    logger.warning(f"Не удалось добавить столбец last_synced_msg_id: {e}")
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "could not connect" in error_msg.lower():
                logger.info(f"[BOT_DB] PostgreSQL недоступен, используется SQLite (это нормально, если PostgreSQL не настроен)")
            else:
                logger.warning(f"[BOT_DB] Ошибка при использовании адаптера: {e}, fallback на SQLite")
    
    # Всегда выполняем инициализацию SQLite для обратной совместимости
    conn = _get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_username TEXT UNIQUE NOT NULL,
            first_synced_at TEXT,
            last_updated_at TEXT,
            has_chunks INTEGER DEFAULT 0,
            has_summaries INTEGER DEFAULT 0,
            summaries_indexed INTEGER DEFAULT 0,
            last_summary_built_at TEXT,
            last_synced_msg_id INTEGER DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_id INTEGER UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_channels (
            user_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL,
            last_mode TEXT DEFAULT 'chunks',
            PRIMARY KEY (user_id, channel_id),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(channel_id) REFERENCES channels(id)
        );
        """
    )
    
    # Миграция: добавляем столбец last_synced_msg_id если его нет
    try:
        cur.execute("SELECT last_synced_msg_id FROM channels LIMIT 1")
        logger.debug("Столбец last_synced_msg_id уже существует")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE channels ADD COLUMN last_synced_msg_id INTEGER DEFAULT NULL")
            conn.commit()
            logger.info("Добавлен столбец last_synced_msg_id в таблицу channels")
        except sqlite3.OperationalError as e:
            logger.warning(f"Не удалось добавить столбец last_synced_msg_id: {e}")
            conn.rollback()
    
    conn.commit()
    conn.close()


def get_or_create_user(tg_id: int) -> int:
    """Получает или создает пользователя в БД."""
    if _use_adapter:
        try:
            adapter = get_db_adapter()
            row = adapter.execute_one("SELECT id FROM users WHERE tg_id = :tg_id", {"tg_id": tg_id})
            if row:
                return row["id"]
            else:
                adapter.execute_write("INSERT INTO users (tg_id) VALUES (:tg_id)", {"tg_id": tg_id})
                # Получаем ID вставленной строки
                row = adapter.execute_one("SELECT id FROM users WHERE tg_id = :tg_id", {"tg_id": tg_id})
                return row["id"] if row else 0
        except Exception as e:
            error_msg = str(e)
            if "Connection refused" in error_msg or "could not connect" in error_msg.lower():
                logger.info(f"[BOT_DB] PostgreSQL недоступен в get_or_create_user, fallback на SQLite")
            else:
                logger.warning(f"[BOT_DB] Ошибка адаптера в get_or_create_user: {e}, fallback на SQLite")
    
    # Fallback на SQLite
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE tg_id = ?", (tg_id,))
    row = cur.fetchone()
    if row:
        uid = row["id"]
    else:
        cur.execute("INSERT INTO users (tg_id) VALUES (?)", (tg_id,))
        conn.commit()
        uid = cur.lastrowid
    conn.close()
    return uid


def get_channel(tg_username: str):
    """Получает канал из БД по username."""
    if _use_adapter:
        try:
            adapter = get_db_adapter()
            row = adapter.execute_one("SELECT * FROM channels WHERE tg_username = :username", {"username": tg_username})
            return row
        except Exception as e:
            logger.warning(f"[BOT_DB] Ошибка адаптера в get_channel: {e}, fallback на SQLite")
    
    # Fallback на SQLite
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM channels WHERE tg_username = ?", (tg_username,))
    row = cur.fetchone()
    conn.close()
    return row


def ensure_channel(tg_username: str):
    """Создает канал в БД, нормализуя имя канала."""
    from config_telegram import normalize_channel_name
    
    # Нормализуем имя канала перед сохранением
    normalized = normalize_channel_name(tg_username)
    if not normalized:
        raise ValueError(f"Некорректное имя канала: {tg_username}")
    
    conn = _get_conn()
    cur = conn.cursor()
    
    # Проверяем, есть ли канал с нормализованным именем
    cur.execute("SELECT * FROM channels WHERE tg_username = ?", (normalized,))
    row = cur.fetchone()
    if row:
        conn.close()
        return row
    
    # Если канал с ненормализованным именем существует, обновляем его
    cur.execute("SELECT * FROM channels WHERE tg_username = ?", (tg_username,))
    old_row = cur.fetchone()
    if old_row:
        # Обновляем имя на нормализованное
        cur.execute(
            "UPDATE channels SET tg_username = ? WHERE id = ?",
            (normalized, old_row["id"])
        )
        conn.commit()
        cur.execute("SELECT * FROM channels WHERE id = ?", (old_row["id"],))
        row = cur.fetchone()
        conn.close()
        return row

    # Создаем новый канал с нормализованным именем
    cur.execute(
        "INSERT INTO channels (tg_username) VALUES (?)",
        (normalized,),
    )
    conn.commit()
    channel_id = cur.lastrowid
    cur.execute("SELECT * FROM channels WHERE id = ?", (channel_id,))
    row = cur.fetchone()
    conn.close()
    return row


def get_last_synced_msg_id(tg_username: str) -> int | None:
    """Получить ID последнего синхронизированного сообщения для канала."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT last_synced_msg_id FROM channels WHERE tg_username = ?", (tg_username,))
    row = cur.fetchone()
    conn.close()
    return row["last_synced_msg_id"] if row and row["last_synced_msg_id"] is not None else None


def set_last_synced_msg_id(tg_username: str, msg_id: int):
    """Установить ID последнего синхронизированного сообщения для канала."""
    ensure_channel(tg_username)
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE channels SET last_synced_msg_id = ? WHERE tg_username = ?",
        (msg_id, tg_username),
    )
    conn.commit()
    conn.close()


def mark_channel_chunks_ready(tg_username: str, last_msg_id: int | None = None):
    # гарантируем, что канал есть
    ensure_channel(tg_username)

    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    
    if last_msg_id is not None:
        cur.execute(
            """
            UPDATE channels
            SET has_chunks = 1,
                first_synced_at = COALESCE(first_synced_at, ?),
                last_updated_at = ?,
                last_synced_msg_id = ?
            WHERE tg_username = ?
            """,
            (now, now, last_msg_id, tg_username),
        )
    else:
        cur.execute(
            """
            UPDATE channels
            SET has_chunks = 1,
                first_synced_at = COALESCE(first_synced_at, ?),
                last_updated_at = ?
            WHERE tg_username = ?
            """,
            (now, now, tg_username),
        )
    conn.commit()
    conn.close()


def mark_channel_summaries_ready(tg_username: str):
    conn = _get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        UPDATE channels
        SET has_summaries = 1,
            summaries_indexed = 1,
            last_summary_built_at = ?
        WHERE tg_username = ?
        """,
        (now, tg_username),
    )
    conn.commit()
    conn.close()


def set_user_channel_mode(user_id: int, channel_id: int, mode: str):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_channels (user_id, channel_id, last_mode)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, channel_id) DO UPDATE SET last_mode=excluded.last_mode
        """,
        (user_id, channel_id, mode),
    )
    conn.commit()
    conn.close()


def get_user_channel_mode(user_id: int, channel_id: int) -> str:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT last_mode FROM user_channels WHERE user_id = ? AND channel_id = ?",
        (user_id, channel_id),
    )
    row = cur.fetchone()
    conn.close()
    return row["last_mode"] if row else "chunks"


def get_user_channels(user_id: int):
    """Список каналов пользователя с флагами готовности."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.*
        FROM channels c
        JOIN user_channels uc ON uc.channel_id = c.id
        WHERE uc.user_id = ?
        ORDER BY c.tg_username
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def sync_channel_status_with_chromadb(channel: str) -> dict:
    """
    Синхронизирует статус канала в SQLite с реальным состоянием ChromaDB.
    Возвращает dict с информацией о синхронизации.
    """
    try:
        from vectorstore.chromadb_store import ChromaStore
        
        store = ChromaStore()
        has_chunks = store.has_chunks_for_channel(channel)
        chunk_count = store.count_chunks_for_channel(channel)
        has_summaries = store.has_summaries_for_channel(channel)
        
        conn = _get_conn()
        cur = conn.cursor()
        
        # Обновляем статус в SQLite
        now = datetime.utcnow().isoformat()
        cur.execute(
            """
            UPDATE channels 
            SET has_chunks = ?,
                has_summaries = ?,
                summaries_indexed = ?,
                last_updated_at = ?
            WHERE tg_username = ?
            """,
            (1 if has_chunks else 0, 1 if has_summaries else 0, 1 if has_summaries else 0, now, channel)
        )
        conn.commit()
        
        # Получаем обновленный статус
        cur.execute("SELECT * FROM channels WHERE tg_username = ?", (channel,))
        row = cur.fetchone()
        conn.close()
        
        return {
            "channel": channel,
            "has_chunks_in_chromadb": has_chunks,
            "chunk_count": chunk_count,
            "has_summaries_in_chromadb": has_summaries,
            "has_chunks_in_sqlite": bool(row["has_chunks"]) if row else False,
            "has_summaries_in_sqlite": bool(row["has_summaries"]) if row else False,
            "synced": True
        }
    except Exception as e:
        return {
            "channel": channel,
            "error": str(e),
            "synced": False
        }


def sync_all_channels_status():
    """Синхронизирует статус всех каналов с ChromaDB."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT tg_username FROM channels")
    channels = [row["tg_username"] for row in cur.fetchall()]
    conn.close()
    
    results = []
    for channel in channels:
        results.append(sync_channel_status_with_chromadb(channel))
    return results


def reset_all_channels_status():
    """Сбрасывает все флаги статуса каналов (используется при очистке ChromaDB)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE channels SET has_chunks = 0, has_summaries = 0, summaries_indexed = 0"
    )
    conn.commit()
    conn.close()
    return {"reset": True, "message": "Все флаги статуса каналов сброшены."}


def delete_channel(tg_username: str) -> dict:
    """
    Удаляет канал из SQLite БД и все связанные записи.
    
    Args:
        tg_username: Имя канала для удаления
        
    Returns:
        dict с результатом операции
    """
    conn = _get_conn()
    cur = conn.cursor()
    
    try:
        # Получаем channel_id
        cur.execute("SELECT id FROM channels WHERE tg_username = ?", (tg_username,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return {"success": False, "message": f"Канал @{tg_username} не найден в БД"}
        
        channel_id = row["id"]
        
        # Удаляем связи пользователей с каналом
        cur.execute("DELETE FROM user_channels WHERE channel_id = ?", (channel_id,))
        
        # Удаляем сам канал
        cur.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Канал @{tg_username} удален из SQLite БД")
        return {"success": True, "message": f"Канал @{tg_username} удален из БД"}
    except Exception as e:
        conn.rollback()
        conn.close()
        logger.error(f"Ошибка при удалении канала @{tg_username}: {e}")
        return {"success": False, "message": f"Ошибка при удалении: {e}"}