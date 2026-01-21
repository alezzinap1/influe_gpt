# tgbot/bot_tasks.py
from types import SimpleNamespace
from typing import List
from tgbot.bot_db import (
    ensure_channel,
    mark_channel_chunks_ready,
    mark_channel_summaries_ready,
)


def run_tg_sync_and_ingest(channel: str):
    """Ленивый импорт main для избежания проблем с torch при запуске бота."""
    from main import (
        cmd_tg_sync,
        cmd_tg_ingest,
    )
    from config_telegram import normalize_channel_name
    from tgbot.bot_db import get_last_synced_msg_id
    
    clean_channel = normalize_channel_name(channel)
    if not clean_channel:
        raise ValueError(f"Некорректное имя канала: {channel}")
    
    ensure_channel(clean_channel)
    args = SimpleNamespace(channel=clean_channel)
    cmd_tg_sync(args)
    cmd_tg_ingest(args)
    
    # Получаем last_msg_id из БД после синхронизации
    last_msg_id = get_last_synced_msg_id(clean_channel)
    mark_channel_chunks_ready(clean_channel, last_msg_id)

def run_build_summaries_and_index(channel: str):
    """Ленивый импорт main для избежания проблем с torch при запуске бота."""
    from main import (
        cmd_tg_build_summaries,
        cmd_tg_index_summaries,
    )
    from config_telegram import normalize_channel_name, raw_parquet_path
    from pathlib import Path
    
    clean_channel = normalize_channel_name(channel)
    if not clean_channel:
        raise ValueError(f"Некорректное имя канала: {channel}")
    
    # Проверяем наличие файла с сырыми данными
    parquet_path = raw_parquet_path(clean_channel)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Файл с данными канала не найден: {parquet_path}\n"
            f"Сначала нужно синхронизировать канал через кнопку '🔄 Синхронизировать'."
        )

    args = SimpleNamespace(channel=clean_channel)
    cmd_tg_build_summaries(args)
    cmd_tg_index_summaries(args)
    mark_channel_summaries_ready(clean_channel)


def rag_answer(channel: str, question: str, mode: str) -> str:
    """
    Выполняет RAG запрос по одному каналу.
    
    Args:
        channel: Имя канала
        question: Вопрос пользователя
        mode: Режим поиска (не используется, оставлен для совместимости)
    
    Returns:
        Ответ от RAG системы
    """
    from tgbot.rag_helpers import execute_rag_query
    return execute_rag_query(question, channel=channel)


def rag_answer_multi(channels: List[str], question: str) -> str:
    """
    Мультиканальный запрос: поиск по нескольким каналам одновременно.
    
    Args:
        channels: Список имен каналов для поиска
        question: Вопрос пользователя
        
    Returns:
        Ответ с агрегированными результатами из всех каналов
    """
    from tgbot.rag_helpers import execute_rag_query
    return execute_rag_query(question, channels=channels)


def delete_channel_completely(channel: str) -> dict:
    """
    Полное удаление канала: SQLite БД, ChromaDB, файлы.
    
    Args:
        channel: Имя канала для удаления
        
    Returns:
        dict с результатом операции
    """
    from config_telegram import normalize_channel_name, raw_parquet_path
    from pathlib import Path
    import shutil
    
    clean_channel = normalize_channel_name(channel)
    if not clean_channel:
        return {"success": False, "message": f"Некорректное имя канала: {channel}"}
    
    results = {
        "channel": clean_channel,
        "sqlite_deleted": False,
        "chromadb_deleted": False,
        "files_deleted": False,
        "errors": []
    }
    
    # 1. Удаление из SQLite БД
    try:
        from tgbot.bot_db import delete_channel
        sqlite_result = delete_channel(clean_channel)
        results["sqlite_deleted"] = sqlite_result.get("success", False)
        
        if not sqlite_result.get("success"):
            results["errors"].append(f"SQLite: {sqlite_result.get('message', 'unknown error')}")
    except Exception as e:
        results["errors"].append(f"SQLite error: {e}")
    
    # 2. Удаление из ChromaDB
    try:
        from vectorstore.chromadb_store import ChromaStore
        store = ChromaStore()
        chroma_result = store.delete_channel_data(clean_channel)
        results["chromadb_deleted"] = chroma_result.get("success", False)
        results["chromadb_stats"] = {
            "chunks_deleted": chroma_result.get("chunks_deleted", 0),
            "summaries_deleted": chroma_result.get("summaries_deleted", 0),
            "total_deleted": chroma_result.get("total_deleted", 0)
        }
        
        if not chroma_result.get("success"):
            results["errors"].append(f"ChromaDB: {chroma_result.get('error', 'unknown error')}")
    except Exception as e:
        results["errors"].append(f"ChromaDB error: {e}")
    
    # 3. Удаление файлов
    try:
        files_deleted = []
        
        # Удаляем raw parquet
        raw_path = raw_parquet_path(clean_channel)
        raw_exists = raw_path.exists()
        
        if raw_exists:
            raw_path.unlink()
            files_deleted.append(f"raw: {raw_path.name}")
        
        # Удаляем processed директорию
        processed_dir = Path(f"data/processed/{clean_channel}")
        processed_exists = processed_dir.exists()
        
        if processed_exists:
            shutil.rmtree(processed_dir)
            files_deleted.append(f"processed: {processed_dir.name}")
        
        results["files_deleted"] = len(files_deleted) > 0
        results["files_deleted_list"] = files_deleted
    except Exception as e:
        results["errors"].append(f"Files error: {e}")
    
    # Общий результат
    # Если канал не найден в SQLite, но также отсутствует в ChromaDB и файлах - это успех (уже удален)
    sqlite_not_found = not results["sqlite_deleted"] and any("не найден" in err.lower() for err in results["errors"])
    chromadb_empty = results["chromadb_deleted"] and results.get("chromadb_stats", {}).get("total_deleted", 0) == 0
    files_empty = len(results.get("files_deleted_list", [])) == 0
    
    # Успех если:
    # 1. Все три компонента успешно удалены, ИЛИ
    # 2. Канал уже был удален (не найден в SQLite, пуст в ChromaDB, нет файлов)
    success_check = (
        (results["sqlite_deleted"] and results["chromadb_deleted"] and 
         (results["files_deleted"] or files_empty)) or
        (sqlite_not_found and chromadb_empty and files_empty)
    )
    results["success"] = success_check
    
    if results["success"]:
        if sqlite_not_found and chromadb_empty and files_empty:
            results["message"] = f"✅ Канал @{clean_channel} уже был полностью удален"
        else:
            results["message"] = f"✅ Канал @{clean_channel} полностью удален"
    else:
        results["message"] = f"❌ Канал @{clean_channel} удален частично. Ошибки: {', '.join(results['errors'])}"
    
    return results
