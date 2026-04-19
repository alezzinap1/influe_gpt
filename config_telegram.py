# config_telegram.py
from pathlib import Path
import os

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_channel_name(channel: str) -> str:
    """
    Нормализует имя канала, убирая различные префиксы и форматирование.
    
    Обрабатывает:
    - https://t.me/channelname -> channelname
    - http://t.me/channelname -> channelname
    - @channelname -> channelname
    - @https://t.me/channelname -> channelname
    - channelname -> channelname
    
    Args:
        channel: Имя канала в любом формате
        
    Returns:
        Нормализованное имя канала (только username)
    """
    if not channel:
        return ""

    s = channel.strip()
    # Несколько проходов: @https://t.me/name → после снятия @ становится URL
    for _ in range(5):
        prev = s
        if s.startswith("https://t.me/"):
            s = s[len("https://t.me/") :]
        elif s.startswith("http://t.me/"):
            s = s[len("http://t.me/") :]
        s = s.lstrip("@").strip()
        if s == prev:
            break

    return s


def raw_parquet_path(channel: str) -> Path:
    """Возвращает путь к parquet файлу с сырыми данными канала."""
    normalized = normalize_channel_name(channel)
    # Заменяем / на _ для безопасности имени файла
    safe = normalized.replace("/", "_")
    return RAW_DIR / f"{safe}.parquet"


TG_API_ID: int = int(os.getenv("TG_API_ID", "0"))
TG_API_HASH: str = os.getenv("TG_API_HASH", "")
TG_SESSION_NAME: str = os.getenv("TG_SESSION_NAME", "my_session")
