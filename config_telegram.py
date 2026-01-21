# config_telegram.py
from pathlib import Path

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
    
    # Убираем пробелы
    channel = channel.strip()
    
    # Убираем https://t.me/ или http://t.me/
    if channel.startswith("https://t.me/"):
        channel = channel[len("https://t.me/"):]
    elif channel.startswith("http://t.me/"):
        channel = channel[len("http://t.me/"):]
    
    # Убираем @ в начале
    channel = channel.lstrip("@")
    
    # Убираем пробелы еще раз
    channel = channel.strip()
    
    return channel


def raw_parquet_path(channel: str) -> Path:
    """Возвращает путь к parquet файлу с сырыми данными канала."""
    normalized = normalize_channel_name(channel)
    # Заменяем / на _ для безопасности имени файла
    safe = normalized.replace("/", "_")
    return RAW_DIR / f"{safe}.parquet"


TG_API_ID: int = 26949227
TG_API_HASH: str = "ce1b9213cd2d0112eb4ab93bcccf25d3"
TG_SESSION_NAME: str = "my_session"
