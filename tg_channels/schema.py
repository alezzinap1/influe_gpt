# telegram_schema.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TgMessage:
    channel: str
    msg_id: int
    date: datetime
    edit_date: Optional[datetime]
    sender_id: Optional[int]
    text: str
    views: Optional[int]
    forwards: Optional[int]  # количество пересылок этого сообщения
    reply_to_msg_id: Optional[int]
    has_media: bool
    # Информация о пересылке (если сообщение само является пересылкой)
    is_forwarded: bool = False  # является ли сообщение пересылкой
    forwarded_from_channel: Optional[str] = None  # откуда переслано (username канала)
    forwarded_from_msg_id: Optional[int] = None  # ID оригинального сообщения (если доступно)
