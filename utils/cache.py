"""Кэширование ответов RAG."""
import hashlib
import json
import logging
import time
from typing import Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)


class CacheBackend:
    """Базовый класс для бэкендов кэша."""
    
    def get(self, key: str) -> Optional[str]:
        """Получить значение из кэша."""
        raise NotImplementedError
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """Сохранить значение в кэш."""
        raise NotImplementedError
    
    def delete(self, key: str):
        """Удалить значение из кэша."""
        raise NotImplementedError
    
    def clear(self):
        """Очистить весь кэш."""
        raise NotImplementedError


class MemoryCacheBackend(CacheBackend):
    """In-memory кэш (для одного процесса)."""
    
    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, tuple[str, float]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[str]:
        """Получить значение из кэша."""
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        if time.time() > expiry:
            # Истек срок действия
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """Сохранить значение в кэш."""
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)
    
    def delete(self, key: str):
        """Удалить значение из кэша."""
        self._cache.pop(key, None)
    
    def clear(self):
        """Очистить весь кэш."""
        self._cache.clear()


class RedisCacheBackend(CacheBackend):
    """Redis кэш (для распределенных систем)."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", default_ttl: int = 3600):
        try:
            import redis
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.default_ttl = default_ttl
            # Проверяем соединение
            self.redis_client.ping()
            logger.info(f"[CACHE] Подключен к Redis: {redis_url}")
        except ImportError:
            raise ImportError("redis не установлен. Установите: pip install redis")
        except Exception as e:
            logger.error(f"[CACHE] Ошибка подключения к Redis: {e}")
            raise
    
    def get(self, key: str) -> Optional[str]:
        """Получить значение из кэша."""
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.warning(f"[CACHE] Ошибка получения из Redis: {e}")
            return None
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """Сохранить значение в кэш."""
        try:
            self.redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"[CACHE] Ошибка сохранения в Redis: {e}")
    
    def delete(self, key: str):
        """Удалить значение из кэша."""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.warning(f"[CACHE] Ошибка удаления из Redis: {e}")
    
    def clear(self):
        """Очистить весь кэш."""
        try:
            self.redis_client.flushdb()
        except Exception as e:
            logger.warning(f"[CACHE] Ошибка очистки Redis: {e}")


class Cache:
    """Кэш для ответов RAG."""
    
    def __init__(self, backend: CacheBackend, enabled: bool = True, default_ttl: int = 3600):
        self.backend = backend
        self.enabled = enabled
        self.default_ttl = default_ttl
    
    def _make_key(self, question: str, channel: Optional[str] = None, channels: Optional[list] = None) -> str:
        """Создает ключ кэша из вопроса и каналов."""
        # Нормализуем вопрос (убираем лишние пробелы, приводим к нижнему регистру)
        normalized_question = " ".join(question.lower().split())
        
        # Добавляем информацию о каналах
        if channels:
            channels_str = ",".join(sorted(channels))
            cache_data = f"{normalized_question}:channels:{channels_str}"
        elif channel:
            cache_data = f"{normalized_question}:channel:{channel}"
        else:
            cache_data = f"{normalized_question}:all"
        
        # Создаем хеш для короткого ключа
        key_hash = hashlib.md5(cache_data.encode()).hexdigest()
        return f"rag:answer:{key_hash}"
    
    def get(self, question: str, channel: Optional[str] = None, channels: Optional[list] = None) -> Optional[str]:
        """Получить ответ из кэша."""
        if not self.enabled:
            return None
        
        key = self._make_key(question, channel, channels)
        cached = self.backend.get(key)
        
        if cached:
            logger.debug(f"[CACHE] Cache hit для вопроса: {question[:50]}...")
            return cached
        
        logger.debug(f"[CACHE] Cache miss для вопроса: {question[:50]}...")
        return None
    
    def set(self, question: str, answer: str, channel: Optional[str] = None, channels: Optional[list] = None, ttl: Optional[int] = None):
        """Сохранить ответ в кэш."""
        if not self.enabled:
            return
        
        key = self._make_key(question, channel, channels)
        ttl = ttl or self.default_ttl
        self.backend.set(key, answer, ttl)
        logger.debug(f"[CACHE] Ответ сохранен в кэш (TTL: {ttl}s)")
    
    def delete(self, question: str, channel: Optional[str] = None, channels: Optional[list] = None):
        """Удалить ответ из кэша."""
        if not self.enabled:
            return
        
        key = self._make_key(question, channel, channels)
        self.backend.delete(key)
    
    def clear(self):
        """Очистить весь кэш."""
        self.backend.clear()


# Глобальный экземпляр кэша
_cache: Optional[Cache] = None


def get_cache() -> Cache:
    """Получает глобальный экземпляр кэша."""
    global _cache
    if _cache is None:
        from settings import CACHE_ENABLED, CACHE_BACKEND, REDIS_URL, CACHE_TTL
        
        if CACHE_BACKEND == "redis":
            try:
                backend = RedisCacheBackend(REDIS_URL, CACHE_TTL)
            except Exception as e:
                logger.warning(f"[CACHE] Не удалось подключиться к Redis, используем memory: {e}")
                backend = MemoryCacheBackend(CACHE_TTL)
        else:
            backend = MemoryCacheBackend(CACHE_TTL)
        
        _cache = Cache(backend, CACHE_ENABLED, CACHE_TTL)
    
    return _cache



