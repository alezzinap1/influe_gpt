"""Rate limiting для пользователей."""
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """Простой rate limiter на основе sliding window."""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Args:
            max_requests: Максимальное количество запросов
            window_seconds: Окно времени в секундах
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[int, List[float]] = defaultdict(list)
        self._lock = Lock()
    
    def is_allowed(self, user_id: int) -> bool:
        """
        Проверяет, разрешен ли запрос для пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            True если запрос разрешен, False если превышен лимит
        """
        if not self.max_requests or not self.window_seconds:
            return True
        
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self._lock:
            # Очищаем старые запросы
            user_requests = self._requests[user_id]
            user_requests[:] = [req_time for req_time in user_requests if req_time > cutoff]
            
            # Проверяем лимит
            if len(user_requests) >= self.max_requests:
                logger.warning(f"[RATE_LIMIT] Пользователь {user_id} превысил лимит: {len(user_requests)}/{self.max_requests}")
                return False
            
            # Добавляем текущий запрос
            user_requests.append(now)
            return True
    
    def get_remaining(self, user_id: int) -> int:
        """
        Возвращает количество оставшихся запросов для пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Количество оставшихся запросов
        """
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self._lock:
            user_requests = self._requests[user_id]
            user_requests[:] = [req_time for req_time in user_requests if req_time > cutoff]
            return max(0, self.max_requests - len(user_requests))
    
    def reset(self, user_id: Optional[int] = None):
        """
        Сбрасывает счетчик запросов для пользователя или всех пользователей.
        
        Args:
            user_id: ID пользователя (если None, сбрасывает всех)
        """
        with self._lock:
            if user_id is None:
                self._requests.clear()
            else:
                self._requests.pop(user_id, None)


# Глобальный экземпляр rate limiter
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(max_requests: int = 10, window_seconds: int = 60) -> RateLimiter:
    """Получает глобальный экземпляр rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_requests, window_seconds)
    return _rate_limiter



