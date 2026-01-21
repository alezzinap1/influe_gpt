"""Утилиты для retry с exponential backoff и graceful degradation."""
import asyncio
import time
import logging
from typing import Callable, TypeVar, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Декоратор для retry с exponential backoff.
    
    Args:
        max_retries: Максимальное количество попыток
        initial_delay: Начальная задержка в секундах
        max_delay: Максимальная задержка в секундах
        exponential_base: База для экспоненциального роста задержки
        exceptions: Кортеж исключений, при которых нужно делать retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"[RETRY] {func.__name__} попытка {attempt + 1}/{max_retries + 1} "
                            f"неудачна: {e}. Повтор через {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            f"[RETRY] {func.__name__} все попытки исчерпаны после {max_retries + 1} попыток"
                        )
                        raise
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"[RETRY] {func.__name__} попытка {attempt + 1}/{max_retries + 1} "
                            f"неудачна: {e}. Повтор через {delay:.1f}s"
                        )
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            f"[RETRY] {func.__name__} все попытки исчерпаны после {max_retries + 1} попыток"
                        )
                        raise
        
        # Определяем, является ли функция async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


class GracefulDegradation:
    """
    Контекстный менеджер для graceful degradation при недоступности LLM.
    Позволяет использовать fallback при ошибках.
    """
    
    def __init__(
        self,
        primary_func: Callable[..., T],
        fallback_func: Optional[Callable[..., T]] = None,
        fallback_message: str = "Сервис временно недоступен. Попробуйте позже.",
    ):
        self.primary_func = primary_func
        self.fallback_func = fallback_func
        self.fallback_message = fallback_message
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
    
    async def call(self, *args: Any, **kwargs: Any) -> T:
        """Вызывает primary_func, при ошибке использует fallback."""
        try:
            if asyncio.iscoroutinefunction(self.primary_func):
                return await self.primary_func(*args, **kwargs)
            else:
                return self.primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[GRACEFUL] Ошибка в primary_func: {e}, используем fallback")
            if self.fallback_func:
                try:
                    if asyncio.iscoroutinefunction(self.fallback_func):
                        return await self.fallback_func(*args, **kwargs)
                    else:
                        return self.fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"[GRACEFUL] Ошибка в fallback_func: {fallback_error}")
                    if isinstance(self.fallback_message, str):
                        return self.fallback_message
                    else:
                        raise
            else:
                if isinstance(self.fallback_message, str):
                    return self.fallback_message
                else:
                    raise

