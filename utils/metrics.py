"""Утилиты для измерения производительности и метрик."""
import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


@contextmanager
def track_time(operation_name: str, log_level: int = logging.INFO):
    """
    Контекстный менеджер для измерения времени выполнения операции.
    
    Args:
        operation_name: Название операции для логирования
        log_level: Уровень логирования (по умолчанию INFO)
    
    Example:
        with track_time("embedding creation"):
            emb = model.encode(text)
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.log(log_level, f"[METRICS] {operation_name} took {elapsed:.2f}s")


def timed_function(operation_name: Optional[str] = None, log_level: int = logging.INFO):
    """
    Декоратор для измерения времени выполнения функции.
    
    Args:
        operation_name: Название операции (если не указано, используется имя функции)
        log_level: Уровень логирования
    
    Example:
        @timed_function("RAG query")
        def query(question: str) -> str:
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__name__}()"
            with track_time(name, log_level):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class PerformanceMetrics:
    """Класс для сбора метрик производительности."""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, operation: str) -> None:
        """Начать измерение операции."""
        self.start_times[operation] = time.time()
    
    def end(self, operation: str) -> float:
        """
        Завершить измерение операции и вернуть время выполнения.
        
        Returns:
            Время выполнения в секундах
        """
        if operation not in self.start_times:
            logger.warning(f"[METRICS] Operation '{operation}' was not started")
            return 0.0
        
        elapsed = time.time() - self.start_times[operation]
        self.metrics[operation] = elapsed
        del self.start_times[operation]
        return elapsed
    
    def get(self, operation: str) -> Optional[float]:
        """Получить время выполнения операции."""
        return self.metrics.get(operation)
    
    def get_all(self) -> Dict[str, float]:
        """Получить все метрики."""
        return self.metrics.copy()
    
    def log_summary(self, prefix: str = "[METRICS]") -> None:
        """Логировать сводку всех метрик."""
        if not self.metrics:
            return
        
        total = sum(self.metrics.values())
        lines = [f"{prefix} Performance summary (total: {total:.2f}s):"]
        for operation, elapsed in sorted(self.metrics.items(), key=lambda x: x[1], reverse=True):
            percentage = (elapsed / total * 100) if total > 0 else 0
            lines.append(f"  {operation}: {elapsed:.2f}s ({percentage:.1f}%)")
        logger.info("\n".join(lines))
    
    def reset(self) -> None:
        """Сбросить все метрики."""
        self.metrics.clear()
        self.start_times.clear()



