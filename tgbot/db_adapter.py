"""Адаптер для работы с БД (PostgreSQL или SQLite) через SQLAlchemy."""
import logging
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Определяем, какую БД использовать
DB_TYPE: Optional[str] = None
DB_URL: Optional[str] = None

try:
    from settings import DATABASE_URL, DB_TYPE as _DB_TYPE
    DB_TYPE = _DB_TYPE
    DB_URL = DATABASE_URL
    if not DB_URL and DB_TYPE == "postgresql":
        # Если PostgreSQL выбран, но URL не задан, используем дефолтный
        DB_URL = "postgresql://localhost/rag_bot"
except (ImportError, AttributeError):
    # Fallback на SQLite если настройки не заданы
    DB_TYPE = "sqlite"
    db_path = Path("data/bot.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    DB_URL = f"sqlite:///{db_path.absolute()}"


class DatabaseAdapter:
    """Адаптер для работы с БД через SQLAlchemy."""
    
    def __init__(self):
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.engine import Engine
            from sqlalchemy.pool import QueuePool
            
            self.db_type = DB_TYPE
            self.db_url = DB_URL
            self.text = text  # Инициализируем text до использования
            
            # Создаем engine с пулом соединений
            if self.db_type == "postgresql":
                # Для PostgreSQL используем пул соединений
                self.engine = create_engine(
                    self.db_url,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,  # Проверяем соединения перед использованием
                    echo=False,
                    connect_args={"connect_timeout": 2}  # Короткий таймаут для быстрой проверки (psycopg2)
                )
                # Проверяем доступность PostgreSQL при инициализации
                try:
                    with self.engine.connect() as conn:
                        conn.execute(self.text("SELECT 1"))
                    logger.info(f"[DB] Используется PostgreSQL с пулом соединений: {self.db_url}")
                except Exception as e:
                    error_msg = str(e)
                    if "Connection refused" in error_msg or "could not connect" in error_msg.lower() or "timeout" in error_msg.lower():
                        logger.info(f"[DB] PostgreSQL недоступен, отключаем адаптер и используем SQLite fallback")
                        # Отключаем engine, чтобы использовать SQLite fallback
                        self.engine = None
                        self.db_type = "sqlite_fallback"
                        self._db_path = Path("data/bot.db")
                        self._db_path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        logger.warning(f"[DB] Ошибка PostgreSQL: {e}, будет использован fallback на SQLite при ошибках")
            else:
                # SQLite - без пула, но с check_same_thread=False
                self.engine = create_engine(
                    self.db_url,
                    connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {},
                    echo=False
                )
                logger.info(f"[DB] Используется SQLite: {self.db_url}")
        except ImportError:
            logger.error("[DB] SQLAlchemy не установлен, используем fallback на sqlite3")
            self.engine = None
            self.db_type = "sqlite_fallback"
            self._db_path = Path("data/bot.db")
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Контекстный менеджер для получения соединения с БД."""
        if self.engine is None:
            # Fallback на sqlite3
            import sqlite3
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
        else:
            conn = self.engine.connect()
            try:
                yield conn
            finally:
                conn.close()
    
    def execute(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Выполняет SELECT запрос и возвращает результаты."""
        if params is None:
            params = {}
        
        with self.get_connection() as conn:
            if self.engine is None:
                # Fallback на sqlite3
                cur = conn.cursor()
                # Заменяем именованные параметры на позиционные для SQLite
                sqlite_query = query
                sqlite_params = []
                for key, value in params.items():
                    sqlite_query = sqlite_query.replace(f":{key}", "?")
                    sqlite_params.append(value)
                cur.execute(sqlite_query, sqlite_params)
                rows = cur.fetchall()
                return [dict(row) for row in rows]
            else:
                # SQLAlchemy
                result = conn.execute(self.text(query), params)
                rows = result.fetchall()
                # Преобразуем Row объекты в dict
                return [dict(row._mapping) for row in rows]
    
    def execute_one(self, query: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Выполняет SELECT запрос и возвращает один результат."""
        results = self.execute(query, params or {})
        return results[0] if results else None
    
    def execute_write(self, query: str, params: Dict[str, Any] = None) -> int:
        """Выполняет INSERT/UPDATE/DELETE запрос и возвращает количество затронутых строк."""
        if params is None:
            params = {}
        
        with self.get_connection() as conn:
            if self.engine is None:
                # Fallback на sqlite3
                cur = conn.cursor()
                sqlite_query = query
                param_order = []
                for key, value in params.items():
                    if f":{key}" in sqlite_query:
                        param_order.append(value)
                        sqlite_query = sqlite_query.replace(f":{key}", "?", 1)
                cur.execute(sqlite_query, tuple(param_order))
                conn.commit()
                return cur.rowcount
            else:
                # SQLAlchemy
                result = conn.execute(self.text(query), params)
                conn.commit()
                return result.rowcount
    
    def execute_script(self, script: str):
        """Выполняет SQL скрипт (для создания таблиц)."""
        # Адаптируем SQL для PostgreSQL если нужно
        if self.db_type == "postgresql":
            script = self._adapt_sql_for_postgres(script)
        
        with self.get_connection() as conn:
            if self.engine is None:
                # Fallback на sqlite3
                conn.executescript(script)
                conn.commit()
            else:
                # SQLAlchemy - выполняем каждую команду отдельно
                for statement in script.split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(self.text(statement))
                conn.commit()
    
    def _adapt_sql_for_postgres(self, script: str) -> str:
        """Адаптирует SQL скрипт для PostgreSQL."""
        # Заменяем SQLite-специфичные конструкции на PostgreSQL
        script = script.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        script = script.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
        # TEXT остается TEXT в PostgreSQL, это нормально
        # INTEGER остается INTEGER в PostgreSQL
        return script
    
    def lastrowid(self, table_name: str = None) -> Optional[int]:
        """Возвращает ID последней вставленной строки."""
        if self.db_type == "sqlite" or self.engine is None:
            # Для SQLite используем lastrowid
            with self.get_connection() as conn:
                if self.engine is None:
                    return conn.lastrowid
                else:
                    # Для SQLAlchemy с SQLite
                    result = conn.execute(self.text("SELECT last_insert_rowid()"))
                    row = result.fetchone()
                    return row[0] if row else None
        else:
            # Для PostgreSQL используем RETURNING или sequence
            if table_name:
                query = f"SELECT currval(pg_get_serial_sequence('{table_name}', 'id'))"
                result = self.execute_one(query)
                return result["currval"] if result else None
            return None


# Глобальный экземпляр адаптера
_db_adapter: Optional[DatabaseAdapter] = None


def get_db_adapter() -> DatabaseAdapter:
    """Получает глобальный экземпляр адаптера БД."""
    global _db_adapter
    if _db_adapter is None:
        _db_adapter = DatabaseAdapter()
    return _db_adapter
