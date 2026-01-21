"""Settings RAG-MVP с валидацией через pydantic-settings."""
import os
from pathlib import Path
from typing import Literal

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field, field_validator
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    HAS_PYDANTIC_SETTINGS = False
    BaseSettings = None
    SettingsConfigDict = None

from exceptions import ConfigurationError


class Settings(BaseSettings):
    """Настройки приложения с валидацией."""
    
    # ChromaDB settings
    chroma_dir: Path = Field(default=Path("./chroma_db"), description="Директория для ChromaDB")
    embed_model: str = Field(default="intfloat/multilingual-e5-base", description="Модель для эмбеддингов")
    
    # RAG query settings
    top_k: int = Field(default=30, ge=1, le=100, description="Количество топ результатов")
    summ_k: int = Field(default=6, ge=1, le=20, description="Количество саммари")
    chunks_k: int = Field(default=30, ge=1, le=100, description="Количество чанков")
    max_ctx_chars: int = Field(default=20000, ge=1000, description="Максимальная длина контекста")
    max_chunk_chars: int = Field(default=800, ge=100, description="Максимальная длина чанка")
    
    # LLM settings
    llm_model: str = Field(default="qwen2.5:0.5b", description="Модель LLM для локального использования")
    ollama_base: str = Field(default="http://localhost:11434", description="URL Ollama сервера")
    
    # API ключи (опциональные, но хотя бы один нужен для работы)
    deepseek_api_key: str = Field(default="", description="DeepSeek API ключ")
    openai_api_key: str = Field(default="", description="OpenAI API ключ")
    gemini_api_key: str = Field(default="", description="Gemini API ключ")
    
    # Бэкенд для саммари
    summary_backend: Literal["deepseek", "gemini", "openai"] = Field(default="deepseek", description="Бэкенд для саммари")
    
    # Telegram sync settings
    max_msgs: int = Field(default=8000, ge=1, description="Максимум сообщений за синхронизацию")
    log_step: int = Field(default=1000, ge=1, description="Шаг логирования")
    
    # LLM parallelism
    llm_semaphore_limit: int = Field(default=8, ge=1, le=50, description="Лимит параллельных LLM запросов")
    
    # Telegram Bot settings (обязательный для бота)
    telegram_bot_token: str = Field(default="", description="Токен Telegram бота")
    
    # Database settings
    db_type: Literal["sqlite", "postgresql"] = Field(default="sqlite", description="Тип БД: sqlite или postgresql")
    database_url: str = Field(
        default="",
        description="URL БД (для PostgreSQL: postgresql://user:pass@host:port/dbname, для SQLite: sqlite:///path/to/db.db)"
    )
    
    # Rate limiting settings
    rate_limit_enabled: bool = Field(default=True, description="Включить rate limiting")
    rate_limit_requests: int = Field(default=10, ge=1, description="Количество запросов на пользователя")
    rate_limit_window: int = Field(default=60, ge=1, description="Окно времени в секундах")
    
    # Cache settings
    cache_enabled: bool = Field(default=True, description="Включить кэширование ответов")
    cache_backend: Literal["memory", "redis"] = Field(default="memory", description="Бэкенд кэша: memory или redis")
    redis_url: str = Field(default="redis://localhost:6379/0", description="URL Redis (если используется redis)")
    cache_ttl: int = Field(default=3600, ge=1, description="Время жизни кэша в секундах (1 час по умолчанию)")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="",
        extra="ignore"
    )
    
    @field_validator("chroma_dir", mode="before")
    @classmethod
    def validate_chroma_dir(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v
    
    def validate_for_bot(self):
        """Проверяет обязательные настройки для бота."""
        if not self.telegram_bot_token:
            raise ConfigurationError(
                "TELEGRAM_BOT_TOKEN обязателен для работы бота. "
                "Добавьте его в .env файл: TELEGRAM_BOT_TOKEN=your_token_here"
            )
    
    def validate_for_rag(self):
        """Проверяет, что есть хотя бы один LLM API ключ."""
        if not any([self.deepseek_api_key, self.openai_api_key, self.gemini_api_key]):
            # Для локального использования (Ollama) это не критично
            pass


# Создаем глобальный экземпляр настроек
if HAS_PYDANTIC_SETTINGS:
    try:
        _settings = Settings()
    except Exception as e:
        raise ConfigurationError(f"Ошибка загрузки настроек: {e}") from e
else:
    # Fallback на старый способ, если pydantic-settings не установлен
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.resolve() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
    load_dotenv(override=False)
    
    class _SettingsFallback:
        def __getattr__(self, name):
            return os.getenv(name.upper(), "")
    
    _settings = _SettingsFallback()

# Экспортируем переменные для обратной совместимости
CHROMA_DIR = str(_settings.chroma_dir) if hasattr(_settings, 'chroma_dir') else os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = _settings.embed_model if hasattr(_settings, 'embed_model') else os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
TOP_K = _settings.top_k if hasattr(_settings, 'top_k') else int(os.getenv("TOP_K", "30"))
SUMM_K = _settings.summ_k if hasattr(_settings, 'summ_k') else int(os.getenv("SUMM_K", "6"))
CHUNKS_K = _settings.chunks_k if hasattr(_settings, 'chunks_k') else int(os.getenv("CHUNKS_K", "30"))
MAX_CTX_CHARS = _settings.max_ctx_chars if hasattr(_settings, 'max_ctx_chars') else int(os.getenv("MAX_CTX_CHARS", "20000"))
MAX_CHUNK_CHARS = _settings.max_chunk_chars if hasattr(_settings, 'max_chunk_chars') else int(os.getenv("MAX_CHUNK_CHARS", "800"))
LLM_MODEL = _settings.llm_model if hasattr(_settings, 'llm_model') else os.getenv("LLM_MODEL", "qwen2.5:0.5b")
DEEPSEEK_API_KEY = _settings.deepseek_api_key if hasattr(_settings, 'deepseek_api_key') else os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = _settings.openai_api_key if hasattr(_settings, 'openai_api_key') else os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = _settings.gemini_api_key if hasattr(_settings, 'gemini_api_key') else os.getenv("GEMINI_API_KEY", "")
SUMMARY_BACKEND = _settings.summary_backend if hasattr(_settings, 'summary_backend') else os.getenv("SUMMARY_BACKEND", "deepseek")
OLLAMA_BASE = _settings.ollama_base if hasattr(_settings, 'ollama_base') else os.getenv("OLLAMA_BASE", "http://localhost:11434")
MAX_MSGS = _settings.max_msgs if hasattr(_settings, 'max_msgs') else int(os.getenv("MAX_MSGS", "8000"))
LOG_STEP = _settings.log_step if hasattr(_settings, 'log_step') else int(os.getenv("LOG_STEP", "1000"))
LLM_SEMAPHORE_LIMIT = _settings.llm_semaphore_limit if hasattr(_settings, 'llm_semaphore_limit') else int(os.getenv("LLM_SEMAPHORE_LIMIT", "8"))
TELEGRAM_BOT_TOKEN = _settings.telegram_bot_token if hasattr(_settings, 'telegram_bot_token') else os.getenv("TELEGRAM_BOT_TOKEN", "")

# Database settings
DB_TYPE = _settings.db_type if hasattr(_settings, 'db_type') else os.getenv("DB_TYPE", "sqlite")
DATABASE_URL = _settings.database_url if hasattr(_settings, 'database_url') else os.getenv("DATABASE_URL", "")

# Rate limiting settings
RATE_LIMIT_ENABLED = _settings.rate_limit_enabled if hasattr(_settings, 'rate_limit_enabled') else (os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true")
RATE_LIMIT_REQUESTS = _settings.rate_limit_requests if hasattr(_settings, 'rate_limit_requests') else int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW = _settings.rate_limit_window if hasattr(_settings, 'rate_limit_window') else int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Cache settings
CACHE_ENABLED = _settings.cache_enabled if hasattr(_settings, 'cache_enabled') else (os.getenv("CACHE_ENABLED", "true").lower() == "true")
CACHE_BACKEND = _settings.cache_backend if hasattr(_settings, 'cache_backend') else os.getenv("CACHE_BACKEND", "memory")
REDIS_URL = _settings.redis_url if hasattr(_settings, 'redis_url') else os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = _settings.cache_ttl if hasattr(_settings, 'cache_ttl') else int(os.getenv("CACHE_TTL", "3600"))

RAG_SYSTEM_PROMPT = (
    "Ты отвечаешь на вопросы только по данному контексту.\n"
    "Отвечай на русском, логично и по делу, без воды."
    "Если в контексте указаны даты или период сообщений (start_ts/end_ts), учитывай их и явно опирайся на время при ответе."
)

RAG_ANSWER_STYLE = (
    "Дай структурированный, технический ответ на русском языке без вводных фраз "
    "типа 'конечно', 'давайте разберём'. Сразу переходи к сути. "
    "Если у сущности есть список частей/модулей — перечисли их полным списком."
)


if __name__ == "__main__":
    print(f"✅ Settings loaded. Chroma={CHROMA_DIR}")
