# Масштабирование и производительность

Документация по настройке системы для работы с множеством пользователей.

> ✅ **Фаза 1 реализована:** PostgreSQL, Rate Limiting и Кэширование готовы к использованию!

## Фаза 1: Критичные улучшения

### 1. PostgreSQL вместо SQLite

**Проблема:** SQLite не поддерживает конкурентные записи и ограничивает пропускную способность.

**Решение:** Миграция на PostgreSQL с пулом соединений.

#### Настройка PostgreSQL

1. **Установите PostgreSQL:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   
   # macOS
   brew install postgresql
   
   # Windows
   # Скачайте с https://www.postgresql.org/download/windows/
   ```

2. **Создайте базу данных:**
   ```sql
   CREATE DATABASE rag_bot;
   CREATE USER rag_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE rag_bot TO rag_user;
   ```

3. **Настройте .env файл:**
   ```env
   DB_TYPE=postgresql
   DATABASE_URL=postgresql://rag_user:your_password@localhost:5432/rag_bot
   ```

4. **Установите зависимости:**
   ```bash
   pip install psycopg2-binary
   ```

#### Обратная совместимость

Если `DB_TYPE` не задан или `DATABASE_URL` пуст, система автоматически использует SQLite. Это позволяет:
- Продолжать работу без изменений для существующих установок
- Постепенно мигрировать на PostgreSQL

### 2. Rate Limiting

**Проблема:** Пользователи могут отправлять неограниченное количество запросов, перегружая систему.

**Решение:** Rate limiting на основе sliding window.

#### Настройка

В `.env` файле:
```env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=10      # Количество запросов
RATE_LIMIT_WINDOW=60        # Окно времени в секундах
```

**По умолчанию:** 10 запросов в минуту на пользователя.

#### Как это работает

- Каждый запрос пользователя проверяется перед обработкой
- Если лимит превышен, пользователь получает сообщение об ошибке
- Счетчик сбрасывается автоматически через указанное время

#### Отключение

Если нужно отключить rate limiting:
```env
RATE_LIMIT_ENABLED=false
```

### 3. Кэширование ответов

**Проблема:** Одинаковые вопросы обрабатываются заново, тратя ресурсы LLM.

**Решение:** Кэширование ответов RAG с поддержкой memory и Redis.

#### Настройка Memory Cache (по умолчанию)

Работает из коробки, не требует дополнительной настройки:
```env
CACHE_ENABLED=true
CACHE_BACKEND=memory
CACHE_TTL=3600  # 1 час
```

**Ограничения:**
- Кэш доступен только в рамках одного процесса
- При перезапуске бота кэш очищается
- Не подходит для множественных инстансов

#### Настройка Redis Cache

Для распределенных систем или множественных инстансов:

1. **Установите Redis:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS
   brew install redis
   
   # Windows
   # Скачайте с https://redis.io/download
   ```

2. **Запустите Redis:**
   ```bash
   redis-server
   ```

3. **Настройте .env:**
   ```env
   CACHE_ENABLED=true
   CACHE_BACKEND=redis
   REDIS_URL=redis://localhost:6379/0
   CACHE_TTL=3600
   ```

4. **Установите зависимости:**
   ```bash
   pip install redis
   ```

#### Как это работает

- Ключ кэша создается из вопроса и каналов (MD5 хеш)
- Ответы сохраняются с TTL (время жизни)
- При повторном запросе ответ возвращается из кэша мгновенно

#### Очистка кэша

Кэш очищается автоматически по истечении TTL. Для ручной очистки:
```python
from utils.cache import get_cache
cache = get_cache()
cache.clear()  # Очистить весь кэш
```

## Ожидаемые улучшения

### До оптимизаций
- **Пропускная способность:** ~10-20 одновременных пользователей
- **Время ответа:** 2-5 секунд (без кэша)
- **Ограничения:** SQLite блокировки, нет rate limiting

### После Фазы 1
- **Пропускная способность:** ~100-200 одновременных пользователей
- **Время ответа:** 0.1-0.5 секунд (с кэшем), 2-5 секунд (без кэша)
- **Улучшения:**
  - PostgreSQL поддерживает конкурентные запросы
  - Rate limiting защищает от перегрузки
  - Кэш ускоряет повторные запросы в 10-50 раз

## Мониторинг

### Логи

Система логирует важные события:
- `[RATE_LIMIT]` - превышение лимита запросов
- `[CACHE]` - попадания и промахи кэша
- `[DB]` - тип используемой БД

### Метрики производительности

Используйте `utils/metrics.py` для отслеживания:
```python
from utils.metrics import PerformanceMetrics

metrics = PerformanceMetrics()
metrics.start("rag_query")
# ... выполнение запроса ...
metrics.end("rag_query")
metrics.log_summary()
```

## Миграция данных

### Из SQLite в PostgreSQL

1. **Экспорт данных из SQLite:**
   ```bash
   sqlite3 data/bot.db .dump > backup.sql
   ```

2. **Адаптация SQL для PostgreSQL:**
   - Замените `INTEGER PRIMARY KEY AUTOINCREMENT` на `SERIAL PRIMARY KEY`
   - Убедитесь, что все типы данных совместимы

3. **Импорт в PostgreSQL:**
   ```bash
   psql -U rag_user -d rag_bot < backup.sql
   ```

4. **Обновите .env:**
   ```env
   DB_TYPE=postgresql
   DATABASE_URL=postgresql://rag_user:password@localhost:5432/rag_bot
   ```

## Рекомендации

1. **Для MVP (до 50 пользователей):**
   - SQLite + Memory Cache + Rate Limiting
   - Простая настройка, работает из коробки

2. **Для продакшена (100+ пользователей):**
   - PostgreSQL + Redis Cache + Rate Limiting
   - Требует настройки инфраструктуры

3. **Для масштабирования (1000+ пользователей):**
   - См. Фазу 2 и Фазу 3 в основном плане развития

## Troubleshooting

### PostgreSQL не подключается

1. Проверьте, что PostgreSQL запущен:
   ```bash
   sudo systemctl status postgresql
   ```

2. Проверьте права доступа:
   ```sql
   GRANT ALL PRIVILEGES ON DATABASE rag_bot TO rag_user;
   ```

3. Проверьте URL в .env:
   ```env
   DATABASE_URL=postgresql://user:password@host:port/dbname
   ```

### Redis не подключается

1. Проверьте, что Redis запущен:
   ```bash
   redis-cli ping
   # Должен вернуть: PONG
   ```

2. Проверьте URL в .env:
   ```env
   REDIS_URL=redis://localhost:6379/0
   ```

3. Если Redis недоступен, система автоматически переключится на Memory Cache

### Rate limiting слишком строгий

Увеличьте лимиты в .env:
```env
RATE_LIMIT_REQUESTS=20
RATE_LIMIT_WINDOW=60
```

### Кэш не работает

1. Проверьте, что кэш включен:
   ```env
   CACHE_ENABLED=true
   ```

2. Проверьте логи на наличие ошибок `[CACHE]`

3. Для Redis проверьте подключение (см. выше)

