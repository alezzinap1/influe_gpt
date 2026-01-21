# Быстрый старт: Масштабирование

Краткая инструкция по включению улучшений Фазы 1 для работы с множеством пользователей.

## Что было добавлено

✅ **PostgreSQL поддержка** - переход с SQLite на PostgreSQL для конкурентных запросов  
✅ **Rate Limiting** - ограничение запросов на пользователя (10/минуту по умолчанию)  
✅ **Кэширование** - ускорение повторных запросов в 10-50 раз

## Быстрая настройка

### Вариант 1: Без изменений (SQLite + Memory Cache)

**Ничего делать не нужно!** Система работает из коробки:
- SQLite для БД (до ~50 пользователей)
- Memory cache для кэширования ответов
- Rate limiting включен (10 запросов/минуту)

### Вариант 2: PostgreSQL (для 100+ пользователей)

1. **Установите PostgreSQL** и создайте БД:
   ```sql
   CREATE DATABASE rag_bot;
   CREATE USER rag_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE rag_bot TO rag_user;
   ```

2. **Добавьте в `.env`:**
   ```env
   DB_TYPE=postgresql
   DATABASE_URL=postgresql://rag_user:your_password@localhost:5432/rag_bot
   ```

3. **Установите зависимости:**
   ```bash
   pip install psycopg2-binary
   ```

4. **Перезапустите бота** - система автоматически переключится на PostgreSQL

### Вариант 3: Redis Cache (для множественных инстансов)

1. **Установите и запустите Redis:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   redis-server
   
   # macOS
   brew install redis
   brew services start redis
   ```

2. **Добавьте в `.env`:**
   ```env
   CACHE_BACKEND=redis
   REDIS_URL=redis://localhost:6379/0
   ```

3. **Установите зависимости:**
   ```bash
   pip install redis
   ```

4. **Перезапустите бота** - кэш автоматически переключится на Redis

## Настройка Rate Limiting

По умолчанию: **10 запросов в минуту** на пользователя.

Изменить в `.env`:
```env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=20      # Количество запросов
RATE_LIMIT_WINDOW=60        # Окно времени в секундах
```

Отключить:
```env
RATE_LIMIT_ENABLED=false
```

## Настройка Кэша

По умолчанию: **Memory cache с TTL 1 час**.

Изменить TTL в `.env`:
```env
CACHE_TTL=7200  # 2 часа в секундах
```

Отключить кэш:
```env
CACHE_ENABLED=false
```

## Проверка работы

После настройки проверьте логи при запуске бота:

```
[DB] Используется SQLite: sqlite:///...          # SQLite
[DB] Используется PostgreSQL с пулом соединений  # PostgreSQL
[CACHE] Подключен к Redis: redis://...          # Redis
```

## Ожидаемые улучшения

- **Пропускная способность:** с 10-20 до 100-200 пользователей
- **Время ответа:** с 2-5 сек до 0.1-0.5 сек (при попадании в кэш)
- **Защита:** rate limiting предотвращает перегрузку

## Подробная документация

См. [SCALING.md](SCALING.md) для детальной информации.



