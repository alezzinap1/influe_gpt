# Решение проблем

Руководство по решению типичных проблем.

## Быстрая проверка

Запустите скрипт проверки из директории `rag_mvp`:

```bash
cd rag_mvp
python check_imports.py
```

Этот скрипт проверяет:
- ✅ Синтаксис всех Python файлов
- ✅ Импорты легковесных модулей (без torch/sentence-transformers)

## Проверка работоспособности

### 1. Проверка CLI команд

```bash
# Проверка справки
python main.py --help

# Проверка статуса индекса
python main.py status

# Проверка справки по командам Telegram
python main.py tg-sync --help
python main.py tg-query --help
python main.py tg-build-summaries --help
```

### 2. Проверка Telegram бота

```bash
# Запуск бота из директории rag_mvp
cd rag_mvp
python tgbot/bot.py

# Или как модуль:
python -m tgbot.bot
```

**Важно:** 
- Запускайте из директории `rag_mvp`, чтобы Python мог найти все модули
- Бот использует ленивые импорты, поэтому может запуститься даже при проблемах с torch
- Если torch не работает, бот запустится, но RAG-функции не будут работать
- **Убедитесь, что только один экземпляр бота запущен** - иначе будет ошибка Conflict

### 3. Проверка основных модулей

```python
# В Python REPL или скрипте
from config_telegram import TG_API_ID, TG_API_HASH
from telegram.schema import TgMessage
from telegram.ingest import ingest_tg_channel
from vectorstore.chromadb_store import ChromaStore
from rag.pipeline import RAGPipeline

# Проверка создания объектов
store = ChromaStore()
print(f"ChromaStore создан: {store.count()} chunks")

rag = RAGPipeline(backend="deepseek")
print("RAGPipeline создан успешно")
```

## Типичные проблемы

### Ошибка "No module named 'tgbot'"

Если видите эту ошибку, убедитесь, что запускаете из правильной директории:

```bash
# Правильно:
cd rag_mvp
python tgbot/bot.py

# Неправильно (из корня проекта):
python rag_mvp/tgbot/bot.py
```

### Ошибка "Conflict: terminated by other getUpdates request"

Если видите эту ошибку, значит уже запущен другой экземпляр бота. **Решение:**

```powershell
# 1. Найти процессы Python
Get-Process python | Select-Object Id, ProcessName, Path

# 2. Остановить процесс бота (если знаете ID)
Stop-Process -Id <PID>

# 3. Или остановить все Python процессы (осторожно!)
Get-Process python | Stop-Process

# 4. Подождать 5-10 секунд и запустить снова
cd rag_mvp
python tgbot/bot.py
```

**Примечание:** 
- ✅ Бот автоматически обрабатывает эту ошибку и выводит понятное сообщение с инструкциями
- ✅ Бот использует `drop_pending_updates=True` для сброса старых обновлений при запуске
- ✅ Добавлен обработчик ошибок, который корректно обрабатывает конфликты и другие ошибки

### Ошибка с torch

Если видите ошибку `OSError: [WinError 127]` или `AttributeError: module 'torch' has no attribute 'Tensor'` - **решение:**

```bash
# 1. Удалить старые версии torch (conda и pip)
conda remove libtorch pytorch-mutex -y
pip uninstall torch torchvision torchaudio -y

# 2. Очистить кэш
pip cache purge

# 3. Установить torch БЕЗ суффикса +cpu (важно для Windows!)
pip install torch torchvision torchaudio --no-cache-dir

# 4. Обновить зависимости
pip install --upgrade --force-reinstall transformers
pip install --upgrade pyarrow datasets

# 5. Проверить
python -c "import torch; print('torch работает!')"
python -c "from transformers import pipeline; print('transformers работает!')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence_transformers работает!')"
```

**Важно:** 
- ✅ Используйте обычную версию torch (`torch==2.9.1`), НЕ версию с `+cpu` суффиксом
- ✅ Python 3.13 поддерживается через pip (conda может не иметь совместимых версий)
- ✅ После установки torch обновите `transformers`, `pyarrow` и `datasets` для совместимости

### Ошибка с ChromaDB

Если ChromaDB не запускается:

```bash
# Удалить поврежденную базу (она пересоздастся автоматически)
rm -rf chroma_db
# или на Windows:
Remove-Item -Recurse -Force chroma_db
```

Если ChromaDB не запускается, проверьте права доступа к директории `chroma_db`:

```bash
# Удалите и пересоздайте директорию
rm -rf chroma_db
python main.py status  # автоматически создастся
```

### Ошибка импорта модулей

Убедитесь, что вы запускаете команды из директории `rag_mvp`:

```bash
cd rag_mvp
python main.py status
```

### Ошибка "raw parquet not found"

Если видите ошибку `raw parquet not found: data\raw\channelname.parquet`:

1. Проверьте, что файл существует:
   ```bash
   ls data/raw/*.parquet
   ```

2. Если файла нет, синхронизируйте канал:
   ```bash
   python main.py tg-sync @channelname
   ```

### Ошибка с метаданными ChromaDB

Если видите ошибку `'NoneType' object cannot be converted to 'PyBool'`:

**Решение:** Эта проблема уже исправлена в коде. Если все еще возникает:

1. Переиндексируйте канал:
   ```bash
   python main.py tg-reindex @channelname
   ```

2. Или переиндексируйте все каналы:
   ```bash
   python reindex_all.py
   ```

## Тестирование функциональности

### Тест 1: Синхронизация канала (требует Telegram API)
```bash
python main.py tg-sync @channelname
```

### Тест 2: Индексация канала
```bash
python main.py tg-ingest @channelname
```

### Тест 3: Запрос к каналу
```bash
python main.py tg-query @channelname "Ваш вопрос"
```

### Тест 4: Построение саммари
```bash
python main.py tg-build-summaries @channelname
```

### Тест 5: Индексация саммари
```bash
python main.py tg-index-summaries @channelname
```

## Чеклист перед коммитом

- [ ] `python check_imports.py` проходит без ошибок
- [ ] `python main.py --help` показывает все команды
- [ ] `python main.py status` работает
- [ ] Нет синтаксических ошибок (проверено линтером)
- [ ] Все импорты корректны

## Получение помощи

Если проблема не решена:

1. Проверьте логи бота (если запущен)
2. Запустите `python check_imports.py` для диагностики
3. Проверьте версии зависимостей: `pip list`
4. Убедитесь, что все переменные окружения настроены правильно

