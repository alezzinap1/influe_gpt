# Настройка окружения

Руководство по установке и настройке проекта.

## Установка зависимостей

```bash
cd rag_mvp
pip install -r requirements.txt
```

**Важно:** В `requirements.txt` включены:
- `python-telegram-bot>=20.0` - для Telegram бота
- `telethon>=1.34.0` - для скачивания сообщений из каналов
- `chromadb` - векторная база данных
- `sentence-transformers` - для создания эмбеддингов
- `torch` - для ML моделей

## Настройка переменных окружения

Создайте файл `rag_mvp/.env` со следующим содержимым:

```env
# Telegram Bot Token (обязательно для работы бота)
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Telegram API credentials (для синхронизации каналов)
TG_API_ID=your_api_id
TG_API_HASH=your_api_hash

# LLM API Keys (хотя бы один для работы RAG)
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# LLM Backend для саммари (deepseek, gemini, openai)
SUMMARY_BACKEND=deepseek
```

## Получение токенов и ключей

### Telegram Bot Token

1. Откройте [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте команду `/newbot`
3. Следуйте инструкциям для создания бота
4. Скопируйте полученный токен в `TELEGRAM_BOT_TOKEN`

### Telegram API credentials

1. Перейдите на https://my.telegram.org/apps
2. Войдите с вашим номером телефона
3. Создайте приложение
4. Скопируйте `api_id` и `api_hash` в соответствующие переменные

### LLM API Keys

- **DeepSeek**: https://platform.deepseek.com/api_keys
- **OpenAI**: https://platform.openai.com/api-keys
- **Gemini**: https://makersuite.google.com/app/apikey

## Важно

- **НЕ коммитьте файл `.env` в git!** Он уже добавлен в `.gitignore`
- Токен бота теперь **обязателен** - без него бот не запустится
- Все API ключи читаются только из `.env` файла для безопасности

## Проверка установки

После установки проверьте работоспособность:

```bash
# Проверка импортов
python check_imports.py

# Проверка статуса
python main.py status

# Проверка справки
python main.py --help
```

## Установка на Windows

### Проблемы с torch

Если видите ошибку `OSError: [WinError 127]` или `AttributeError: module 'torch' has no attribute 'Tensor'`:

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

## Первый запуск

После настройки окружения:

1. **Создайте сессию Telegram:**
   ```bash
   python main.py tg-sync @channelname
   ```
   При первом запуске потребуется ввести номер телефона и код подтверждения.

2. **Проверьте работу бота:**
   ```bash
   python tgbot/bot.py
   ```

3. **Проверьте индексацию:**
   ```bash
   python main.py tg-ingest @channelname
   python main.py status
   ```

## Структура директорий

После первого запуска создадутся следующие директории:

```
rag_mvp/
├── .env                    # Переменные окружения (не в git)
├── my_session.session      # Сессия Telegram
├── chroma_db/              # База данных ChromaDB
├── data/
│   ├── raw/                # Parquet файлы с сообщениями
│   └── summaries/          # JSON файлы с саммари
└── tgbot/
    └── bot.db              # SQLite база для бота
```

