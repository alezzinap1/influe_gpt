# tgbot/bot.py

import sys
import os
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging
from typing import List

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardRemove,
)
from telegram.constants import ParseMode
from telegram import error as tg_error
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

from tgbot.bot_db import (
    init_db,
    get_or_create_user,
    get_channel,
    ensure_channel,
    set_user_channel_mode,
    get_user_channel_mode,
    get_user_channels,
    mark_channel_chunks_ready,
    mark_channel_summaries_ready,
)
from tgbot.bot_tasks import (
    run_tg_sync_and_ingest,
    run_build_summaries_and_index,
    rag_answer,
    rag_answer_multi,
)

def clean_model_text(text: str) -> str:
    """
    Убирает символы разметки из текста модели, но сохраняет HTML ссылки.
    Если в тексте есть HTML ссылки (<a href=...>), они сохраняются для использования с HTML parse_mode.
    """
    if not text:
        return text
    
    # Если в тексте есть HTML ссылки, не трогаем их
    if "<a href=" in text:
        # Только убираем markdown символы, которые могут конфликтовать с HTML
        # HTML ссылки уже экранированы в source_links.py
        for ch in ["*", "_", "`", "~"]:
            text = text.replace(ch, "")
        return text
    
    # Если HTML ссылок нет, убираем все markdown символы как раньше
    for ch in ["*", "_", "`", "~"]:
        text = text.replace(ch, "")
    return text


# Токен бота загружается из settings.py (читается из .env)
from settings import TELEGRAM_BOT_TOKEN, _settings
from exceptions import ConfigurationError

try:
    if hasattr(_settings, 'validate_for_bot'):
        _settings.validate_for_bot()
    elif not TELEGRAM_BOT_TOKEN:
        raise ConfigurationError(
            "TELEGRAM_BOT_TOKEN не задан в .env файле. "
            "Создайте файл rag_mvp/.env и добавьте строку: TELEGRAM_BOT_TOKEN=your_token_here"
        )
except ConfigurationError:
    raise
except Exception as e:
    raise ConfigurationError(f"Ошибка валидации настроек: {e}") from e

TOKEN = TELEGRAM_BOT_TOKEN

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Уменьшаем частоту логов httpx (только WARNING и выше)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Уменьшаем шум от python-telegram-bot (только важные сообщения)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)

# Фильтр для скрытия токена из логов
class TokenFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'msg'):
            # Скрываем токен в сообщениях
            if isinstance(record.msg, str):
                record.msg = record.msg.replace(TOKEN, "BOT_TOKEN_HIDDEN")
        if hasattr(record, 'args') and record.args:
            # Скрываем токен в аргументах
            record.args = tuple(
                str(arg).replace(TOKEN, "BOT_TOKEN_HIDDEN") if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True

# Применяем фильтр ко всем логгерам
for handler in logging.root.handlers:
    handler.addFilter(TokenFilter())

logger = logging.getLogger(__name__)

# FSM states
NO_CHANNEL, CHANNEL_MENU, ASKING, MULTI_CHANNEL_SELECT = range(4)


class BotMessageManager:
    """Минимум спама: всегда стараемся редактировать, а не слать новое."""

    @staticmethod
    async def thinking(update: Update, text: str = "🤔 Думаю..."):
        """Отправляет сообщение 'Думаю...' и возвращает его для последующего удаления."""
        try:
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    text, reply_markup=ReplyKeyboardRemove()
                )
                return None  # Не можем удалить отредактированное сообщение
            else:
                msg = await update.message.reply_text(
                    text, reply_markup=ReplyKeyboardRemove()
                )
                return msg  # Возвращаем для последующего удаления
        except Exception as e:
            logger.error(f"thinking() error: {e}")
            return None

    @staticmethod
    async def answer(
        update: Update,
        text: str,
        buttons: List[List[InlineKeyboardButton]] | None = None,
        parse_mode: str | None = None,
    ):
        """Отправляет ответ. Всегда отправляет новое сообщение, не редактирует существующие."""
        if buttons:
            kb = InlineKeyboardMarkup(buttons)
        else:
            kb = ReplyKeyboardRemove()

        try:
            if update.callback_query:
                # Для callback_query редактируем сообщение с кнопками
                await update.callback_query.edit_message_text(
                    text, reply_markup=kb, parse_mode=parse_mode
                )
                return None
            else:
                # Для обычных сообщений всегда отправляем новое, не редактируем
                msg = await update.message.reply_text(
                    text, reply_markup=kb, parse_mode=parse_mode
                )
                return msg
        except Exception as e:
            logger.error(f"answer() error: {e}")
            try:
                if update.effective_message:
                    msg = await update.effective_message.reply_text(
                        text, reply_markup=kb, parse_mode=parse_mode
                    )
                    return msg
            except Exception:
                pass
            return None

    @staticmethod
    async def error(update: Update, text: str):
        buttons = [[InlineKeyboardButton("🏠 Каналы", callback_data="channels")]]
        await BotMessageManager.answer(
            update, f"❌ {text}", buttons, parse_mode=None
        )


# ======================
# Handlers
# ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info(f"Received /start command from user {update.effective_user.id}")
    user = update.effective_user
    get_or_create_user(user.id)

    buttons = [
        [InlineKeyboardButton("📺 Мои каналы", callback_data="channels")],
        [InlineKeyboardButton("➕ Добавить канал", callback_data="add_channel")],
    ]
    try:
        await BotMessageManager.answer(
            update,
            "🎉 Telegram RAG бот\n\nВыберите действие:",
            buttons,
            parse_mode=None,
        )
        logger.info(f"Start message sent to user {user.id}")
    except Exception as e:
        logger.error(f"Error sending start message: {e}", exc_info=True)
    return CHANNEL_MENU


async def show_channels(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    tg_user = update.effective_user
    uid = get_or_create_user(tg_user.id)
    channels = get_user_channels(uid)

    if not channels:
        buttons = [[InlineKeyboardButton("➕ Добавить канал", callback_data="add_channel")]]
        await BotMessageManager.answer(
            update,
            "У вас пока нет каналов.\nДобавьте первый канал для анализа:",
            buttons,
            parse_mode=None,
        )
        return CHANNEL_MENU

    buttons: List[List[InlineKeyboardButton]] = []
    for row in channels[:10]:
        status = "✅" if row["has_chunks"] else "⏳"
        buttons.append(
            [
                InlineKeyboardButton(
                    f"{status} @{row['tg_username']}",
                    callback_data=f"select:{row['tg_username']}",
                )
            ]
        )

    buttons.append([InlineKeyboardButton("➕ Добавить канал", callback_data="add_channel")])
    
    # Добавляем кнопку для мультиканального поиска (если есть хотя бы 2 готовых канала)
    ready_channels = [row for row in channels if row["has_chunks"]]
    if len(ready_channels) >= 2:
        buttons.append([InlineKeyboardButton("🔍 Мультиканальный поиск", callback_data="multi_search")])

    lines = ["Ваши каналы:", ""]
    for row in channels[:5]:
        status = "✅" if row["has_chunks"] else "⏳"
        lines.append(f"{status} @{row['tg_username']}")
    if len(channels) > 5:
        lines.append(f"... и ещё {len(channels) - 5}")
    msg = await BotMessageManager.answer(update, "\n".join(lines), buttons, parse_mode=None)
    # Сохраняем ID сообщения для последующего удаления после ответа
    if msg:
        context.user_data["channels_list_msg_id"] = msg.message_id
        context.user_data["channels_list_chat_id"] = msg.chat_id
    elif update.callback_query and update.callback_query.message:
        # Если это callback_query, сохраняем ID редактированного сообщения
        context.user_data["channels_list_msg_id"] = update.callback_query.message.message_id
        context.user_data["channels_list_chat_id"] = update.callback_query.message.chat_id
    return CHANNEL_MENU


async def channel_menu_for(update: Update, channel: str) -> int:
    from config_telegram import normalize_channel_name
    # Нормализуем имя канала перед поиском
    normalized_channel = normalize_channel_name(channel)
    if not normalized_channel:
        await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
        return CHANNEL_MENU
    row = get_channel(normalized_channel)
    if not row:
        await BotMessageManager.error(update, f"Канал @{normalized_channel} не найден")
        return CHANNEL_MENU

    # Автоматическая синхронизация статуса с ChromaDB (только если канал не готов)
    # Это ускоряет работу, так как не проверяем ChromaDB для готовых каналов
    if not row or (row and not row["has_chunks"]):
        from tgbot.bot_db import sync_channel_status_with_chromadb
        sync_result = sync_channel_status_with_chromadb(normalized_channel)
        # Обновляем row после синхронизации
        row = get_channel(normalized_channel)

    if not row["has_chunks"]:
        buttons = [
            [InlineKeyboardButton("🔄 Синхронизировать", callback_data=f"sync:{normalized_channel}")],
            [InlineKeyboardButton("🏠 Каналы", callback_data="channels")],
        ]
        await BotMessageManager.answer(
            update,
            f"@{normalized_channel}\n\nКанал ещё не синхронизирован.\nНажмите, чтобы загрузить историю сообщений.",
            buttons,
            parse_mode=None,
        )
        return CHANNEL_MENU

    buttons: List[List[InlineKeyboardButton]] = [
        [InlineKeyboardButton("💬 Light (быстрый)", callback_data=f"light:{normalized_channel}")],
    ]
    if row["summaries_indexed"]:
        buttons.append(
            [InlineKeyboardButton("📚 Full (сводки)", callback_data=f"full:{normalized_channel}")]
        )
        status = "Полностью готов (есть сводки)"
    else:
        status = "Готов только по сообщениям (сводок нет)"
    buttons.append(
        [InlineKeyboardButton("🔄 Обновить сводки", callback_data=f"deep:{normalized_channel}")]
    )
    buttons.append(
        [InlineKeyboardButton("🔄 Проверить статус", callback_data=f"sync_status:{normalized_channel}")]
    )
    buttons.append(
        [InlineKeyboardButton("🗑️ Удалить канал", callback_data=f"delete_confirm:{normalized_channel}")]
    )
    buttons.append([InlineKeyboardButton("🏠 Каналы", callback_data="channels")])

    await BotMessageManager.answer(
        update,
        f"@{normalized_channel}\n\n{status}\n\nВыберите режим запроса:",
        buttons,
        parse_mode=None,
    )
    return CHANNEL_MENU


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query:
        logger.warning("handle_callback called but no callback_query in update")
        return CHANNEL_MENU
    
    logger.info(f"[BOT] Получен callback: {query.data} от пользователя {query.from_user.id} (@{query.from_user.username or 'без username'})")
    
    # Обработка ошибок для query.answer() - callback query может быть уже истекшим
    try:
        await query.answer()
    except tg_error.BadRequest as e:
        if "too old" in str(e).lower() or "timeout" in str(e).lower() or "invalid" in str(e).lower():
            logger.warning(f"Callback query истек или невалиден, пропускаем answer(): {e}")
        else:
            logger.warning(f"Ошибка BadRequest при answer() callback query: {e}")
    except Exception as e:
        logger.warning(f"Неожиданная ошибка при answer() callback query: {e}")
    
    # Продолжаем обработку даже если answer() не удался

    data = query.data.split(":", 1)
    action = data[0]
    value = data[1] if len(data) > 1 else None
    tg_user = query.from_user
    uid = get_or_create_user(tg_user.id)

    if action == "channels":
        return await show_channels(update, context)

    if action == "add_channel":
        await query.edit_message_text(
            "Введите username канала:\n@channelname или channelname",
            reply_markup=None,
        )
        return NO_CHANNEL

    if action == "select":
        channel = value
        logger.info(f"[BOT] Пользователь {tg_user.id} выбрал канал {channel}")
        # Нормализуем имя канала перед обработкой
        from config_telegram import normalize_channel_name
        normalized_channel = normalize_channel_name(channel)
        if not normalized_channel:
            await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
            return CHANNEL_MENU
        row = ensure_channel(normalized_channel)
        set_user_channel_mode(uid, row["id"], "chunks")
        context.user_data["channel"] = normalized_channel
        return await channel_menu_for(update, normalized_channel)

    if action in ("light", "full"):
        channel = value
        from config_telegram import normalize_channel_name
        normalized_channel = normalize_channel_name(channel)
        if not normalized_channel:
            await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
            return CHANNEL_MENU
        mode = "chunks" if action == "light" else "full"
        logger.info(f"[BOT] Пользователь {tg_user.id} выбрал режим {mode} для канала {normalized_channel}")
        row = ensure_channel(normalized_channel)
        set_user_channel_mode(uid, row["id"], mode)
        context.user_data["channel"] = normalized_channel
        context.user_data["mode"] = mode
        # Сохраняем сообщение для последующего удаления
        prompt_msg = await query.message.reply_text(
            "Задайте вопрос по этому каналу:",
            reply_markup=ReplyKeyboardRemove(),
        )
        context.user_data["prompt_message_id"] = prompt_msg.message_id
        context.user_data["prompt_chat_id"] = prompt_msg.chat_id
        return ASKING

    if action == "sync":
        channel = value
        from config_telegram import normalize_channel_name
        normalized_channel = normalize_channel_name(channel)
        if not normalized_channel:
            await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
            return CHANNEL_MENU
        logger.info(f"[BOT] Пользователь {tg_user.id} запустил синхронизацию канала {normalized_channel}")
        await background_sync(query, normalized_channel)
        return await channel_menu_for(update, normalized_channel)

    if action == "deep":
        channel = value
        from config_telegram import normalize_channel_name
        normalized_channel = normalize_channel_name(channel)
        if not normalized_channel:
            await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
            return CHANNEL_MENU
        logger.info(f"[BOT] Пользователь {tg_user.id} запустил построение сводок для канала {normalized_channel}")
        await background_deep(query, normalized_channel)
        return await channel_menu_for(update, normalized_channel)

    if action == "sync_status":
        channel = value or context.user_data.get("channel")
        if channel:
            from config_telegram import normalize_channel_name
            from tgbot.bot_db import sync_channel_status_with_chromadb
            normalized_channel = normalize_channel_name(channel)
            if not normalized_channel:
                await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
                return CHANNEL_MENU
            await query.edit_message_text(
                "Проверяю статус канала...",
                reply_markup=None,
            )
            result = sync_channel_status_with_chromadb(normalized_channel)
            status_text = (
                f"@{normalized_channel}\n\n"
                f"Статус синхронизирован:\n"
                f"Чанков в ChromaDB: {result.get('chunk_count', 0)}\n"
                f"Саммари в ChromaDB: {'✅' if result.get('has_summaries_in_chromadb') else '❌'}\n"
                f"Статус готовности: {'✅ Готов' if result.get('has_chunks_in_chromadb') else '⏳ Не готов'}"
            )
            await query.edit_message_text(status_text, reply_markup=None)
            await asyncio.sleep(2)
            return await channel_menu_for(update, normalized_channel)

    if action == "delete_confirm":
        channel = value
        from config_telegram import normalize_channel_name
        normalized_channel = normalize_channel_name(channel)
        if not normalized_channel:
            await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
            return CHANNEL_MENU
        buttons = [
            [InlineKeyboardButton("✅ Да, удалить", callback_data=f"delete_yes:{normalized_channel}")],
            [InlineKeyboardButton("❌ Отмена", callback_data=f"select:{normalized_channel}")],
        ]
        await query.edit_message_text(
            f"⚠️ Удалить канал @{normalized_channel}?\n\n"
            f"Будет удалено:\n"
            f"• Из базы данных\n"
            f"• Из ChromaDB (все чанки и саммари)\n"
            f"• Все файлы канала\n\n"
            f"Действие необратимо!",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return CHANNEL_MENU

    if action == "delete_yes":
        channel = value
        from config_telegram import normalize_channel_name
        normalized_channel = normalize_channel_name(channel)
        if not normalized_channel:
            await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
            return CHANNEL_MENU
        logger.info(f"[BOT] Пользователь {tg_user.id} удаляет канал {normalized_channel}")
        await query.edit_message_text(
            "🗑️ Удаление канала...\nЭто может занять несколько секунд.",
            reply_markup=None,
        )
        try:
            from tgbot.bot_tasks import delete_channel_completely
            result = await asyncio.to_thread(delete_channel_completely, normalized_channel)
            
            if result.get("success"):
                stats = result.get("chromadb_stats", {})
                message = (
                    f"✅ Канал @{channel} удален\n\n"
                    f"Удалено:\n"
                    f"• Чанков: {stats.get('chunks_deleted', 0)}\n"
                    f"• Саммари: {stats.get('summaries_deleted', 0)}\n"
                    f"• Файлы: {'✅' if result.get('files_deleted') else '❌'}"
                )
            else:
                message = f"❌ Ошибка при удалении:\n{result.get('message', 'unknown error')}"
            
            await query.edit_message_text(message, reply_markup=None)
            await asyncio.sleep(2)
            return await show_channels(update, context)
        except Exception as e:
            logger.exception(f"[BOT] Ошибка при удалении канала {channel}")
            await query.edit_message_text(f"❌ Ошибка: {e}", reply_markup=None)
            await asyncio.sleep(2)
            return await show_channels(update, context)

    if action == "multi_search":
        # Показываем список готовых каналов для выбора
        channels = get_user_channels(uid)
        ready_channels = [row for row in channels if row["has_chunks"]]
        
        if len(ready_channels) < 2:
            await query.edit_message_text(
                "Для мультиканального поиска нужно минимум 2 готовых канала.",
                reply_markup=None,
            )
            await asyncio.sleep(2)
            return await show_channels(update, context)
        
        # Инициализируем список выбранных каналов
        context.user_data["selected_channels"] = []
        
        buttons: List[List[InlineKeyboardButton]] = []
        for row in ready_channels[:10]:
            channel_name = row['tg_username']
            # Проверяем, выбран ли канал
            selected = channel_name in context.user_data.get("selected_channels", [])
            prefix = "✅" if selected else "☐"
            buttons.append([
                InlineKeyboardButton(
                    f"{prefix} @{channel_name}",
                    callback_data=f"multi_toggle:{channel_name}",
                )
            ])
        
        buttons.append([InlineKeyboardButton("🔍 Начать поиск", callback_data="multi_start")])
        buttons.append([InlineKeyboardButton("❌ Отмена", callback_data="channels")])
        
        selected_count = len(context.user_data.get("selected_channels", []))
        await query.edit_message_text(
            f"🔍 Мультиканальный поиск\n\n"
            f"Выберите каналы для поиска (минимум 2):\n"
            f"Выбрано: {selected_count}",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        # Сохраняем ID сообщения для последующего удаления после ответа
        if query.message:
            context.user_data["multi_search_msg_id"] = query.message.message_id
            context.user_data["multi_search_chat_id"] = query.message.chat_id
        return MULTI_CHANNEL_SELECT

    if action == "multi_toggle":
        channel = value
        selected_channels = context.user_data.get("selected_channels", [])
        
        if channel in selected_channels:
            selected_channels.remove(channel)
        else:
            selected_channels.append(channel)
        
        context.user_data["selected_channels"] = selected_channels
        
        # Обновляем список каналов
        channels = get_user_channels(uid)
        ready_channels = [row for row in channels if row["has_chunks"]]
        
        buttons: List[List[InlineKeyboardButton]] = []
        for row in ready_channels[:10]:
            channel_name = row['tg_username']
            selected = channel_name in selected_channels
            prefix = "✅" if selected else "☐"
            buttons.append([
                InlineKeyboardButton(
                    f"{prefix} @{channel_name}",
                    callback_data=f"multi_toggle:{channel_name}",
                )
            ])
        
        buttons.append([InlineKeyboardButton("🔍 Начать поиск", callback_data="multi_start")])
        buttons.append([InlineKeyboardButton("❌ Отмена", callback_data="channels")])
        
        selected_count = len(selected_channels)
        await query.edit_message_text(
            f"🔍 Мультиканальный поиск\n\n"
            f"Выберите каналы для поиска (минимум 2):\n"
            f"Выбрано: {selected_count}",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        # Сохраняем ID сообщения для последующего удаления после ответа
        if query.message:
            context.user_data["multi_search_msg_id"] = query.message.message_id
            context.user_data["multi_search_chat_id"] = query.message.chat_id
        return MULTI_CHANNEL_SELECT

    if action == "multi_start":
        selected_channels = context.user_data.get("selected_channels", [])
        if len(selected_channels) < 2:
            await query.edit_message_text(
                "Выберите минимум 2 канала для поиска.",
                reply_markup=None,
            )
            await asyncio.sleep(2)
            return await show_channels(update, context)
        
        context.user_data["multi_channels"] = selected_channels
        context.user_data["mode"] = "multi"
        
        prompt_msg = await query.message.reply_text(
            f"Задайте вопрос по каналам: {', '.join(f'@{ch}' for ch in selected_channels)}",
            reply_markup=ReplyKeyboardRemove(),
        )
        context.user_data["prompt_message_id"] = prompt_msg.message_id
        context.user_data["prompt_chat_id"] = prompt_msg.chat_id
        return ASKING

    return CHANNEL_MENU


async def background_sync(query, channel: str):
    logger.info(f"[BOT] Начало фоновой синхронизации канала {channel}")
    # обновляем только текст, убираем inline-клаву
    await query.edit_message_text(
        "Синхронизация истории канала...\nЭто может занять до 10 минут.",
        reply_markup=None,
    )
    
    # Минимальный прогресс-индикатор: обновляем сообщение раз в 15 секунд
    progress_task = None
    try:
        async def update_progress():
            steps = ["⏳ Загрузка сообщений...", "⏳ Индексация данных...", "⏳ Завершение..."]
            for i, step in enumerate(steps):
                await asyncio.sleep(15)  # Обновляем каждые 15 секунд
                try:
                    await query.edit_message_text(
                        f"Синхронизация истории канала...\n{step}",
                        reply_markup=None,
                    )
                except Exception:
                    pass  # Игнорируем ошибки редактирования (сообщение может быть удалено)
        
        progress_task = asyncio.create_task(update_progress())
        
        await asyncio.to_thread(run_tg_sync_and_ingest, channel)
        
        # Отменяем прогресс-индикатор
        if progress_task:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"[BOT] Синхронизация канала {channel} успешно завершена")
        await query.edit_message_text("✅ Синхронизация завершена.", reply_markup=None)
        await asyncio.sleep(1)
    except Exception as e:
        # Отменяем прогресс-индикатор при ошибке
        if progress_task:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        
        logger.exception(f"[BOT] Ошибка синхронизации канала {channel}")
        await query.edit_message_text(f"❌ Ошибка синхронизации: {e}", reply_markup=None)



async def background_deep(query, channel: str):
    logger.info(f"[BOT] Начало построения сводок для канала {channel}")
    await query.edit_message_text(
        "Построение сводок и индекса...\nВ среднем 5-15 минут.",
        reply_markup=None,
    )
    try:
        await asyncio.to_thread(run_build_summaries_and_index, channel)
        mark_channel_summaries_ready(channel)
        logger.info(f"[BOT] Построение сводок для канала {channel} успешно завершено")
        await query.edit_message_text("Сводки готовы и проиндексированы.", reply_markup=None)
        await asyncio.sleep(1)
    except FileNotFoundError as e:
        error_msg = str(e)
        if "Файл с данными канала не найден" in error_msg or "raw parquet not found" in error_msg:
            error_msg = (
                "❌ Канал не синхронизирован.\n\n"
                "Сначала нужно загрузить данные канала:\n"
                "1. Нажмите '🔄 Синхронизировать'\n"
                "2. Дождитесь завершения загрузки\n"
                "3. Затем попробуйте построить сводки снова"
            )
        logger.exception(f"[BOT] Ошибка построения сводок для канала {channel}")
        await query.edit_message_text(error_msg, reply_markup=None)
    except Exception as e:
        logger.exception(f"[BOT] Ошибка построения сводок для канала {channel}")
        await query.edit_message_text(f"Ошибка сводок: {e}", reply_markup=None)



async def handle_channel_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Пользователь вводит новый канал, когда мы в состоянии NO_CHANNEL."""
    from config_telegram import normalize_channel_name
    
    tg_user = update.effective_user
    uid = get_or_create_user(tg_user.id)

    text = (update.message.text or "").strip()
    if not text:
        logger.warning(f"[BOT] Пользователь {tg_user.id} ввел пустой канал")
        await update.message.reply_text("Введите корректный username канала.")
        return NO_CHANNEL

    channel = normalize_channel_name(text)
    if not channel:
        logger.warning(f"[BOT] Пользователь {tg_user.id} ввел некорректный канал: {text}")
        await update.message.reply_text("Введите корректный username канала (например: @channelname или https://t.me/channelname).")
        return NO_CHANNEL
    
    logger.info(f"[BOT] Пользователь {tg_user.id} добавляет канал {channel} (из ввода: {text})")
    row = ensure_channel(channel)
    set_user_channel_mode(uid, row["id"], "chunks")
    context.user_data["channel"] = channel
    msg = await update.message.reply_text(
        f"Канал @{channel} добавлен. Открываю меню...",
        reply_markup=ReplyKeyboardRemove(),
    )
    # Сохраняем ID сообщения для последующего удаления после ответа
    if msg:
        context.user_data["channel_added_msg_id"] = msg.message_id
        context.user_data["channel_added_chat_id"] = msg.chat_id
    await asyncio.sleep(0.5)
    return await channel_menu_for(update, channel)


async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    tg_user = update.effective_user
    uid = get_or_create_user(tg_user.id)
    
    # Проверка rate limit
    from settings import RATE_LIMIT_ENABLED, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
    from utils.rate_limiter import get_rate_limiter
    
    if RATE_LIMIT_ENABLED:
        rate_limiter = get_rate_limiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
        if not rate_limiter.is_allowed(tg_user.id):
            remaining = rate_limiter.get_remaining(tg_user.id)
            await BotMessageManager.error(
                update,
                f"⚠️ Превышен лимит запросов. "
                f"Максимум {RATE_LIMIT_REQUESTS} запросов в {RATE_LIMIT_WINDOW} секунд. "
                f"Попробуйте через {RATE_LIMIT_WINDOW} секунд."
            )
            return ASKING
    
    logger.info(f"[BOT] Пользователь {tg_user.id} (@{tg_user.username or 'без username'}) задает вопрос")

    # Проверяем, это мультиканальный запрос?
    multi_channels = context.user_data.get("multi_channels")
    channel = context.user_data.get("channel")
    
    if not multi_channels and not channel:
        logger.warning(f"[BOT] Канал не выбран пользователем {tg_user.id}")
        await BotMessageManager.error(update, "Сначала выберите канал.")
        return CHANNEL_MENU

    # Для мультиканального запроса пропускаем проверку одного канала
    if not multi_channels:
        from config_telegram import normalize_channel_name
        normalized_channel = normalize_channel_name(channel)
        if not normalized_channel:
            await BotMessageManager.error(update, f"Некорректное имя канала: {channel}")
            return CHANNEL_MENU
        row = get_channel(normalized_channel)
        if not row:
            logger.warning(f"[BOT] Канал {normalized_channel} не найден в БД")
            await BotMessageManager.error(update, "Канал не найден.")
            return CHANNEL_MENU
        
        # Автоматическая синхронизация статуса с ChromaDB перед проверкой (только если канал не готов)
        # Это ускоряет работу, так как не проверяем ChromaDB для готовых каналов
        if not row or (row and not row["has_chunks"]):
            logger.debug(f"[BOT] Синхронизация статуса канала {normalized_channel} с ChromaDB...")
            from tgbot.bot_db import sync_channel_status_with_chromadb
            sync_result = sync_channel_status_with_chromadb(normalized_channel)
            # Обновляем row после синхронизации
            row = get_channel(normalized_channel)
        
        if not row["has_chunks"]:
            chunk_count = sync_result.get("chunk_count", 0)
            logger.warning(f"[BOT] Канал {channel} не готов: найдено {chunk_count} чанков")
            error_msg = (
                f"Канал ещё не синхронизирован.\n"
                f"Найдено чанков в базе: {chunk_count}\n"
                f"Нажмите '🔄 Синхронизировать' для загрузки данных."
            )
            await BotMessageManager.error(update, error_msg)
            return CHANNEL_MENU

        mode = context.user_data.get("mode", "chunks")
        channel_id = row["id"]
        mode = get_user_channel_mode(uid, channel_id) or mode
    else:
        mode = "multi"

    question = (update.message.text or "").strip()
    if not question:
        return ASKING

    if multi_channels:
        logger.info(f"[BOT] Обработка мультиканального вопроса по каналам: {', '.join(multi_channels)}")
    else:
        logger.info(f"[BOT] Обработка вопроса по каналу {channel}, режим: {mode}")
    logger.debug(f"[BOT] Вопрос: {question[:100]}..." if len(question) > 100 else f"[BOT] Вопрос: {question}")

    # Отправляем "Думаю..." и сохраняем сообщение для удаления
    thinking_msg = await BotMessageManager.thinking(update)
    if thinking_msg:
        context.user_data["thinking_message_id"] = thinking_msg.message_id
        context.user_data["thinking_chat_id"] = thinking_msg.chat_id

    try:
        logger.info(f"[BOT] Вызов RAG pipeline для генерации ответа...")
        
        # Проверяем, это мультиканальный запрос?
        if multi_channels and len(multi_channels) >= 2:
            # Мультиканальный запрос
            ans = rag_answer_multi(multi_channels, question)
            logger.info(f"[BOT] Мультиканальный ответ получен, длина: {len(ans)} символов")
        else:
            # Обычный запрос по одному каналу
            ans = rag_answer(normalized_channel, question, mode)
            logger.info(f"[BOT] Ответ получен, длина: {len(ans)} символов")
    except Exception:
        logger.exception("[BOT] Ошибка при генерации ответа RAG")
        # Удаляем "Думаю..." при ошибке
        if thinking_msg:
            try:
                await thinking_msg.delete()
            except Exception:
                pass
        await BotMessageManager.error(update, "Ошибка анализа. Попробуйте другой вопрос.")
        return ASKING

    # Кнопки зависят от типа запроса
    if multi_channels:
        buttons = [
            [InlineKeyboardButton("❓ Новый вопрос", callback_data="multi_search")],
            [InlineKeyboardButton("🏠 Каналы", callback_data="channels")],
        ]
    else:
        buttons = [
            [InlineKeyboardButton("❓ Новый вопрос", callback_data=f"light:{normalized_channel}")],
            [InlineKeyboardButton("🏠 Каналы", callback_data="channels")],
        ]

    # Очищаем текст от markdown символов, но сохраняем HTML ссылки если они есть
    clean = clean_model_text(ans)
    
    # Определяем parse_mode: если в тексте есть HTML ссылки (<a href=...>), используем HTML
    parse_mode = None
    if "<a href=" in clean:
        parse_mode = ParseMode.HTML
        # HTML уже экранирован в source_links.py, но нужно убедиться что остальной текст безопасен
        # clean_model_text убирает markdown, но не трогает HTML
    
    # Отправляем ответ LLM - это сообщение НИКОГДА не должно удаляться
    answer_msg = await BotMessageManager.answer(
        update,
        clean,
        buttons,
        parse_mode=parse_mode,
    )
    
    # ВАЖНО: answer_msg НИКОГДА не удаляется - ответы LLM должны оставаться в чате
    # Удаляем только служебные сообщения после отправки ответа
    bot = context.bot
    
    # Защита от случайного удаления ответа LLM
    # answer_msg не сохраняется в context.user_data и не удаляется
    
    # Удаляем "Думаю..."
    if thinking_msg:
        try:
            await thinking_msg.delete()
        except Exception as e:
            logger.warning(f"Failed to delete thinking message: {e}")
    
    # Удаляем "Задайте вопрос по этому каналу:"
    prompt_msg_id = context.user_data.get("prompt_message_id")
    prompt_chat_id = context.user_data.get("prompt_chat_id")
    if prompt_msg_id and prompt_chat_id:
        try:
            await bot.delete_message(chat_id=prompt_chat_id, message_id=prompt_msg_id)
            context.user_data.pop("prompt_message_id", None)
            context.user_data.pop("prompt_chat_id", None)
        except Exception as e:
            logger.warning(f"Failed to delete prompt message: {e}")
    
    # Удаляем список каналов "Ваши каналы:"
    channels_list_msg_id = context.user_data.get("channels_list_msg_id")
    channels_list_chat_id = context.user_data.get("channels_list_chat_id")
    if channels_list_msg_id and channels_list_chat_id:
        try:
            await bot.delete_message(chat_id=channels_list_chat_id, message_id=channels_list_msg_id)
            context.user_data.pop("channels_list_msg_id", None)
            context.user_data.pop("channels_list_chat_id", None)
        except Exception as e:
            logger.warning(f"Failed to delete channels list message: {e}")
    
    # Удаляем сообщение мультиканального поиска
    multi_search_msg_id = context.user_data.get("multi_search_msg_id")
    multi_search_chat_id = context.user_data.get("multi_search_chat_id")
    if multi_search_msg_id and multi_search_chat_id:
        try:
            await bot.delete_message(chat_id=multi_search_chat_id, message_id=multi_search_msg_id)
            context.user_data.pop("multi_search_msg_id", None)
            context.user_data.pop("multi_search_chat_id", None)
        except Exception as e:
            logger.warning(f"Failed to delete multi search message: {e}")
    
    # Удаляем сообщение "Канал добавлен"
    channel_added_msg_id = context.user_data.get("channel_added_msg_id")
    channel_added_chat_id = context.user_data.get("channel_added_chat_id")
    if channel_added_msg_id and channel_added_chat_id:
        try:
            await bot.delete_message(chat_id=channel_added_chat_id, message_id=channel_added_msg_id)
            context.user_data.pop("channel_added_msg_id", None)
            context.user_data.pop("channel_added_chat_id", None)
        except Exception as e:
            logger.warning(f"Failed to delete channel added message: {e}")

    return ASKING


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Централизованный обработчик ошибок бота."""
    from exceptions import RAGError, ConfigurationError, LLMError, ChromaDBError
    
    error = context.error
    logger.error(f"Exception while handling an update: {error}", exc_info=error)
    
    # Telegram API ошибки
    if isinstance(error, tg_error.Conflict):
        logger.error("=" * 60)
        logger.error("ОШИБКА: Другой экземпляр бота уже запущен!")
        logger.error("Остановите все другие экземпляры бота перед запуском нового.")
        logger.error("=" * 60)
        sys.exit(1)
    elif isinstance(error, tg_error.NetworkError):
        logger.warning(f"Network error: {error}. Retrying...")
    elif isinstance(error, tg_error.TimedOut):
        logger.warning(f"Request timed out: {error}")
    
    # Наши типизированные исключения
    elif isinstance(error, ConfigurationError):
        logger.error(f"Ошибка конфигурации: {error}")
        if update and hasattr(update, 'message') and update.message:
            await update.message.reply_text(
                f"❌ Ошибка конфигурации: {error}\n"
                "Проверьте настройки в .env файле."
            )
    elif isinstance(error, LLMError):
        logger.error(f"Ошибка LLM: {error}")
        if update and hasattr(update, 'message') and update.message:
            await update.message.reply_text(
                f"❌ Ошибка генерации ответа: {error}\n"
                "Попробуйте позже или проверьте настройки LLM."
            )
    elif isinstance(error, ChromaDBError):
        logger.error(f"Ошибка ChromaDB: {error}")
        if update and hasattr(update, 'message') and update.message:
            await update.message.reply_text(
                f"❌ Ошибка базы данных: {error}\n"
                "Попробуйте позже или переиндексируйте каналы."
            )
    elif isinstance(error, RAGError):
        logger.error(f"Ошибка RAG: {error}")
        if update and hasattr(update, 'message') and update.message:
            await update.message.reply_text(
                f"❌ Ошибка поиска: {error}\n"
                "Попробуйте позже."
            )
    else:
        logger.error(f"Unexpected error: {error}")


def main():
    init_db()

    app = Application.builder().token(TOKEN).build()

    # Добавить обработчик ошибок
    app.add_error_handler(error_handler)

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            NO_CHANNEL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_channel_input),
                CallbackQueryHandler(handle_callback),
            ],
            CHANNEL_MENU: [
                CallbackQueryHandler(handle_callback),
            ],
            ASKING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, ask_question),
                CallbackQueryHandler(handle_callback),
            ],
            MULTI_CHANNEL_SELECT: [
                CallbackQueryHandler(handle_callback),
            ],
        },
        fallbacks=[CommandHandler("channels", show_channels)],
        allow_reentry=True,
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("channels", show_channels))

    logger.info("Bot starting...")
    logger.info("Если видите ошибку Conflict, остановите другие экземпляры бота!")
    logger.info("Бот готов к работе. Отправьте /start в Telegram для начала.")
    
    try:
        # Сбросить pending updates и запустить polling
        app.run_polling(
            drop_pending_updates=True,
            close_loop=False,
        )
    except tg_error.Conflict as e:
        logger.error("=" * 60)
        logger.error("КОНФЛИКТ: Другой экземпляр бота уже запущен!")
        logger.error(f"Ошибка: {e}")
        logger.error("Решение:")
        logger.error("1. Найдите и остановите другие процессы Python")
        logger.error("2. Подождите 5-10 секунд")
        logger.error("3. Запустите бота снова")
        logger.error("=" * 60)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

