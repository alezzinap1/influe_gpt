#!/usr/bin/env python3
"""
Скрипт для проверки импортов и синтаксиса основных модулей проекта.
Запускается без загрузки тяжелых зависимостей (torch, sentence-transformers).
"""

import sys
import ast
from pathlib import Path

def check_syntax(file_path: Path) -> bool:
    """Проверяет синтаксис Python файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source, filename=str(file_path))
        return True
    except SyntaxError as e:
        print(f"[ERROR] Синтаксическая ошибка в {file_path}:")
        print(f"   Строка {e.lineno}: {e.text}")
        print(f"   {e.msg}")
        return False
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке {file_path}: {e}")
        return False

def check_imports_lightweight():
    """Проверяет импорты без тяжелых зависимостей."""
    import sys
    base_dir = Path(__file__).parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    
    print("Проверка импортов (без тяжелых зависимостей)...\n")
    
    modules_to_check = [
        "config_telegram",
        "tg_channels.schema",
        "settings",
        "utils.keywords",
        "utils.chunker",
    ]
    
    all_ok = True
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
        except ImportError as e:
            print(f"[ERROR] {module_name}: {e}")
            all_ok = False
        except Exception as e:
            print(f"[WARN] {module_name}: {e} (может быть нормально)")
    
    return all_ok

def main():
    # Определяем базовую директорию (rag_mvp)
    # Если скрипт запущен из scripts/, поднимаемся на уровень выше
    base_dir = Path(__file__).parent
    if base_dir.name == "scripts":
        base_dir = base_dir.parent
    
    print("=" * 60)
    print("Проверка работоспособности RAG MVP проекта")
    print("=" * 60)
    print()
    
    # Проверка синтаксиса основных файлов
    print("Проверка синтаксиса файлов...\n")
    
    files_to_check = [
        base_dir / "main.py",
        base_dir / "settings.py",
        base_dir / "config_telegram.py",
        base_dir / "telegram" / "ingest.py",
        base_dir / "telegram" / "schema.py",
        base_dir / "telegram" / "prompts.py",
        base_dir / "telegram" / "summaries.py",
        base_dir / "rag" / "pipeline.py",
        base_dir / "rag" / "llm_backends.py",
        base_dir / "vectorstore" / "chromadb_store.py",
        base_dir / "tgbot" / "bot.py",
        base_dir / "tgbot" / "bot_db.py",
        base_dir / "tgbot" / "bot_tasks.py",
    ]
    
    syntax_ok = True
    for file_path in files_to_check:
        if file_path.exists():
            if check_syntax(file_path):
                print(f"[OK] {file_path}")
            else:
                syntax_ok = False
        else:
            print(f"[WARN] {file_path} не найден")
    
    print()
    
    # Проверка легковесных импортов
    imports_ok = check_imports_lightweight()
    
    print()
    print("=" * 60)
    if syntax_ok and imports_ok:
        print("[SUCCESS] Все проверки пройдены успешно!")
        print("\nДля полной проверки работоспособности:")
        print("1. Убедитесь, что все зависимости установлены: pip install -r requirements.txt")
        print("2. Проверьте CLI команды:")
        print("   python main.py status")
        print("   python main.py tg-sync --help")
        print("3. Проверьте Telegram бота:")
        print("   python tgbot/bot.py")
        return 0
    else:
        print("[ERROR] Обнаружены проблемы. Исправьте ошибки выше.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

