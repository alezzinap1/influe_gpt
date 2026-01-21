from sentence_transformers import SentenceTransformer
import sys
import os
import threading
import logging
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
from typing import List, Dict, Any, Optional
import chromadb

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from settings import *

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class ChromaStore:
    """
    Singleton для ChromaStore - переиспользует модель и соединение.
    Экономит память и время загрузки.
    """
    _instance: Optional['ChromaStore'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(ChromaStore, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Инициализируем только один раз
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
            logger.info(f"[ChromaStore] Инициализация модели на устройстве: {device}")
            self.model = SentenceTransformer(EMBED_MODEL, device=device)
            self.device = device
            
            self.client = chromadb.PersistentClient(path=CHROMA_DIR)
            self.collection = self.client.get_or_create_collection(name="rag_docs")
            self._initialized = True

    def add_chunks(self, chunks: List[Dict[str, Any]], chroma_batch_size: int = 500):
        """
        Добавляет чанки батчами для оптимизации производительности.
        
        Args:
            chunks: Список чанков для добавления
            chroma_batch_size: Размер батча для добавления в ChromaDB (по умолчанию 500)
        """
        import math
        import pandas as pd
        
        total = len(chunks)
        if total == 0:
            return
        
        logger.info(f"[ChromaStore] Добавление {total} чанков батчами по {chroma_batch_size}")
        
        for batch_start in range(0, total, chroma_batch_size):
            batch_end = min(batch_start + chroma_batch_size, total)
            batch_chunks = chunks[batch_start:batch_end]
            
            ids, metadatas, docs = [], [], []
            texts_for_embed = []
            
            for i, chunk in enumerate(batch_chunks):
                m = dict(chunk.get("metadata", {}))
                
                m_clean = {}
                for k, v in m.items():
                    if v is None:
                        continue
                    if pd.isna(v) if hasattr(pd, 'isna') else False:
                        continue
                    if isinstance(v, float) and math.isnan(v):
                        continue
                    if isinstance(v, str) and v.lower() in ('nan', 'none', '<na>', 'null'):
                        continue
                    m_clean[k] = v
                
                if m_clean.get("type") == "summary" or m_clean.get("type") == "author_report":
                    base = f"{m_clean.get('channel', 'unknown')}_{m_clean.get('type')}_{m_clean.get('period', m_clean.get('doc_id', ''))}"
                    id_ = f"{base}_{batch_start + i}"
                else:
                    chunk_id = m_clean.get("chunk_id", batch_start + i)
                    channel = m_clean.get("channel", "unknown")
                    id_ = f"tg_{channel}_{chunk_id}"
                
                ids.append(id_)
                metadatas.append(m_clean)
                docs.append(chunk["text"])
                texts_for_embed.append(f"passage: {chunk['text']}")
            
            if texts_for_embed:
                embeddings = self.model.encode(
                    texts_for_embed,
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True
                ).tolist()
                
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=docs,
                )
                
                if batch_end % 500 == 0 or batch_end == total:
                    logger.info(f"[ChromaStore] Добавлено {batch_end}/{total} чанков...")
        
        logger.info(f"[ChromaStore] Добавлено {total} чанков")

    def count(self):
        return self.collection.count()

    def has_chunks_for_channel(self, channel: str) -> bool:
        """Проверяет, есть ли обычные чанки (не summary) для указанного канала в ChromaDB."""
        try:
            # Оптимизация: используем get с limit=1 для быстрой проверки наличия
            # Это быстрее, чем загружать все метаданные
            results = self.collection.get(
                where={"channel": channel},
                limit=1
            )
            # Проверяем, что это не summary и не author_report
            if results.get("metadatas") and len(results["metadatas"]) > 0:
                for meta in results["metadatas"]:
                    chunk_type = meta.get("type", "")
                    # Обычные чанки Telegram не имеют type или имеют type="chunk"
                    if chunk_type not in ("summary", "author_report"):
                        return True
            return False
        except Exception as e:
            print(f"Error checking chunks for channel {channel}: {e}")
            return False

    def count_chunks_for_channel(self, channel: str) -> int:
        """Подсчитывает количество обычных чанков (не summary) для канала."""
        try:
            # Загружаем все метаданные для точного подсчета
            # Это нормально, так как синхронизация вызывается только для неготовых каналов
            results = self.collection.get(
                where={"channel": channel}
            )
            if not results.get("metadatas"):
                return 0
            
            # Подсчитываем только чанки (не summary и не author_report)
            count = 0
            for meta in results["metadatas"]:
                chunk_type = meta.get("type", "")
                # Обычные чанки Telegram не имеют type или имеют type="chunk"
                if chunk_type not in ("summary", "author_report"):
                    count += 1
            return count
        except Exception as e:
            logger.error(f"Error counting chunks for channel {channel}: {e}")
            return 0

    def has_summaries_for_channel(self, channel: str) -> bool:
        """Проверяет, есть ли саммари для указанного канала в ChromaDB."""
        try:
            # ChromaDB требует использовать $and для множественных условий
            results = self.collection.get(
                where={"$and": [{"channel": channel}, {"type": "summary"}]},
                limit=1
            )
            return bool(results.get("metadatas") and len(results["metadatas"]) > 0)
        except Exception as e:
            print(f"Error checking summaries for channel {channel}: {e}")
            return False

    def delete_channel_data(self, channel: str) -> dict:
        """
        Удаляет все данные канала из ChromaDB (чанки, саммари, author_report).
        
        Args:
            channel: Имя канала для удаления
            
        Returns:
            dict с информацией о количестве удаленных записей
        """
        try:
            # Подсчитываем количество перед удалением
            chunks_count = self.count_chunks_for_channel(channel)
            
            # Получаем все записи канала для подсчета саммари
            all_results = self.collection.get(where={"channel": channel})
            summaries_count = 0
            if all_results.get("metadatas"):
                for meta in all_results["metadatas"]:
                    if meta.get("type") in ("summary", "author_report"):
                        summaries_count += 1
            
            # Удаляем все данные канала
            self.collection.delete(where={"channel": channel})
            
            return {
                "success": True,
                "chunks_deleted": chunks_count,
                "summaries_deleted": summaries_count,
                "total_deleted": chunks_count + summaries_count
            }
        except Exception as e:
            print(f"Error deleting channel data for {channel}: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_deleted": 0,
                "summaries_deleted": 0,
                "total_deleted": 0
            }


if __name__ == "__main__":
    print(ChromaStore().count())
