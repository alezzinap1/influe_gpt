"""RAG query LLM."""
from vectorstore.chromadb_store import ChromaStore
import sys
import time
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from settings import LLM_MODEL, OLLAMA_BASE, SUMM_K, CHUNKS_K, MAX_CTX_CHARS, MAX_CHUNK_CHARS
from rag.llm_backends import OllamaBackend, DeepSeekBackend, OpenAIBackend, GeminiBackend
from settings import RAG_SYSTEM_PROMPT, RAG_ANSWER_STYLE
from transformers import pipeline
import numpy as np
from rank_bm25 import BM25Okapi
from utils.keywords import extract_keywords, string_to_keywords
from utils.metrics import track_time, PerformanceMetrics

sys.path.insert(0, '.')  # settings/ChromaStore

logger = logging.getLogger(__name__)

_CHROMA_QUERY_INCLUDE = ["metadatas", "documents", "distances"]


def _documents_fingerprint(docs: List[str]) -> str:
    """Инкрементальный отпечаток списка документов для кэша BM25 без join всего корпуса в одну строку."""
    h = hashlib.md5()
    sep = b"|"
    for i, doc in enumerate(docs):
        h.update(str(i).encode("ascii", errors="ignore"))
        h.update(sep)
        h.update(doc.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _chroma_query(collection, emb: List[float], n_results: int, where: Optional[Dict[str, Any]]):
    return collection.query(
        query_embeddings=[emb],
        n_results=n_results,
        include=_CHROMA_QUERY_INCLUDE,
        where=where,
    )


class RAGPipeline:
    def __init__(self, backend: str = "local"):
        self.store = ChromaStore()
        if backend == "local":
            self.llm = OllamaBackend()
        elif backend == "deepseek":
            self.llm = DeepSeekBackend()
        elif backend == "openai":
            self.llm = OpenAIBackend()
        elif backend == "gemini":
            self.llm = GeminiBackend()
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")
        # Ленивая загрузка classifier - загружается только при первом использовании
        self._classifier = None
        # Кэширование BM25 индекса
        self.bm25_index = None
        self.bm25_corpus = None
        self.bm25_corpus_hash = None
    
    @property
    def classifier(self):
        """Ленивая загрузка classifier при первом обращении."""
        if self._classifier is None:
            logger.debug("[RAG] Загрузка classifier...")
            self._classifier = pipeline("zero-shot-classification",
                                       model="typeform/distilbert-base-uncased-mnli",  # 29MB вместо 714MB
                                       device=-1)
            logger.debug("[RAG] Classifier загружен")
        return self._classifier

    def _filter_by_keywords(self, results: Dict[str, Any], question_keywords: List[str]) -> Dict[str, Any]:
        """
        Keyword-based постфильтрация после semantic search.
        Фильтрует результаты по keywords в metadata.
        Время: ~0.001s
        
        Args:
            results: Результаты запроса из ChromaDB
            question_keywords: Список keywords из вопроса
            
        Returns:
            Отфильтрованные результаты
        """
        if not question_keywords or not results.get("metadatas") or not results["metadatas"][0]:
            return results
        
        # Фильтруем результаты по keywords
        filtered_metas = []
        filtered_docs = []
        filtered_dists = []
        
        for meta, doc, dist in zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0]
        ):
            # Проверяем, есть ли keywords в metadata чанка
            chunk_keywords_str = meta.get("keywords", "")
            if not chunk_keywords_str:
                # Если keywords нет, оставляем чанк (обратная совместимость)
                filtered_metas.append(meta)
                filtered_docs.append(doc)
                filtered_dists.append(dist)
                continue
            
            # Проверяем совпадение хотя бы одного keyword
            chunk_keywords = [kw.strip().lower() for kw in chunk_keywords_str.split(",")]
            question_keywords_lower = [kw.lower() for kw in question_keywords if len(kw) >= 3]
            
            # Если есть хотя бы одно совпадение, оставляем чанк
            if any(qkw in chunk_keywords for qkw in question_keywords_lower):
                filtered_metas.append(meta)
                filtered_docs.append(doc)
                filtered_dists.append(dist)
        
        # Возвращаем отфильтрованные результаты
        return {
            "metadatas": [filtered_metas] if filtered_metas else [[]],
            "documents": [filtered_docs] if filtered_docs else [[]],
            "distances": [filtered_dists] if filtered_dists else [[]],
        }

    def query(
            self,
            question,
            source: str | None = None,
            channel: str | None = None,
            channels: List[str] | None = None,  # Список каналов для мультиканального поиска
            extra_year_summaries: dict[int, str] | None = None,
            extra_quarter_summaries: dict[tuple[int, int], str] | None = None,
    ):

        start_total = time.time()  # 🟢 TOTAL START
        
        logger.info(f"[RAG] ========== Начало обработки запроса ==========")
        logger.info(f"[RAG] Вопрос: {question[:100]}..." if len(question) > 100 else f"[RAG] Вопрос: {question}")
        if channels:
            logger.info(f"[RAG] Каналы: {', '.join(channels)} (мультиканальный поиск)")
        else:
            logger.info(f"[RAG] Канал: {channel or 'все каналы'}")
        logger.info(f"[RAG] Backend: {self.llm.__class__.__name__}")

        t1 = time.time()
        query_text = f"query: {question}"
        logger.debug(f"[RAG] Создание embedding для запроса...")
        emb = self.store.model.encode(query_text).tolist()
        embedding_time = time.time() - t1
        logger.info(f"[RAG] Embedding создан за {embedding_time:.2f}s")

        # --- 2) два запроса в Chroma: саммари и сырые чанки ---
        t2 = time.time()

        where_base: dict | None = None
        # Поддержка мультиканального поиска
        if channels and len(channels) > 0:
            # Фильтр по списку каналов: channel IN [ch1, ch2, ...]
            where_base = {
                "$or": [{"channel": {"$eq": ch}} for ch in channels]
            }
            if source:
                where_base = {
                    "$and": [
                        {"source": {"$eq": source}},
                        {"$or": [{"channel": {"$eq": ch}} for ch in channels]}
                    ]
                }
        elif source and channel:
            where_base = {
                "$and": [
                    {"source": {"$eq": source}},
                    {"channel": {"$eq": channel}},
                ]
            }
        elif source:
            where_base = {"source": {"$eq": source}}
        elif channel:
            where_base = {"channel": {"$eq": channel}}

        # 2.1 Саммари
        if channels and len(channels) > 0:
            # Мультиканальный поиск: саммари из всех выбранных каналов
            where_summ = {
                "$and": [
                    {"type": {"$eq": "summary"}},
                    {"$or": [{"channel": {"$eq": ch}} for ch in channels]}
                ]
            }
        elif channel:
            where_summ = {
                "$and": [
                    {"type": {"$eq": "summary"}},
                    {"channel": {"$eq": channel}},
                ]
            }
        else:
            # Все каналы
            where_summ = {"type": {"$eq": "summary"}}

        # 2.2 Сырые чанки
        where_chunks = {"type": {"$ne": "summary"}}
        if where_base:
            where_chunks = {"$and": [where_chunks, where_base]}

        logger.info(f"[RAG] Поиск в ChromaDB: саммари (K={SUMM_K}) и чанки (K={CHUNKS_K})...")
        coll = self.store.collection
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_summ = pool.submit(_chroma_query, coll, emb, SUMM_K, where_summ)
                fut_chunks = pool.submit(_chroma_query, coll, emb, CHUNKS_K, where_chunks)
                res_summ = fut_summ.result()
                res_chunks = fut_chunks.result()
        except Exception as e:
            logger.warning("[RAG] Параллельный запрос Chroma не удался (%s), последовательный режим.", e)
            res_summ = _chroma_query(coll, emb, SUMM_K, where_summ)
            res_chunks = _chroma_query(coll, emb, CHUNKS_K, where_chunks)
        
        # НОВОЕ: Keyword-based постфильтрация (после получения результатов)
        # Извлекаем keywords из вопроса для фильтрации
        question_keywords = extract_keywords(question, top_k=5, use_keybert=False)
        logger.debug(f"[RAG] Извлечено keywords из вопроса: {question_keywords}")
        
        if question_keywords:
            logger.debug(f"[RAG] Применение keyword-фильтрации к результатам...")
            before_count = len(res_chunks.get('metadatas', [[]])[0]) if res_chunks.get('metadatas') else 0
            filtered_chunks = self._filter_by_keywords(res_chunks, question_keywords)
            after_count = len(filtered_chunks.get('metadatas', [[]])[0]) if filtered_chunks.get('metadatas') else 0
            
            # Применяем фильтрацию только если она не слишком агрессивная (осталось >= 30% результатов)
            # Это защищает от случаев, когда все чанки отфильтровываются
            if before_count > 0 and after_count >= max(1, before_count * 0.3):
                res_chunks = filtered_chunks
                logger.info(f"[RAG] Keyword-фильтрация применена: {before_count} -> {after_count} чанков")
            else:
                logger.warning(f"[RAG] Keyword-фильтрация слишком агрессивная ({before_count} -> {after_count}), пропускаем")
                logger.info(f"[RAG] Используем все {before_count} чанков без keyword-фильтрации")
        
        chroma_time = time.time() - t2
        logger.info(f"[RAG] ChromaDB запрос выполнен за {chroma_time:.2f}s")
        logger.info(f"[RAG] Найдено саммари: {len(res_summ.get('metadatas', [[]])[0]) if res_summ.get('metadatas') else 0}")
        logger.info(f"[RAG] Найдено чанков: {len(res_chunks.get('metadatas', [[]])[0]) if res_chunks.get('metadatas') else 0}")

        # --- 3) раздельный реранк ---
        t3 = time.time()
        logger.info(f"[RAG] Начало reranking...")
        question_words = set(question.lower().split())
        
        # Используем уже извлеченные keywords для BM25 (извлечены выше для фильтрации)
        logger.debug(f"[RAG] Keywords для BM25: {question_keywords}")

        def rerank(res: Dict[str, Any]) -> List[Tuple[float, str, Dict[str, Any]]]:
            """Rerank результаты поиска с использованием BM25 и keyword matching."""
            scored: List[Tuple[float, str, Dict[str, Any]]] = []
            if not res["metadatas"] or not res["metadatas"][0]:
                return scored

            metas = res["metadatas"][0]
            docs = res["documents"][0]
            dists = res["distances"][0]

            # Кэширование BM25 индекса: создаем только если документы изменились
            docs_hash = _documents_fingerprint(docs)
            
            if self.bm25_corpus_hash != docs_hash:
                logger.debug(f"[RAG] Создание BM25 индекса для {len(docs)} документов...")
                # Токенизация документов для BM25
                tokenized_docs = [doc.lower().split() for doc in docs]
                self.bm25_index = BM25Okapi(tokenized_docs)
                self.bm25_corpus = tokenized_docs
                self.bm25_corpus_hash = docs_hash
                logger.debug(f"[RAG] BM25 индекс создан и закэширован")
            else:
                logger.debug(f"[RAG] Использование закэшированного BM25 индекса")

            for i, (meta, doc, dist) in enumerate(zip(metas, docs, dists)):
                doc_lower = doc.lower()
                channel = meta.get("channel", "").lower()

                sem_score = -dist

                # НОВОЕ: BM25 scoring для keywords
                bm25_score = 0
                if question_keywords and self.bm25_index:
                    tokenized_query = [kw for kw in question_keywords if len(kw) >= 3]
                    if tokenized_query:
                        bm25_scores = self.bm25_index.get_scores(tokenized_query)
                        bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0
                
                # Старый keyword matching (оставляем как fallback, но с меньшим весом)
                kw_hits = 0
                for w in question_words:
                    if len(w) < 3:
                        continue
                    if w in doc_lower or w in channel:
                        kw_hits += 1
                kw_score = kw_hits * 0.5  # Уменьшаем вес старого метода

                # boost для совпадения в названии канала
                channel_boost = 1.0
                if channel and any(w in channel for w in question_words if len(w) > 3):
                    channel_boost = 1.3

                # Комбинируем: semantic + BM25 + старый keyword
                total_score = sem_score + bm25_score * 2.0 + kw_score * channel_boost
                scored.append((total_score, doc, meta))

            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        def mmr_select(
            scored: List[Tuple[float, str, Dict[str, Any]]],
            emb_query: List[float],
            doc_embs_precomputed: Optional[np.ndarray] = None,
            top_n: int = 15,
            lambda_mult: float = 0.8
        ) -> List[Tuple[float, str, Dict[str, Any]]]:
            """
            scored: [(score, doc, meta), ...] после rerank
            emb_query: вектор запроса (list/np.array)
            doc_embs_precomputed: предвычисленные embeddings (если есть, не кодируем повторно)
            """
            if not scored:
                return []

            # Если embeddings уже вычислены, используем их (оптимизация)
            if doc_embs_precomputed is not None:
                doc_embs = np.array(doc_embs_precomputed)
            else:
                # Fallback: кодируем если не переданы
                docs = [doc for _, doc, _ in scored]
                doc_embs = np.array(self.store.model.encode(docs))

            q = np.array(emb_query)

            # нормируем
            q = q / (np.linalg.norm(q) + 1e-8)
            doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)

            # косинусная близость
            rel = doc_embs @ q  # [len]

            selected_idx = []
            remaining_idx = list(range(len(scored)))

            while remaining_idx and len(selected_idx) < top_n:
                if not selected_idx:
                    # первый — просто самый релевантный
                    best_i = int(np.argmax(rel[remaining_idx]))
                    idx = remaining_idx[best_i]
                    selected_idx.append(idx)
                    remaining_idx.remove(idx)
                    continue

                # для каждого кандидата считаем max similarity с уже выбранными
                selected_vecs = doc_embs[selected_idx]  # [k, dim]
                sims_to_selected = doc_embs[remaining_idx] @ selected_vecs.T  # [m, k]
                max_sims = sims_to_selected.max(axis=1)  # [m]

                mmr_scores = (
                    lambda_mult * rel[remaining_idx] - (1 - lambda_mult) * max_sims
                )
                best_i = int(np.argmax(mmr_scores))
                idx = remaining_idx[best_i]
                selected_idx.append(idx)
                remaining_idx.remove(idx)

            # вернуть в том же формате [(score, doc, meta)]
            return [scored[i] for i in selected_idx]

        scored_summaries = rerank(res_summ)
        scored_chunks = rerank(res_chunks)
        rerank_time = time.time() - t3
        logger.info(f"[RAG] Reranking завершен за {rerank_time:.2f}s")
        logger.info(f"[RAG] Отранжировано саммари: {len(scored_summaries)}, чанков: {len(scored_chunks)}")

        # ОПТИМИЗАЦИЯ: Кодируем документы ОДИН раз и переиспользуем для MMR
        logger.info(f"[RAG] Применение Smart MMR для диверсификации результатов...")
        docs_for_mmr = [doc for _, doc, _ in scored_chunks]
        doc_embs_for_mmr = self.store.model.encode(docs_for_mmr)  # ОДИН раз
        candidates_embs = np.array(doc_embs_for_mmr)

        # Используем для smart_mmr_lambda
        lambda_mmr = self.smart_mmr_lambda(question, candidates_embs)
        
        # Передаем embeddings в mmr_select, чтобы не кодировать повторно
        scored_chunks = mmr_select(scored_chunks, emb, doc_embs_for_mmr, top_n=15, lambda_mult=lambda_mmr)
        logger.info(f"[RAG] Smart MMR завершен (λ={lambda_mmr:.2f}), выбрано чанков: {len(scored_chunks)}")

        # --- 4) сбор контекста: сначала саммари, потом чанки ---
        t4 = time.time()
        logger.info(f"[RAG] Сбор контекста из саммари и чанков...")
        context_parts = []

        MAX_SUMM_IN_CTX = 6
        for score, doc, meta in scored_summaries[:MAX_SUMM_IN_CTX]:
            ch = meta.get("channel", channel or "unknown")
            period = meta.get("period", meta.get("year", ""))
            header = f"САММАРИ: {ch}"
            if period:
                header += f" | ПЕРИОД: {period}"
            context_parts.append(f"{header}\n{doc.strip()}")

        MAX_CHUNKS_IN_CTX = 15
        
        def short_date(d: str) -> str:
            """Берем только часть 'YYYY-MM-DD' из ISO даты."""
            return d.split("T", 1)[0] if "T" in d else d
        
        for i, (score, doc, meta) in enumerate(scored_chunks[:MAX_CHUNKS_IN_CTX]):
            chan = meta.get("channel")
            start_ts = meta.get("start_ts")
            end_ts = meta.get("end_ts")
            msg_dates = meta.get("msg_dates")
            is_forwarded = meta.get("is_forwarded", False)
            forwarded_from = meta.get("forwarded_from")

            header = f"КАНАЛ: {chan}" if chan else "КАНАЛ: unknown"
            
            # Добавляем пометку о пересылке
            if is_forwarded and forwarded_from:
                header += f" | ПЕРЕСЫЛКА из {forwarded_from}"
            
            if msg_dates:
                dates = msg_dates.split(",")
                start_d = short_date(dates[0])
                end_d = short_date(dates[-1])
                header += f" | СООБЩЕНИЯ: {start_d} — {end_d}"
            elif start_ts and end_ts:
                header += f" | ПЕРИОД: {short_date(start_ts)} — {short_date(end_ts)}"
            elif start_ts:
                header += f" | ДАТА: {short_date(start_ts)}"

            # Мягкая обрезка: только если чанк очень длинный
            doc_trimmed = doc[:MAX_CHUNK_CHARS] + "..." if len(doc) > MAX_CHUNK_CHARS else doc
            context_parts.append(f"{header}\n{doc_trimmed.strip()}")

        context = "\n\n".join(context_parts)

        if len(context) > MAX_CTX_CHARS:
            context = context[-MAX_CTX_CHARS:]

        # агрегаты по чанкам
        total_chunks = len(context_parts)
        chunk_lens = [len(part) for part in context_parts]
        total_chunk_chars = sum(chunk_lens)

        context_time = time.time() - t4
        logger.info(f"[RAG] Контекст собран за {context_time:.2f}s")
        logger.info(f"[RAG] Контекст: {total_chunks} частей, {len(context)} символов (~{len(context) // 4} токенов)")
        logger.debug(f"[RAG] Источники (саммари): {[s[2].get('period', s[2].get('doc_id', 'unk')) for s in scored_summaries[:4]]}")
        logger.debug(f"[RAG] Источники (чанки): {[c[2].get('msg_dates', 'unk')[:20] for c in scored_chunks[:4]]}")

        if not context.strip():
            total_time = time.time() - start_total
            logger.warning(f"[RAG] Контекст пуст, возвращаем сообщение об ошибке")
            logger.info(f"[RAG] ========== Запрос завершен (нет контекста) за {total_time:.2f}s ==========")
            return "Нет контекста."

        t5 = time.time()
        logger.info(f"[RAG] Формирование промпта для LLM...")
        prompt = f"""СИСТЕМА:
        {RAG_SYSTEM_PROMPT}
        КОНТЕКСТ:
        {context}
        ИНСТРУКЦИЯ:
        {RAG_ANSWER_STYLE}
        ВОПРОС:
        {question}
        ОТВЕТ:
        """
        logger.debug(f"[RAG] Промпт сформирован, длина: {len(prompt)} символов")

        logger.info(f"[RAG] Отправка запроса в LLM ({self.llm.__class__.__name__})...")
        try:
            from exceptions import LLMError
            answer = self.llm.generate(prompt)
            llm_time = time.time() - t5
            logger.info(f"[RAG] LLM ответ получен за {llm_time:.2f}s")
            logger.debug(f"[RAG] Длина ответа: {len(answer)} символов")
        except Exception as e:
            llm_time = time.time() - t5
            logger.error(f"[RAG] Ошибка LLM: {e}", exc_info=True)
            # Graceful degradation: пробуем fallback на Ollama если доступен
            try:
                logger.warning(f"[RAG] Пробуем fallback на Ollama...")
                fallback_llm = OllamaBackend()
                answer = fallback_llm.generate(prompt)
                logger.info(f"[RAG] Fallback успешен, ответ получен от Ollama")
            except Exception as fallback_error:
                logger.error(f"[RAG] Fallback также не удался: {fallback_error}")
                from exceptions import LLMError
                raise LLMError(
                    "Сервис генерации ответов временно недоступен. "
                    "Попробуйте повторить запрос через несколько минут."
                ) from fallback_error

        try:
            from utils.source_links import filter_sources_strict, format_sources_section
            
            filtered_sources = filter_sources_strict(
                scored_chunks[:MAX_CHUNKS_IN_CTX],
                max_sources=7
            )
            
            if filtered_sources:
                sources_section = format_sources_section(filtered_sources, use_html=True)
                answer = f"{answer}{sources_section}"
                logger.debug(f"[RAG] Добавлено {len(filtered_sources)} источников в конец ответа")
        except Exception as e:
            logger.warning(f"[RAG] Ошибка при добавлении источников: {e}", exc_info=True)

        total_time = time.time() - start_total
        logger.info(f"[RAG] ========== Запрос успешно обработан за {total_time:.2f}s ==========")
        logger.info(f"[RAG] Время по этапам: embedding={embedding_time:.2f}s, chroma={chroma_time:.2f}s, rerank={rerank_time:.2f}s, context={context_time:.2f}s, llm={llm_time:.2f}s")
        return answer

    def smart_mmr_lambda(self, question: str, candidates_embs: np.array) -> float:
        """Гибрид: RuBERT-Tiny (0.02s) + Dynamic Density (0.01s)"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # 1. 🟢 RuBERT-Tiny классификатор (0.02s)
        result = self.classifier(question,
                                 ["список разных проектов", "детали одного проекта"])
        rubert_score = result['scores'][0]  # 0.92 = список
        rubert_type = 0.15 if rubert_score > 0.6 else 0.75

        # 2. 🔵 Dynamic по плотности кандидатов (0.01s)
        if len(candidates_embs) >= 5:
            sim_matrix = cosine_similarity(candidates_embs[:10])
            redundancy = sim_matrix.mean()
            dynamic_lambda = 1.0 - redundancy * 0.7  # 0.2-0.9
        else:
            dynamic_lambda = 0.5

        # 3. 🟣 Комбинируем 40% RuBERT + 60% данные
        final_lambda = 0.4 * rubert_type + 0.6 * dynamic_lambda

        print(f"🤖 RuBERT={rubert_type:.2f} Density={dynamic_lambda:.2f} → λ={final_lambda:.2f}")
        return final_lambda

if __name__ == "__main__":
    print(RAGPipeline().query(sys.argv[1]))
