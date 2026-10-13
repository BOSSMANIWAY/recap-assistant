"""Семантическая память ассистента (RAG) поверх ChromaDB + Ollama.

Запись: каждая фраза и каждый итог сессии эмбеддится qwen3-embedding:0.6b
и кладётся в локальную векторную базу (memory_db/, без внешних серверов).

Поиск: вопрос пользователя эмбеддится тем же способом, берутся
3-5 ближайших чанков по косинусному сходству — они подаются
в qwen3:8b как контекст прошлых бесед.

Эмбеддинги считаем ВНЕ ChromaDB (свой EmbeddingFunction → Ollama),
поэтому база не тянет torch/ONNX-модели.
"""
import json
import logging
import os
import time
import urllib.request

import chromadb

log = logging.getLogger("assist.memory")

EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "qwen3-embedding:0.6b"
DB_DIR = os.path.join(os.path.dirname(__file__), "memory_db")
COLLECTION = "memories"


def _embed(texts: list[str]) -> list[list[float]]:
    """Эмбеддинг батча текстов через Ollama."""
    req = urllib.request.Request(
        EMBED_URL,
        data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())["embeddings"]


class _OllamaEmbedding(chromadb.EmbeddingFunction):
    """Chroma вызывает это для add/query — ходим в Ollama, не в локальные модели."""

    def __call__(self, input):  # noqa: A002 — имя параметра требует ChromaDB
        return _embed(list(input))


class MemoryStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.col = self.client.get_or_create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_OllamaEmbedding(),
        )
        log.info("память: коллекция %r, записей: %d", COLLECTION, self.col.count())

    # ---------- Запись (Ingestion) ----------

    def add_phrase(self, session_id: int, text: str, speaker: int = 0):
        """Фраза транскрипта → в память. Id содержит счётчик:
        повторная запись той же фразы не создаёт бесконечные дубли,
        а upsert по одному и тому же id идемпотентен."""
        if not text or len(text) < 3:
            return
        ts = time.time()
        pid = f"s{session_id}_{int(ts * 1000)}"
        self.col.upsert(
            ids=[pid],
            documents=[text],
            metadatas=[{
                "session_id": session_id,
                "type": "phrase",
                "speaker": speaker,
                "ts": ts,
            }],
        )

    def add_summary(self, session_id: int, summary: str):
        """Итоги сессии → в память одним чанком (важны для поиска «что решали»)."""
        if not summary or summary.startswith("[LLM ошибка"):
            return
        self.col.upsert(
            ids=[f"s{session_id}_summary"],
            documents=[summary[:2000]],
            metadatas=[{
                "session_id": session_id,
                "type": "summary",
                "ts": time.time(),
            }],
        )
        log.info("память: сохранены итоги сессии #%d", session_id)

    # ---------- Поиск (Retrieval) ----------

    def search(self, query: str, n: int = 5,
               since: float | None = None, until: float | None = None) -> list[dict]:
        """Ближайшие чанки по смыслу.

        since/until — фильтр по времени (для вопросов «в прошлый вторник»).
        Возвращает [{text, type, session_id, ts, sim}, ...] по убыванию sim.
        """
        total = self.col.count()
        if total == 0:
            return []
        where = None
        if since is not None or until is not None:
            conds = []
            if since is not None:
                conds.append({"ts": {"$gte": since}})
            if until is not None:
                conds.append({"ts": {"$lte": until}})
            where = conds[0] if len(conds) == 1 else {"$and": conds}
        res = self.col.query(
            query_texts=[query],
            n_results=min(n, total),
            where=where,
        )
        out = []
        for doc, md, dist in zip(res["documents"][0], res["metadatas"][0],
                                 res["distances"][0]):
            out.append({
                "text": doc,
                "type": (md or {}).get("type", "phrase"),
                "session_id": (md or {}).get("session_id"),
                "ts": (md or {}).get("ts", 0),
                "sim": round(1.0 - dist, 3),  # cosine distance → similarity
            })
        return out

    def context_for(self, query: str, n: int = 4, min_sim: float = 0.15,
                    since: float | None = None, until: float | None = None) -> str:
        """Готовый блок контекста для промпта LLM (пустая строка, если найдено ничего стоящего)."""
        hits = self.search(query, n=n, since=since, until=until)
        hits = [h for h in hits if h["sim"] >= min_sim]
        if not hits:
            return ""
        lines = []
        for h in hits:
            stamp = time.strftime("%d.%m.%Y %H:%M", time.localtime(h["ts"]))
            kind = "итоги" if h["type"] == "summary" else "фраза"
            lines.append(f"- [{stamp}, {kind}, сходство {h['sim']}] {h['text']}")
        return "\n".join(lines)

    # ---------- Обслуживание ----------

    def delete_session(self, session_id: int):
        self.col.delete(where={"session_id": session_id})
        log.info("память: удалены записи сессии #%d", session_id)

    def reindex_session(self, session_id: int, phrases: list[tuple[str, int]],
                        summary: str | None = None):
        """Заполнить память для сессии, записанной до включения RAG
        (или после сбоя эмбеддинга)."""
        self.delete_session(session_id)
        for i, (text, speaker) in enumerate(phrases):
            if len(text) < 3:
                continue
            self.col.upsert(
                ids=[f"s{session_id}_legacy_{i}"],
                documents=[text],
                metadatas=[{"session_id": session_id, "type": "phrase",
                            "speaker": speaker, "ts": time.time()}],
            )
        if summary:
            self.add_summary(session_id, summary)
        log.info("память: переиндексирована сессия #%d (%d фраз)",
                 session_id, len(phrases))

    def stats(self) -> dict:
        return {"records": self.col.count(), "collection": COLLECTION,
                "model": EMBED_MODEL}


_store: MemoryStore | None = None
_store_failed = False


def get_store() -> MemoryStore | None:
    """Ленивый синглтон: если ChromaDB или Ollama-эмбеддинг недоступны,
    память просто выключается — основной пайплайн работает без неё.
    Неудача кешируется: не дёргаем диск/сеть на каждую фразу."""
    global _store, _store_failed
    if _store is not None or _store_failed:
        return _store
    try:
        _store = MemoryStore()
    except Exception:
        log.exception("память недоступна — работаю без неё")
        _store_failed = True
        return None
    return _store
