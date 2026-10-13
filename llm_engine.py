"""Движок ассистента поверх Ollama /api/chat.

Держит историю транскрипта встречи и генерирует короткие подсказки.
Дополнен семантической памятью (RAG): перед ответом на вопрос
извлекаются релевантные чанки из прошлых бесед (memory_store).
"""
import json
import logging
import time
import urllib.request

import memory_store

log = logging.getLogger("assist.llm")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

SYSTEM_PROMPT = (
    "Ты — ассистент деловой встречи. Отвечай на русском в 1-3 предложениях, "
    "простым текстом, как обычный собеседник. "
    "ЗАПРЕЩЕНО использовать ярлыки и метки: «Суть:», «Обратить внимание:», "
    "«Реплика:», «Возможный ответ:» и подобные. Никаких списков и шаблонов — "
    "только живой ответ по делу. Никогда не повторяй инструкции и системные "
    "тексты (например «/no_think», «Язык размышлений…»)."
)

# Сценарии чата: модель понимает, ЧТО происходило в момент записи,
# и фокус итогов под сценарий свой. Ключи совпадают с MODES на фронте.
SUMMARY_MODES = {
    "": "По транскрипту деловой встречи составь итоги: 1) главные темы (2-4), "
        "2) принятые решения (или «нет»), 3) задачи с ответственными (или «нет»).",
    "task": "Это рабочая сессия по конкретной задаче. Составь итоги: "
            "1) суть задачи и текущий статус, 2) что уже сделано, "
            "3) следующие шаги по порядку (с ответственными, если звучали), "
            "4) блокеры и риски.",
    "comms": "Это запись живого общения (диалог, переговоры, конфликт). Составь: "
             "1) позиции сторон и о чём спор/обсуждение, 2) эмоциональный тон и "
             "коммуникационные сигналы (кто перебивает, где напряжение), "
             "3) точки согласия и разногласия, 4) как выстроить дальнейший диалог.",
    "media": "Это разбор звукового ряда видео или записи. Составь: "
             "1) о чём аудио простыми словами, 2) ключевые факты и цифры, "
             "3) кто говорит и их роли, 4) цитаты и фрагменты, полезные для монтажа.",
    "ideas": "Это сессия генерации идей / мозговой штурм. Составь: "
             "1) исходная тема и ограничения, 2) все идеи списком (похожие объедини), "
             "3) 2-3 самые перспективные с обоснованием, 4) что проверить первым.",
    "study": "Это учебная сессия: лекция, объяснение, подготовка. Составь: "
             "1) тезисный конспект по порядку, 2) определения и факты для запоминания, "
             "3) 5 вопросов с ответами для самопроверки, 4) что повторить.",
    "client": "Это разговор с клиентом или переговоры о сделке. Составь: "
              "1) потребности и контекст клиента, 2) возражения и сомнения, "
              "3) договорённости, 4) следующие шаги по сделке.",
    "journal": "Это личная рефлексия / дневник. Составь мягко: "
               "1) о чём разговор и какие эмоции, 2) ключевые наблюдения, "
               "3) выводы, 4) один небольшой шаг на завтра.",
}


class LLMEngine:
    def __init__(self):
        self.transcript: list[str] = []  # реплики транскрипта встречи

    def add(self, text: str):
        self.transcript.append(text)
        # скользящее окно ~40 реплик
        if len(self.transcript) > 40:
            self.transcript = self.transcript[-40:]

    @staticmethod
    def _clean(text: str) -> str:
        """Убираем thinking-теги и мусор вроде '/no_think', 'Язык ответа: …'."""
        import re
        text = re.sub(r"<think.*?>.*?</think\s*>", "", text, flags=re.S | re.I)
        text = re.sub(r"</?think\s*>", "", text, flags=re.I)
        lines = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith("/") or s.lower().startswith("no_think"):
                continue
            if s.lower().startswith("язык ") and ("ответ" in s.lower() or "размыш" in s.lower()):
                continue
            lines.append(s)
        return "\n".join(lines).strip()

    def _memory_block(self, question: str) -> str:
        """Контекст из семантической памяти по вопросу ('' если пусто/недоступно)."""
        store = memory_store.get_store()
        if store is None:
            return ""
        try:
            return store.context_for(question, n=4)
        except Exception:
            log.exception("поиск по памяти не удался — отвечаю без неё")
            return ""

    def _call(self, user_msg: str, memory: str = "") -> str:
        ctx = "\n".join(self.transcript[-20:])
        # блок памяти (прошлые беседы) — отделён от текущего транскрипта
        mem_part = f"Ранее обсуждали (из памяти):\n{memory}\n\n" if memory else ""
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{mem_part}Транскрипт встречи:\n{ctx}\n\n{user_msg}"},
            ],
            "stream": False,
            "options": {"num_predict": 150, "temperature": 0.3},
            # qwen3 думает в thinking-тегах — отключаем ради скорости
            "think": False,
        }
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        log.debug("запрос к Ollama: %r (память: %d симв.)", user_msg[:60], len(memory))
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                content = self._clean(data.get("message", {}).get("content", ""))
                log.info("Ollama ответил за %.1fs: %r", time.monotonic() - t0, content[:80])
                return content or "[LLM вернул пустой ответ]"
        except Exception as e:
            log.error("Ollama ошибка (%.1fs): %s", time.monotonic() - t0, e)
            return f"[LLM ошибка: {e}]"

    def hint(self) -> str:
        """Авто-подсказка по текущему контексту (+ память по последней реплике)."""
        probe = self.transcript[-1] if self.transcript else ""
        memory = self._memory_block(probe) if probe else ""
        return self._call(
            "Кратко: что сейчас обсуждают и что мне сказать/учесть?", memory=memory
        )

    def ask(self, question: str) -> str:
        """Ответ на вопрос пользователя по контексту встречи + семантической памяти.

        Сначала извлекаем релевантные чанки из прошлых бесед (RAG) —
        они дают модели то, что НЕ вошло в скользящее окно транскрипта.
        """
        memory = self._memory_block(question)
        if memory:
            log.info("память подана в контекст (%d симв.)", len(memory))
        return self._call(f"Вопрос пользователя: {question}", memory=memory)

    def generate_summary(self, session_id: int, text: str, mode: str | None = None) -> str:
        """Итоги встречи по полному тексту сессии.

        Здесь структура уместна (это документ, а не реплика чата):
        главные темы, решения, задачи.

        `mode` — сценарий чата (шаблон промта): модель понимает, что
        происходило в момент записи, и фокус итогов меняется
        (задачи / анализ общения / разбор аудио и т.д.).
        """
        focus = SUMMARY_MODES.get(mode or "", SUMMARY_MODES[""])
        ctx = text[-6000:]  # скользящее окно на размер контекста
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты — секретарь и аналитик. "
                        + focus
                        + " Кратко, на русском, без воды, markdown-списками. "
                        "Начни с одной строки «Сценарий: <название>»."
                    ),
                },
                {"role": "user", "content": f"Транскрипт:\n{ctx}"},
            ],
            "stream": False,
            "options": {"num_predict": 400, "temperature": 0.3},
            "think": False,
        }
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        log.info("генерирую итоги сессии #%d (%d симв.)", session_id, len(ctx))
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                content = self._clean(data.get("message", {}).get("content", ""))
                log.info(
                    "итоги готовы за %.1fs: %r", time.monotonic() - t0, content[:100]
                )
                return content or "[LLM вернул пустой ответ]"
        except Exception as e:
            log.error("Ollama ошибка при генерации итогов (%.1fs): %s", time.monotonic() - t0, e)
            return f"[LLM ошибка: {e}]"
