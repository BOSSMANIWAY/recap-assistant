"""Ассистент встречи: стриминг → Whisper → LLM + сессии + итоги + память (RAG)."""
import asyncio
import json
import logging
import os
import signal
import wave

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

import whisper_engine
from whisper_engine import WhisperEngine
from llm_engine import LLMEngine
import llm_engine
import memory_store
import computer_control
import browser_engine
from audio_utils import PhraseAssembler
from voiceprint import SpeakerTracker
from session_store import (
    create_session,
    add_phrase,
    end_session,
    reopen_session,
    set_mode,
    get_mode,
    list_sessions,
    get_session,
    delete_session,
    rename_session,
    set_summary,
    session_text,
)

app = FastAPI()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
for noisy in ("websockets", "urllib3", "asyncio", "h11"):
    logging.getLogger(noisy).setLevel(logging.INFO)
log = logging.getLogger("assist")

# Менеджер сервисов браузера: Chrome CDP + ai_browser, поднимаются/останавливаются
# вместе с сервером (через startup/shutdown-события FastAPI).
from service_manager import get_manager
svc_mgr = get_manager()

whisper = WhisperEngine()
llm = LLMEngine()
log.info("сервер готов: whisper=%s, llm=%s", whisper_engine.MODEL_PATH, llm_engine.MODEL)

SAMPLE_RATE = 16000
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Режим приложения: recap (запись встреч) | computer (терминал) | browser
# (агент в Chrome). Голос и транскрипция работают во всех трёх; меняется
# только то, чем становится реплика: контекст встречи / команда / задача.
app_mode = "recap"

# Активная запись. Держим состояние вне соединения WebSocket:
# если сеть моргнула, клиент переподключается с session_id и запись
# продолжается в ту же сессию — ни один сегмент аудио не теряется вместе
# с соединением.
class _Live:
    def __init__(self, session_id: int):
        self.session_id = session_id
        self.asm = PhraseAssembler()
        self.speaker = SpeakerTracker()
        # очередь фраз живёт столько же, сколько сессия: фразы, не
        # успевшие обработаться до обрыва связи, транскрибируются после
        # переподключения
        self.q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self.wav_path = os.path.join(RECORDINGS_DIR, f"session_{session_id}.wav")
        resume = os.path.exists(self.wav_path)
        self.wav = wave.open(self.wav_path, "ab" if resume else "wb")
        if not resume:
            self.wav.setnchannels(1)
            self.wav.setsampwidth(2)          # int16
            self.wav.setframerate(SAMPLE_RATE)
        self.n_chunks = 0

    def close_wav(self):
        try:
            self.wav.close()
        except Exception:
            log.exception("не удалось закрыть WAV #%d", self.session_id)

_live: _Live | None = None


def _mem():
    """Магазин памяти или None (если ChromaDB/Ollama-эмбеддинг недоступны)."""
    return memory_store.get_store()


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    """Протокол:
    - бинарные кадры = PCM int16 mono 16 кГц (аудио с микрофона);
    - JSON-текст = команды: resume {session_id}, ask {q}, hint, stop.

    Аудио пишется на диск (WAV) и кладётся в очередь фраз; транскрипция
    идёт отдельным воркером, поэтому медленный Whisper никогда не блокирует
    приём микрофона и буфер переполняется только при полностью выключенном
    сервере.
    """
    global _live
    await ws.accept()
    log.info("клиент подключился: %s", ws.client)
    loop = asyncio.get_event_loop()

    # 1) клиент может продолжить прежнюю сессию (после обрыва связи).
    # Первое сообщение — resume/mode-команда либо уже аудио (его буферизуем).
    resume_id: int | None = None
    pending_mode: str | None = None
    pending_bytes: list[bytes] = []
    first_msg = await ws.receive()
    if first_msg.get("type") == "websocket.disconnect":
        return
    if first_msg.get("bytes") is not None:
        pending_bytes.append(first_msg["bytes"])
    elif first_msg.get("text"):
        try:
            first = json.loads(first_msg["text"])
            if first.get("cmd") == "resume":
                resume_id = int(first.get("session_id") or 0) or None
            elif first.get("cmd") == "mode":
                pending_mode = str(first.get("mode") or "") or None
            elif first.get("cmd") == "app_mode":
                # канал мог открыться вне записи (режимы «Компьютер»/«Браузер») —
                # режим применяем сразу, сессию при этом не создаём
                a = str(first.get("app") or "recap")
                app_mode = a if a in ("recap", "computer", "browser") else "recap"
                if app_mode != "computer":
                    computer_control.get_agent().reset()
                await ws.send_json({"type": "app_mode", "app": app_mode})
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    if resume_id is not None and _live is not None and _live.session_id == resume_id:
        sid = resume_id
        reopen_session(sid)
        if pending_mode is not None:
            set_mode(sid, pending_mode)
        await ws.send_json({"type": "session_resumed", "session_id": sid})
        log.info("возобновляю сессию #%d", sid)
    else:
        # прежняя живая сессия (если была) — закрываем как прерванную
        if _live is not None:
            end_session(_live.session_id)
            _live.close_wav()
        sid = create_session(pending_mode)
        _live = _Live(sid)
        await ws.send_json({"type": "session_started", "session_id": sid})

    live = _live

    async def transcribe_worker():
        """Выбирает фразы из очереди и транскрибирует в executor'е."""
        L = live  # фиксируем сессию: после stop live станет None,
                  # а накопленные фразы всё равно должны обработаться
        while True:
            phrase = await L.q.get()
            try:
                spk = await loop.run_in_executor(None, L.speaker.identify, phrase)
                text = await loop.run_in_executor(None, whisper.transcribe, phrase)
                if text:
                    log.info("транскрипт: %r (говорящий %d)", text[:80], spk + 1)
                    llm.add(text)
                    add_phrase(L.session_id, text, speaker=spk)
                    # семантическая память: фраза → эмбеддинг → векторная база.
                    # эмбеддинг — блокирующий HTTP, поэтому тоже в executor:
                    # память не должна тормозить приём микрофона
                    mem = _mem()
                    if mem is not None:
                        try:
                            await loop.run_in_executor(
                                None, mem.add_phrase, L.session_id, text, spk
                            )
                        except Exception:
                            log.exception("память: не удалось сохранить фразу")
                    await ws.send_json(
                        {"type": "transcript", "text": text, "speaker": spk}
                    )
                else:
                    log.debug("фраза пустая (текста нет)")
            except WebSocketDisconnect:
                raise
            except Exception:
                # ошибка одной фразы не должна ронять воркер — запись продолжается
                log.exception("ошибка транскрипции фразы — пропускаю")
            finally:
                L.q.task_done()

    worker = asyncio.create_task(transcribe_worker())

    async def drain_phrases(timeout: float = 600.0):
        """Дождаться обработки накопленных фраз (для stop)."""
        try:
            await asyncio.wait_for(live.q.join(), timeout)
        except asyncio.TimeoutError:
            log.warning("ожидание транскрипции прервано по таймауту")

    async def handle_audio(raw_bytes: bytes):
        if live is None:  # сессию уже закрыли (stop) — игнорируем хвост
            return
        # PCM int16 → float32; пишем на диск ДО разбора VAD,
        # чтобы исходник записи сохранился даже при ошибках выше
        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        if len(pcm) == 0:
            return
        live.n_chunks += 1
        try:
            live.wav.writeframes(raw_bytes)
        except Exception:
            log.exception("не удалось дописать WAV — запись продолжается")
        if live.n_chunks % 100 == 0:
            log.info("аудио #%d: %.0f сек записано",
                     live.session_id, live.n_chunks * 0.1)
        chunk = pcm.astype(np.float32) / 32768.0
        phrase = live.asm.add(chunk)
        if phrase is not None:
            # очередь полная (Whisper отстаёт) — ждём, а НЕ выбрасываем:
            # потеря фразы хуже, чем задержка показа на экране
            await live.q.put(phrase)

    async def finish_session():
        """Flush + закрытие сессии + итоги."""
        nonlocal live
        global _live
        if live is None:
            return
        sid = live.session_id
        phrase = live.asm.flush()
        if phrase is not None:
            await live.q.put(phrase)
        await drain_phrases()
        live.close_wav()
        end_session(sid)
        txt = session_text(sid)
        if txt:
            llm.add("[Сессия закрыта. Генерирую итоги.]")
            summary = await loop.run_in_executor(
                None, llm.generate_summary, sid, txt, get_mode(sid)
            )
            log.info("итоги: %r", summary[:100])
            set_summary(sid, summary)
            # итоги — самые ценные чанки памяти («что решали в прошлый раз»);
            # сохраняем одним куском, поиск по ним точнее
            mem = _mem()
            if mem is not None:
                try:
                    await loop.run_in_executor(None, mem.add_summary, sid, summary)
                except Exception:
                    log.exception("память: не удалось сохранить итоги")
            await ws.send_json(
                {"type": "summary", "text": summary, "session_id": sid,
                 "mode": get_mode(sid) or ""}
            )
        await ws.send_json({"type": "session_closed", "session_id": sid})

    try:
        # аудио, пришедшее до команды resume — не выбрасываем
        for b in pending_bytes:
            await handle_audio(b)
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            raw_bytes = msg.get("bytes")
            if raw_bytes is not None:
                await handle_audio(raw_bytes)
                continue
            raw = msg.get("text")
            if raw is None:
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            cmd = data.get("cmd")
            if cmd == "app_mode":
                # режим приложения: recap (запись встреч) | computer | browser
                app_mode = str(data.get("app") or "recap")
                if app_mode not in ("recap", "computer", "browser"):
                    app_mode = "recap"
                if app_mode != "computer":
                    computer_control.get_agent().reset()
                log.info("режим приложения: %s", app_mode)
                await ws.send_json({"type": "app_mode", "app": app_mode})
            elif cmd == "ask":
                q = data["q"]
                log.info("вопрос (%s): %r", app_mode, q[:80])
                if app_mode == "computer":
                    # tool-цикл qwen3: команды терминала → результат → ответ
                    await ws.send_json({"type": "thinking",
                                        "text": "выполняю на компьютере…"})
                    res = await loop.run_in_executor(
                        None, computer_control.get_agent().turn, q)
                    for ev in res["events"]:
                        await ws.send_json({"type": "cmd_event", **ev})
                    await ws.send_json({"type": "answer", "text": res["answer"]})
                elif app_mode == "browser":
                    # задача автономному агенту в Chrome (ai_browser, qwen3)
                    async def br_progress(text: str):
                        try:
                            await ws.send_json({"type": "progress", "text": text})
                        except Exception:
                            pass  # клиент мог отключиться — агент продолжит

                    async def br_finish(reason: str):
                        try:
                            await ws.send_json({"type": "answer", "text": reason})
                        except Exception:
                            pass

                    ok = await browser_engine.get_bridge().start_task(
                        q, br_progress, br_finish)
                    if not ok:
                        await ws.send_json({"type": "answer", "text":
                            "Браузер недоступен: ai_browser не поднялся или "
                            "задача уже выполняется. Подробности в логе сервера."})
                else:
                    # llm.ask сам извлекает контекст из памяти (RAG):
                    # прошлые беседы дополняют скользящее окно транскрипта
                    ans = await loop.run_in_executor(None, llm.ask, q)
                    log.info("ответ LLM: %r", ans[:80])
                    await ws.send_json({"type": "answer", "text": ans})
            elif cmd == "browser_stop":
                await browser_engine.get_bridge().stop()
                await ws.send_json({"type": "answer",
                                    "text": "Останавливаю задачу в браузере."})
            elif cmd == "browser_input":
                # данные для агента (SMS-код, пароль): wait_user ждёт их
                await browser_engine.get_bridge().user_input(
                    str(data.get("value") or ""))
                await ws.send_json({"type": "answer",
                                    "text": "Данные переданы агенту."})
            elif cmd == "hint":
                h = await loop.run_in_executor(None, llm.hint)
                log.info("подсказка LLM: %r", h[:80])
                await ws.send_json({"type": "hint", "text": h})
            elif cmd == "stop":
                log.info("стоп-команда, завершаю сессию #%d",
                         live.session_id if live else -1)
                await finish_session()
                live = None
                _live = None
            elif cmd == "resume":
                sid = int(data.get("session_id") or 0)
                if _live is not None and _live.session_id == sid:
                    reopen_session(sid)
                    await ws.send_json({"type": "session_resumed", "session_id": sid})
            elif cmd == "mode":
                # сменить сценарий можно и в середине записи
                m = str(data.get("mode") or "") or None
                if live is not None:
                    set_mode(live.session_id, m)
    except WebSocketDisconnect:
        log.info("клиент отключился (сессия #%d остаётся открытой для resume)",
                 live.session_id if live else -1)
    finally:
        worker.cancel()
        # при обрыве связи сессию НЕ закрываем и WAV НЕ выбрасываем:
        # клиент переподключится и продолжит ту же запись


# ---- REST API для сессий ----

@app.get("/api/sessions")
async def api_list_sessions(limit: int = 50):
    return list_sessions(limit)


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: int):
    return get_session(session_id) or {}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: int):
    delete_session(session_id)
    # удаляем и векторные записи: «удалённая» сессия не должна
    # всплывать в контексте памяти
    mem = _mem()
    if mem is not None:
        try:
            mem.delete_session(session_id)
        except Exception:
            log.exception("память: не удалось удалить записи сессии #%d", session_id)
    return {"ok": True}


@app.patch("/api/sessions/{session_id}")
async def api_rename_session(session_id: int, body: dict):
    rename_session(session_id, body.get("title", ""))
    return {"ok": True}


@app.post("/api/sessions/{session_id}/mode")
async def api_set_mode(session_id: int, body: dict):
    """Сменить сценарий сессии (влияет на фокус итогов)."""
    set_mode(session_id, str(body.get("mode") or "") or None)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/summary")
async def api_summarize_session(session_id: int):
    """Перегенерировать итоги для уже закрытой сессии."""
    txt = session_text(session_id)
    if not txt:
        return {"error": "сессия пустая или не найдена"}
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(
        None, llm.generate_summary, session_id, txt, get_mode(session_id)
    )
    set_summary(session_id, summary)
    mem = _mem()
    if mem is not None:
        try:
            await loop.run_in_executor(None, mem.add_summary, session_id, summary)
        except Exception:
            log.exception("память: не удалось обновить итоги в памяти")
    return {"summary": summary}


# ---- REST API для семантической памяти (RAG) ----

@app.get("/api/memory/search")
async def api_memory_search(q: str, n: int = 5,
                            since: float | None = None,
                            until: float | None = None):
    """Поиск по прошлым беседам: ближайшие чанки по смыслу.

    since/until — unix-время (для «что мы решали в прошлый вторник»).
    """
    mem = _mem()
    if mem is None:
        return {"error": "память недоступна (ChromaDB/Ollama-эмбеддинг)"}
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, mem.search, q, n, since, until)
    except Exception as e:
        log.exception("память: поиск не удался")
        return {"error": str(e)}


@app.get("/api/memory/stats")
async def api_memory_stats():
    mem = _mem()
    if mem is None:
        return {"error": "память недоступна (ChromaDB/Ollama-эмбеддинг)"}
    return mem.stats()


@app.post("/api/memory/reindex/{session_id}")
async def api_memory_reindex(session_id: int):
    """Заполнить память для сессии, записанной до включения RAG
    (или после сбоя эмбеддинга)."""
    sess = get_session(session_id)
    if not sess:
        return {"error": "сессия не найдена"}
    mem = _mem()
    if mem is None:
        return {"error": "память недоступна (ChromaDB/Ollama-эмбеддинг)"}
    phrases = [
        (p.get("text", ""), p.get("speaker", 0))
        for p in (sess.get("phrases") or [])
    ]
    if not phrases:
        return {"error": "в сессии нет фраз"}
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None, mem.reindex_session, session_id, phrases, sess.get("summary")
        )
    except Exception as e:
        log.exception("память: переиндексация не удалась")
        return {"error": str(e)}
    return {"ok": True, "phrases": len(phrases)}


_NOCACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}


# ─── жизненный цикл сервисов браузера ────────────────────────────

@app.on_event("startup")
async def _startup_services():
    """Поднимаем Chrome CDP и ai_browser одновременно с сервером."""
    await svc_mgr.start_all()


@app.on_event("shutdown")
async def _shutdown_services():
    """Останавливаем браузерный агент и всё, что подняли сами."""
    try:
        await browser_engine.get_bridge().shutdown()
    except Exception:
        pass
    svc_mgr.stop_all()
    log.info("сервисы браузера остановлены")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html", headers=_NOCACHE)


@app.get("/static/{path:path}")
async def static(path: str):
    return FileResponse(f"static/{path}", headers=_NOCACHE)
