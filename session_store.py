"""Хранение сессий транскрипции в SQLite.

Каждая запись с микрофона — сессия: время, реплики с таймстампами,
позже — сгенерированные итоги встречи.
"""
import logging
import os
import sqlite3
import threading
import time

log = logging.getLogger("assist.store")

DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    """Соединение одно: доступ идёт из потоков executor'а,
    поэтому check_same_thread=False + блокатор."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at REAL NOT NULL,
                ended_at REAL,
                title TEXT,
                summary TEXT
            )"""
        )
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                text TEXT NOT NULL,
                speaker INTEGER DEFAULT 0
            )"""
        )
        # миграция старой БД: колонка speaker появилась позже
        cols = [r["name"] for r in _conn.execute("PRAGMA table_info(phrases)")]
        if "speaker" not in cols:
            _conn.execute(
                "ALTER TABLE phrases ADD COLUMN speaker INTEGER DEFAULT 0"
            )
            log.info("миграция: добавлена колонка phrases.speaker")
        # миграция: сценарий чата (шаблон промта) — влияет на итоги встречи
        scols = [r["name"] for r in _conn.execute("PRAGMA table_info(sessions)")]
        if "mode" not in scols:
            _conn.execute("ALTER TABLE sessions ADD COLUMN mode TEXT")
            log.info("миграция: добавлена колонка sessions.mode")
        _conn.commit()
    return _conn


def create_session(mode: str | None = None) -> int:
    with _lock:
        cur = _db().execute(
            "INSERT INTO sessions (started_at, mode) VALUES (?, ?)",
            (time.time(), mode),
        )
        _db().commit()
        log.info("сессия #%d начата (mode=%r)", cur.lastrowid, mode)
        return cur.lastrowid


def set_mode(session_id: int, mode: str | None):
    """Сменить сценарий чата (влияет на фокус итогов)."""
    with _lock:
        _db().execute(
            "UPDATE sessions SET mode=? WHERE id=?",
            ((mode or "")[:60] or None, session_id),
        )
        _db().commit()
        log.info("сессия #%d: режим %r", session_id, mode)


def get_mode(session_id: int) -> str | None:
    with _lock:
        r = _db().execute(
            "SELECT mode FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return r["mode"] if r else None


def add_phrase(session_id: int, text: str, speaker: int = 0):
    with _lock:
        db = _db()
        db.execute(
            "INSERT INTO phrases (session_id, ts, text, speaker) VALUES (?, ?, ?, ?)",
            (session_id, time.time(), text, speaker),
        )
        # заголовок сессии — первая реплика
        n = db.execute(
            "SELECT COUNT(*) FROM phrases WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        if n == 1:
            db.execute("UPDATE sessions SET title=? WHERE id=?", (text[:80], session_id))
        db.commit()


def end_session(session_id: int):
    with _lock:
        _db().execute("UPDATE sessions SET ended_at=? WHERE id=?", (time.time(), session_id))
        _db().commit()
        log.info("сессия #%d закрыта", session_id)


def reopen_session(session_id: int):
    """Клиент переподключился — сессия продолжается, снимаем метку конца."""
    with _lock:
        _db().execute("UPDATE sessions SET ended_at=NULL WHERE id=?", (session_id,))
        _db().commit()
        log.info("сессия #%d возобновлена (переподключение)", session_id)


def list_sessions(limit: int = 50) -> list[dict]:
    with _lock:
        rows = _db().execute(
            """SELECT s.id, s.started_at, s.ended_at, s.title, s.summary, s.mode,
                      COUNT(p.id) AS n_phrases
               FROM sessions s LEFT JOIN phrases p ON p.session_id = s.id
               GROUP BY s.id
               HAVING COUNT(p.id) > 0 OR s.summary IS NOT NULL
               ORDER BY s.started_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session(session_id: int) -> dict | None:
    with _lock:
        db = _db()
        s = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if s is None:
            return None
        phrases = db.execute(
            "SELECT ts, text, speaker FROM phrases WHERE session_id=? ORDER BY ts",
            (session_id,),
        ).fetchall()
        out = dict(s)
        out["phrases"] = [dict(p) for p in phrases]
        return out


def delete_session(session_id: int):
    with _lock:
        db = _db()
        db.execute("DELETE FROM phrases WHERE session_id=?", (session_id,))
        db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        db.commit()
        log.info("сессия #%d удалена", session_id)


def set_summary(session_id: int, text: str):
    with _lock:
        _db().execute("UPDATE sessions SET summary=? WHERE id=?", (text, session_id))
        _db().commit()


def rename_session(session_id: int, title: str):
    with _lock:
        _db().execute("UPDATE sessions SET title=? WHERE id=?", (title.strip()[:200], session_id))
        _db().commit()
        log.info("сессия #%d переименована в %r", session_id, title[:40])


def session_text(session_id: int) -> str:
    """Весь текст сессии — для генерации итогов (с метками говорящих)."""
    with _lock:
        rows = _db().execute(
            "SELECT text, speaker FROM phrases WHERE session_id=? ORDER BY ts",
            (session_id,),
        ).fetchall()
    parts = []
    for r in rows:
        if r["speaker"]:
            parts.append(f"[Говорящий {r['speaker'] + 1}] {r['text']}")
        else:
            parts.append(r["text"])
    return "\n".join(parts)
