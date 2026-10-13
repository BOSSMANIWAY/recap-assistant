"""
Chat Store — сохранение истории чатов и переключение между ними.

Каждый чат = один JSON-файл в каталоге ai_browser_chats/: id, заголовок,
метки времени и список сообщений ровно в том виде, в каком они показываются
в UI (role/type/text + служебные поля). Формат плоский и читаемый руками —
без БД и внешних зависимостей, чтобы чаты можно было переносить/бэкапить копированием.

Заголовок чата берётся из первого сообщения пользователя: так список чатов
осмысленный без отдельного шага «дать имя».
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CHATS_DIR = Path("ai_browser_chats")
MAX_TITLE = 70


def _safe_id(chat_id: str) -> str:
    """id чата попадает в имя файла — оставляем только безопасные символы."""
    cleaned = "".join(c for c in str(chat_id) if c.isalnum() or c in "-_")[:64]
    if not cleaned:
        raise ValueError(f"Некорректный id чата: {chat_id!r}")
    return cleaned


class ChatStore:
    """Хранилище чатов: создание, дописывание сообщений, список, переключение."""

    def __init__(self, directory: Optional[Path] = None):
        self.dir = Path(directory) if directory else DEFAULT_CHATS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    # ─── Путь/метаданные ────────────────────────────────────────────

    def _path(self, chat_id: str) -> Path:
        return self.dir / f"{_safe_id(chat_id)}.json"

    def load(self, chat_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(chat_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Чат {chat_id} повреждён и не читается: {e}")
            return None
        if not isinstance(data, dict):
            return None
        data.setdefault("messages", [])
        return data

    def save(self, chat: Dict[str, Any]) -> Dict[str, Any]:
        """Сохранить чат целиком (идемпотентно: перезапись файла)."""
        chat_id = _safe_id(chat.get("id") or uuid.uuid4().hex[:12])
        chat["id"] = chat_id
        chat["updated_at"] = time.time()
        chat.setdefault("created_at", chat["updated_at"])
        chat.setdefault("title", "")
        if not chat.get("title"):
            chat["title"] = self._title_from(chat.get("messages") or [])
        tmp = self._path(chat_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(chat, ensure_ascii=False, indent=1), encoding="utf-8")
        # Запись через tmp+replace: при обрыве процесса файл не останется пустым
        tmp.replace(self._path(chat_id))
        return chat

    @staticmethod
    def _title_from(messages: List[Dict[str, Any]]) -> str:
        """Заголовок = первое сообщение пользователя (первая строка, без переносов)."""
        for m in messages:
            if m.get("role") == "user" and (m.get("text") or "").strip():
                first = (m["text"].strip().splitlines() or [""])[0]
                return first[:MAX_TITLE]
        return "Без названия"

    # ─── Сообщения ──────────────────────────────────────────────────

    def create(self, first_message: str = "", chat_id: str = "") -> Dict[str, Any]:
        """Новый чат; первая реплика пользователя задаёт заголовок."""
        chat = {
            "id": _safe_id(chat_id) if chat_id else uuid.uuid4().hex[:12],
            "title": "",
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
        }
        if first_message.strip():
            chat["messages"].append(self.msg("user", first_message.strip()))
            chat["title"] = self._title_from(chat["messages"])
        return self.save(chat)

    @staticmethod
    def msg(role: str, text: str, **extra: Any) -> Dict[str, Any]:
        """Одно сообщение чата. extra — служебные поля UI (task_id, kind, files)."""
        m: Dict[str, Any] = {"role": role, "text": text, "ts": time.time()}
        m.update({k: v for k, v in extra.items() if v is not None})
        return m

    def append(self, chat_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Дописать сообщение в существующий чат (самый частый вызов)."""
        chat = self.load(chat_id)
        if chat is None:
            return None
        chat.setdefault("messages", []).append(message)
        if not chat.get("title"):
            chat["title"] = self._title_from(chat["messages"])
        return self.save(chat)

    def set_messages(self, chat_id: str, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Заменить историю целиком — для сохранения состояния открытого чата."""
        chat = self.load(chat_id)
        if chat is None:
            return None
        chat["messages"] = messages
        return self.save(chat)

    def rename(self, chat_id: str, title: str) -> Optional[Dict[str, Any]]:
        chat = self.load(chat_id)
        if chat is None:
            return None
        chat["title"] = (title or "").strip()[:MAX_TITLE] or self._title_from(chat.get("messages") or [])
        return self.save(chat)

    # ─── Список / удаление ──────────────────────────────────────────

    def list(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Краткий список чатов (без текста сообщений) — для сайдбара, новые сверху."""
        items: List[Dict[str, Any]] = []
        for path in self.dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict) or "id" not in data:
                continue
            msgs = data.get("messages") or []
            items.append({
                "id": data["id"],
                "title": data.get("title") or self._title_from(msgs),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "count": len(msgs),
                # Превью для подсказки при наведении — только первая реплика
                "preview": next((m.get("text", "")[:120] for m in msgs
                                 if m.get("role") == "user"), ""),
            })
        items.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
        return items[:limit]

    def delete(self, chat_id: str) -> bool:
        path = self._path(chat_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def stats(self) -> Dict[str, Any]:
        files = list(self.dir.glob("*.json"))
        return {"chats": len(files), "dir": str(self.dir.resolve()),
                "size": sum(f.stat().st_size for f in files)}
