"""
Agent Memory - Память сессии агента
Хранит историю действий, контекст, состояние
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentMemory:
    """Память агента сессии"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {
            "current_url": None,
            "current_title": None,
            "actions_taken": [],
            "pages_visited": [],
            "forms_filled": [],
            "tabs_opened": [],
            "errors": [],
            "goals": [],
            "completed_goals": [],
        }
        self._storage_path = Path(f"ai_browser_sessions/{self.session_id}")
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def add_action(self, action: str, details: Dict[str, Any] = None):
        """Добавить действие в историю"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details or {},
        }
        self.history.append(entry)
        self.context["actions_taken"].append(action)
        logger.debug(f"Action logged: {action}")

    def add_step(self, tool: str, plan: List[str], thought: str,
                 action: Dict[str, Any], result: Dict[str, Any],
                 verified: str = "", progress: Optional[bool] = None,
                 target: str = ""):
        """Добавить шаг с планом — для показа модели на следующих шагах.
        Сохраняет plan, params действия, результат (detail/error/ok),
        verified-строку и verdict progress. Это нужно, чтобы model
        видела историю: что делала, что получилось, а не пустоту.
        target — подпись элемента («button «Start»»): element_id пересоздаются
        на каждом снимке, и без подписи два разных клика выглядят в истории
        как один и тот же «click e1»."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": tool,
            "details": {
                "plan": plan,
                "thought": thought,
                "params": {k: v for k, v in action.items()
                           if k not in ("tool", "thought", "plan")},
                "target": target,
                "result": result,
                "verified": verified,
                "progress": progress,
            },
        }
        self.history.append(entry)
        self.context["actions_taken"].append(tool)
        logger.debug(f"Step logged: {tool} plan={len(plan)} пунктов result_ok={result.get('ok')}")

    def add_page_visit(self, url: str, title: str):
        """Записать посещение страницы"""
        self.context["pages_visited"].append({
            "url": url,
            "title": title,
            "timestamp": datetime.now().isoformat(),
        })

    def add_goal(self, goal: str):
        """Добавить цель"""
        self.context["goals"].append(goal)

    def complete_goal(self, goal: str):
        """Отметить цель как выполненную"""
        if goal in self.context["goals"]:
            self.context["goals"].remove(goal)
            self.context["completed_goals"].append({
                "goal": goal,
                "completed_at": datetime.now().isoformat(),
            })

    def add_error(self, error: str, details: Dict = None):
        """Записать ошибку"""
        self.context["errors"].append({
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })

    def update_context(self, url: str, title: str):
        """Обновить текущий контекст"""
        self.context["current_url"] = url
        self.context["current_title"] = title

    def get_summary(self) -> str:
        """Получить краткую сводку сессии"""
        lines = [
            f" Сессия: {self.session_id}",
            f" URL: {self.context['current_url'] or 'N/A'}",
            f" Заголовок: {self.context['current_title'] or 'N/A'}",
            f" Действий: {len(self.history)}",
            f" Посещено страниц: {len(self.context['pages_visited'])}",
            f" Целей: {len(self.context['goals'])} активных, "
            f"{len(self.context['completed_goals'])} выполнено",
            f" Ошибок: {len(self.context['errors'])}",
        ]

        if self.context["goals"]:
            lines.append("\n Цели:")
            for g in self.context["goals"]:
                lines.append(f"   - {g}")

        if self.context["errors"]:
            lines.append("\n Ошибки:")
            for e in self.context["errors"][-5:]:  # последние 5
                lines.append(f"   - {e['error']}")

        return "\n".join(lines)

    def get_context_for_llm(self) -> str:
        """Контекст для отправки LLM"""
        lines = [
            "=== КОНТЕКСТ СЕССИИ ===",
            f"URL: {self.context['current_url']}",
            f"Заголовок: {self.context['current_title']}",
            f"Посещено страниц: {len(self.context['pages_visited'])}",
        ]

        if self.context["pages_visited"]:
            lines.append("\n Посещённые страницы:")
            for p in self.context["pages_visited"][-10:]:
                lines.append(f"  - {p['title']}: {p['url']}")

        if self.context["goals"]:
            lines.append("\n Активные цели:")
            for g in self.context["goals"]:
                lines.append(f"  - {g}")

        if self.context["completed_goals"]:
            lines.append("\n Выполненные цели:")
            for g in self.context["completed_goals"]:
                lines.append(f"  - {g['goal']}")

        if self.context["errors"]:
            lines.append("\n Последние ошибки:")
            for e in self.context["errors"][-3:]:
                lines.append(f"  - {e['error']}")

        lines.append("\n=== КОНЕЦ КОНТЕКСТА ===")
        return "\n".join(lines)

    def save(self):
        """Сохранить память на диск"""
        data = {
            "session_id": self.session_id,
            "history": self.history,
            "context": self.context,
            "saved_at": datetime.now().isoformat(),
        }
        path = self._storage_path / "memory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Memory saved to {path}")

    def load(self):
        """Загрузить память с диска"""
        path = self._storage_path / "memory.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = data.get("history", [])
            self.context.update(data.get("context", {}))
            logger.info(f"Memory loaded from {path}")
            return True
        return False

    def save_history(self):
        """Сохранить полную историю действий"""
        path = self._storage_path / "history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def get_recent_actions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Получить последние N действий"""
        return self.history[-n:] if self.history else []
