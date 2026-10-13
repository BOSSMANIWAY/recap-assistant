"""Локальный браузерный агент: Chrome CDP + AgentLoop без отдельного сервера.

Раньше путь был: main.py → browser_bridge (WS) → ai_browser:8765 → Chrome.
Теперь всё в одном процессе: main.py → browser_engine → Chrome CDP :9222.

Интерфейс совместим с прежним BrowserBridge (start_task/stop/user_input),
поэтому main.py менять не нужно.
"""
import asyncio
import logging

from agent.loop import AgentLoop
from agent.llm_client import LLMClient
from agent.memory import AgentMemory
from browser.controller import BrowserController

log = logging.getLogger("assist.browser")

PROVIDER = "ollama"
MODEL = "qwen3:8b"
MAX_STEPS = 40


class BrowserEngine:
    """Один контроллер и одна задача на приложение (как было в web_backend)."""

    def __init__(self):
        self.controller: BrowserController | None = None
        self.loop: AgentLoop | None = None
        self.task_active = False
        self.cb_progress = None
        self.cb_finish = None
        self._lock = asyncio.Lock()

    async def _ensure_controller(self) -> bool:
        """Подключиться к Chrome CDP (или поднять managed Chrome)."""
        if self.controller is not None and self.controller.connected:
            return True
        # Мёртвый контроллер мог оставить playwright-процесс — закрываем.
        if self.controller is not None:
            try:
                await self.controller.disconnect()
            except Exception:
                pass
            self.controller = None
        ctrl = BrowserController()
        try:
            # mode="auto": attach к CDP :9222, при неудаче — managed Chrome.
            await ctrl.connect()
        except Exception as e:
            log.error("не удалось подключиться к Chrome: %s", str(e)[:200])
            return False
        self.controller = ctrl
        log.info("браузер подключён (CDP %s)",
                 "attach" if not ctrl._is_persistent else "managed")
        return True

    async def start_task(self, prompt: str, cb_progress, cb_finish) -> bool:
        if not prompt.strip():
            return False
        async with self._lock:
            if self.task_active:
                return False
            if not await self._ensure_controller():
                return False
            self.cb_progress = cb_progress
            self.cb_finish = cb_finish
            self.task_active = True

            async def on_progress(text: str):
                if self.cb_progress:
                    await self.cb_progress(str(text))

            async def on_finish(reason: str):
                finished = self.task_active
                self.task_active = False
                if finished and self.cb_finish:
                    await self.cb_finish(str(reason))

            llm = LLMClient(provider=PROVIDER, model=MODEL)
            self.loop = AgentLoop(
                controller=self.controller,
                memory=AgentMemory(),
                llm_client=llm,
                max_steps=MAX_STEPS,
                model=llm.model,
                on_progress=on_progress,
                on_finish=on_finish,
            )
            await self._progress(f"Задача: {prompt}")
            asyncio.create_task(self._run(prompt))
            return True

    async def _run(self, prompt: str):
        try:
            await self.loop.run(prompt)
        except Exception as e:
            log.exception("задача браузера упала")
            if self.task_active:
                self.task_active = False
                if self.cb_finish:
                    try:
                        await self.cb_finish(f"ошибка: {e}")
                    except Exception:
                        pass

    async def _progress(self, text: str):
        if self.cb_progress:
            try:
                await self.cb_progress(text)
            except Exception:
                pass

    async def stop(self):
        if self.loop is not None:
            try:
                self.loop.stop()
            except Exception:
                log.warning("browser: stop не доставлен")

    async def user_input(self, value: str):
        if self.loop is not None:
            try:
                await self.loop.provide_user_input(value)
            except Exception:
                log.warning("browser: user_input не доставлен")

    async def shutdown(self):
        """Остановка при завершении сервера: задача + соединение с Chrome."""
        try:
            if self.loop is not None:
                self.loop.stop()
            if self.controller is not None:
                await self.controller.disconnect()
        except Exception:
            pass
        self.controller = None
        self.task_active = False


_engine: BrowserEngine | None = None


def get_bridge() -> BrowserEngine:
    """Имя прежнее — main.py работает без правок."""
    global _engine
    if _engine is None:
        _engine = BrowserEngine()
    return _engine
