"""Жизненный цикл внешних сервисов браузера.

Один вход — `.venv/bin/uvicorn main:app --port 8099`: при старте поднимаем
Chrome с CDP (:9222) и ai_browser (:8765), при Ctrl+C — убиваем всё, что
запустили сами.
"""
import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("assist.services")

ROOT = Path(__file__).resolve().parent
BROWSER_DIR = ROOT / "ai_browser"
PROFILE_DIR = BROWSER_DIR / "ai_browser_profile"
CDP_PORT = 9222
BROWSER_PORT = 8765
CHROME_APP = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        s = socket.socket()
        s.settimeout(1)
        ok = s.connect_ex((host, port)) == 0
        s.close()
        return ok
    except OSError:
        return False


def find_chrome() -> str:
    if os.path.exists(CHROME_APP):
        return CHROME_APP
    for p in os.environ.get("PATH", "").split(os.pathsep):
        for name in ("google-chrome", "chrome", "chromium"):
            cand = os.path.join(p, name)
            if os.path.exists(cand):
                return cand
    return ""


class ServiceManager:
    def __init__(self):
        self.procs: list[subprocess.Popen] = []

    async def start_all(self) -> None:
        """Только Chrome CDP: браузерный агент теперь живёт в этом же
        процессе (browser_engine), отдельный сервер ai_browser не нужен."""
        try:
            await self._start_chrome()
        except Exception as e:
            log.warning("автозапуск сервисов не завершён: %s", e)

    def stop_all(self) -> None:
        for proc in reversed(self.procs):
            self._terminate(proc)
        self.procs.clear()

    def _spawn(self, argv: list[str], cwd: Path, port: int, label: str):
        proc = subprocess.Popen(
            argv, cwd=str(cwd),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.procs.append(proc)
        log.info("%s запущен (pid=%s)", label, proc.pid)
        return proc

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except Exception:
                return
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()

    async def _wait_port(self, port: int, seconds: int, label: str) -> bool:
        for _ in range(seconds):
            if port_open(port):
                return True
            await asyncio.sleep(1)
        log.warning("%s не поднял порт %d за %d сек", label, port, seconds)
        return False

    async def _start_chrome(self) -> None:
        if port_open(CDP_PORT):
            log.info("Chrome CDP уже на :%d — не трогаю", CDP_PORT)
            return
        chrome = find_chrome()
        if not chrome:
            log.warning("Chrome не найден — браузерный режим будет недоступен")
            return

        if self._chrome_running_without_cdp():
            log.info("Chrome запущен без CDP — закрываю для перезапуска с :%d", CDP_PORT)
            await self._quit_chrome()

        PROFILE_DIR.mkdir(exist_ok=True)
        for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
            (PROFILE_DIR / name).unlink(missing_ok=True)

        self._spawn(
            [chrome,
             f"--user-data-dir={PROFILE_DIR}",
             f"--remote-debugging-port={CDP_PORT}",
             "--no-first-run", "--no-default-browser-check"],
            cwd=BROWSER_DIR, port=CDP_PORT, label="Chrome CDP",
        )
        if await self._wait_port(CDP_PORT, 20, "Chrome"):
            log.info("Chrome CDP готов на :%d", CDP_PORT)

    def _chrome_running_without_cdp(self) -> bool:
        try:
            r = subprocess.run(["pgrep", "-x", "Google Chrome"],
                               capture_output=True)
            return r.returncode == 0
        except OSError:
            return False

    async def _quit_chrome(self) -> None:
        try:
            subprocess.run(["osascript", "-e",
                            'tell application "Google Chrome" to quit'],
                           timeout=10)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("не удалось штатно закрыть Chrome: %s", e)
        for _ in range(10):
            if not self._chrome_running_without_cdp():
                return
            await asyncio.sleep(1)


_manager: ServiceManager | None = None


def get_manager() -> ServiceManager:
    global _manager
    if _manager is None:
        _manager = ServiceManager()
    return _manager
