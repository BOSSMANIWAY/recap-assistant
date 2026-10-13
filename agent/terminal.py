"""
Terminal Executor — «руки» агента за пределами браузера.

Зачем: браузерный агент видит только DOM. Многие задачи (настроить сервер,
отдать webhook-ответ, проверить хост, скачать файл) требуют shell. Здесь
агент получает два способа выполнить работу:

1. command   — прямое выполнение shell-команды (быстро, предсказуемо).
2. prompt    — делегирование второму агенту: `koda -y -p "<prompt>" -m <model>`.
   Koda — CLI-агент со своими инструментами (файлы, shell, сеть). Через него
   основной агент (Perplexity) решает многошаговые терминальные задачи, не
   зная синтаксиса каждой утилиты: формулирует цель, Koda разбирается сам.

Отсюда и «два агента в одном решении»: агент в браузере читает данные со
страницы (например Webhook Inspector), составляет по ним задание и передаёт
его в терминал; результат возвращается в контекст браузера.

Безопасность: запуск вне песочницы, поэтому есть deny-список разрушительных
команд, жёсткие таймауты и ограничение объёма вывода. Фича отключается одним
флагом TERMINAL_ENABLED=0 — тогда tool terminal возвращает ошибку и модель
работает только браузером.
"""

import asyncio
import logging
import os
import re
import shutil
import signal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Команды, которые не выполняем никогда: необратимое уничтожение данных,
# повреждение диска/системы, обход прав, pipes на исполнение из сети.
DENY_PATTERNS: List[tuple] = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*(/|/\*|~|\$HOME)(\s|$)",
     "рекурсивное удаление корня/домашнего каталога"),
    (r"\bsudo\b|\bdoas\b|\bsu\s+-", "повышение привилегий"),
    (r"\bmkfs(\.\w+)?\b|\bfdisk\b|\bgdisk\b|\bdiskutil\s+erase",
     "разметка/стирание диска"),
    (r"\bdd\b[^\n]*\bof=/dev/", "запись сырых данных на устройство"),
    (r">\s*/dev/(sd|disk|rdisk|nvme)", "перезапись блочного устройства"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "выключение/перезагрузка машины"),
    (r":\(\)\s*\{[^}]*\|\s*[^}]*&[^}]*\}", "fork-bomb"),
    (r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/(\s|$)", "массовая смена прав на корне"),
    (r"(curl|wget)[^\n|;&]*\|\s*(sudo\s+)?(ba|z|k|da)?sh\b",
     "исполнение скрипта из сети без проверки"),
    (r"\bgit\s+push\b[^\n]*(--force|-f)\b", "force-push (уничтожает историю удалённого репозитория)"),
    (r"\bhistory\s+-c\b", "затирание истории оболочки"),
]

# Не блокируем, но предупреждаем: легитимные, но необратимые действия.
WARN_PATTERNS: List[tuple] = [
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f)\b",
     "git сбросит незакоммиченные изменения"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*-?[rR]", "рекурсивное удаление файлов"),
    (r"\bkillall\b|\bpkill\b", "завершение процессов"),
    (r"(cat|head|tail|less|grep)\s+[^\n]*(\.ssh|\.aws|\.env|id_rsa|\.netrc|keychain)",
     "вывод возможных секретов — не переноси их в браузер и не отправляй на сайты"),
    (r"\bcrontab\b|\blaunchctl\b|\bsystemctl\b", "изменение планировщика/служб системы"),
]

# Сколько текста отдаём модели: вывод webhook-ответа или log обычно короткий,
# всё остальное — шум, раздувающий контекст.
STDOUT_TAIL = 3000
STDOUT_HEAD = 1200
CAPTURE_LIMIT = 200_000


class TerminalExecutor:
    """Выполняет shell-команды и делегирует задания Koda-агенту."""

    def __init__(
        self,
        enabled: bool = True,
        cwd: Optional[str] = None,
        default_timeout: float = 120.0,
        max_timeout: float = 900.0,
        koda_bin: str = "koda",
        koda_model: str = "koda-base",
        on_progress: Optional[Callable[[str], Any]] = None,
    ):
        self.enabled = enabled
        self.cwd = str(Path(cwd or os.getcwd()).expanduser())
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.koda_bin = koda_bin
        self.koda_model = koda_model
        self.on_progress = on_progress
        self.shell = self._pick_shell()
        self._last_pid: Optional[int] = None

    # ─── Окружение ──────────────────────────────────────────────────

    @staticmethod
    def _pick_shell() -> str:
        for candidate in (os.environ.get("SHELL"), "/bin/bash", "/bin/zsh", "/bin/sh"):
            if candidate and Path(candidate).exists():
                return candidate
        return "/bin/sh"

    def _env(self) -> Dict[str, str]:
        """PATH с типовыми директориями пакетных менеджеров macOS/Linux.

        Агент часто запускается не из интерактивной оболочки (launchd, web-
        бэкенд), где PATH урезан до /usr/bin:/bin — без /opt/homebrew/bin
        вызов koda/curl/brew превращается в «command not found».
        """
        env = dict(os.environ)
        extra = [
            "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
            "/usr/sbin", "/sbin",
            str(Path.home() / ".local" / "bin"),
            str(Path.home() / "bin"),
        ]
        cur = [p for p in env.get("PATH", "").split(os.pathsep) if p]
        env["PATH"] = os.pathsep.join(cur + [p for p in extra if p not in cur])
        # Цветной вывод ломает разбор текста моделью + раздувает контекст.
        env["TERM"] = "dumb"
        env.pop("FORCE_COLOR", None)
        env["NO_COLOR"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"   # git не должен ждать логин/пароль
        env["HISTFILE"] = os.devnull       # команды агента не идут в историю
        return env

    def resolve_koda(self) -> Optional[str]:
        """Полный путь к бинарнику koda (или None, если не установлен)."""
        found = shutil.which(self.koda_bin, path=self._env()["PATH"])
        if found:
            return found
        if self.koda_bin != "koda":
            # Имя задано явно (KODA_BIN) — искать чужие koda по путам нельзя.
            return None
        for guess in ("/opt/homebrew/bin/koda", "/usr/local/bin/koda",
                      str(Path.home() / ".local" / "bin" / "koda")):
            if Path(guess).exists():
                return guess
        return None

    # ─── Проверки безопасности ──────────────────────────────────────

    def check_denied(self, command: str) -> Optional[str]:
        for pattern, reason in DENY_PATTERNS:
            if re.search(pattern, command, re.I):
                return reason
        return None

    def check_warnings(self, command: str) -> List[str]:
        return [reason for pattern, reason in WARN_PATTERNS
                if re.search(pattern, command, re.I)]

    # ─── Публичный API ──────────────────────────────────────────────

    async def run(
        self,
        command: str = "",
        prompt: str = "",
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        via_koda: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Выполнить команду (command) или задание (prompt → Koda).

        via_koda=None → режим выбирается по наличию prompt: текст задания без
        готовой команды означает «разберись сам», т.е. делегирование.
        """
        if not self.enabled:
            return {"ok": False,
                    "error": "Терминал отключён (TERMINAL_ENABLED=0) — работай только браузером"}

        command = (command or "").strip()
        prompt = (prompt or "").strip()
        if via_koda is None:
            via_koda = bool(prompt) and not command
        if via_koda and not prompt:
            prompt, command = command, ""
        if not via_koda and not command:
            return {"ok": False, "error": "Не указано действие: нужен command или prompt"}

        if via_koda:
            return await self._run_koda(prompt, cwd=cwd, timeout=timeout)
        return await self._run_shell(command, cwd=cwd, timeout=timeout)

    async def _run_shell(self, command: str, cwd: Optional[str] = None,
                         timeout: Optional[float] = None) -> Dict[str, Any]:
        denied = self.check_denied(command)
        if denied:
            logger.warning(f"Terminal command denied ({denied}): {command[:160]}")
            return {
                "ok": False,
                "error": f"Команда запрещена политикой безопасности ({denied}). "
                         f"Выполни задачу безопасной альтернативой или finish с reason.",
                "command": command[:300],
            }

        warnings = self.check_warnings(command)
        # label = 1200, не 120: ssh/sshpass-команды с опциями легко длиннее
        # 120 симв., и в панели/чате команда обрывалась на середине пути.
        result = await self._spawn([self.shell, "-c", command], cwd=cwd,
                                   timeout=timeout, label=command[:1200])
        if warnings:
            result.setdefault("warnings", warnings)
            result["detail"] = ("[WARN] " + "; ".join(warnings) + "\n") + result.get("detail", "")
        result["command"] = command[:300]
        return result

    async def _run_koda(self, prompt: str, cwd: Optional[str] = None,
                        timeout: Optional[float] = None) -> Dict[str, Any]:
        """Делегирование задания Koda-агенту (второй агент, yolo-режим)."""
        koda = self.resolve_koda()
        if not koda:
            return {
                "ok": False,
                "error": f"Koda CLI не найден (koda_bin={self.koda_bin}). "
                         f"Выполни задачу прямой командой через command=…",
            }
        # Koda сам выполняет shell-действия внутри своего yolo-цикла — фильтр по
        # строке prompt бесполезен (задание формулируется словами). Ограничиваемся
        # предупреждением: модель видит формулировку и отвечает за неё.
        warnings = self.check_warnings(prompt)
        argv = [koda, "-y", "-p", prompt, "-m", self.koda_model]
        # Делегированная задача — это несколько шагов внутри Koda, времени нужно
        # больше, чем одной команде: поднимаем потолок до max_timeout.
        timeout = min(float(timeout or self.max_timeout), self.max_timeout)
        result = await self._spawn(argv, cwd=cwd, timeout=timeout,
                                   label=f"koda -p «{prompt[:90]}»")
        result["via_koda"] = True
        result["koda_prompt"] = prompt[:500]
        if warnings:
            result.setdefault("warnings", warnings)
        if result.get("ok"):
            result["detail"] = (
                f"Koda-агент ({self.koda_model}) выполнил задание. Его ответ ниже — "
                f"используй результат по задаче (введи данные в браузер / finish).\n"
                + result.get("detail", "")
            )
        return result

    # ─── Низкоуровневый запуск ──────────────────────────────────────

    async def _spawn(self, argv: List[str], cwd: Optional[str] = None,
                     timeout: Optional[float] = None, label: str = "") -> Dict[str, Any]:
        """Запуск процесса, пословное чтение вывода, жёсткий таймаут.

        Убиваем всю группу процессов: `sleep 100` в shell оставляет сироту,
        который держит pipe и мешает завершению шага.
        """
        timeout = float(timeout or self.default_timeout)
        timeout = max(1.0, min(timeout, self.max_timeout))
        workdir = str(Path(cwd).expanduser()) if cwd else self.cwd
        if not Path(workdir).is_dir():
            return {"ok": False, "error": f"Рабочий каталог не существует: {workdir}"}

        await self._progress(f"[TERM] $ {label}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workdir,
                env=self._env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError:
            return {"ok": False, "error": f"Не найден исполняемый файл: {argv[0]}"}
        except Exception as e:
            return {"ok": False, "error": f"Не удалось запустить процесс: {e}"}

        self._last_pid = proc.pid
        chunks: List[str] = []
        state = {"total": 0, "truncated": False, "last_beat": asyncio.get_running_loop().time()}
        started = state["last_beat"]

        async def _collect():
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                state["total"] += len(line_bytes)
                if state["total"] > CAPTURE_LIMIT:
                    state["truncated"] = True
                    # Буфер переполнен — процесс молча добиваем, иначе разрастётся RAM
                    self._terminate(proc)
                    break
                text = line_bytes.decode("utf-8", errors="replace")
                chunks.append(text)
                now = asyncio.get_running_loop().time()
                # Долгий процесс (koda, сборка, скачивание) не должен выглядеть
                # зависшим: раз в 15с показываем последнюю строку вывода.
                if now - state["last_beat"] > 15:
                    state["last_beat"] = now
                    await self._progress(f"[TERM] …{int(now - started)}с: {(text.strip() or '…')[:120]}")

        try:
            await asyncio.wait_for(_collect(), timeout=timeout)
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            self._terminate(proc)
            await self._progress(f"[TERM] Превышен таймаут {timeout:.0f}с — процесс остановлен")
            partial = self._format_output("".join(chunks))
            if partial:
                await self._progress(f"[TERM] {partial[:8000]}")
            return {
                "ok": False,
                "error": f"Команда не завершилась за {timeout:.0f}с и остановлена. "
                         f"Разбей её на части или запусти с меньшим объёмом работы.",
                "timeout": True,
                "output_partial": self._format_output("".join(chunks)),
            }
        except Exception as e:
            self._terminate(proc)
            return {"ok": False, "error": f"Ошибка выполнения: {e}"}

        output = "".join(chunks)
        exit_code = proc.returncode if proc.returncode is not None else -1
        # Отрицательный код = убиты сигналом (SIGKILL от нашего таймаута и т.п.)
        killed = exit_code < 0
        shown = self._format_output(output)
        clipped = state["truncated"] or len(output) > STDOUT_HEAD + STDOUT_TAIL + 20

        # Вывод — в нижнюю терминальную панель UI. Раньше в _progress уходила
        # только строка команды: пользователь видел "$ ping ..." и ни одной
        # строки ответа, хотя команда отработала.
        if shown:
            await self._progress(f"[TERM] {shown[:8000]}")
        elif not killed:
            await self._progress(f"[TERM] (вывод пустой)")
        await self._progress(f"[TERM] exit={exit_code}")

        result: Dict[str, Any] = {
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "output": shown,
            "detail": (f"exit={exit_code}, вывод {len(output)} симв."
                       + (" (обрезан)" if clipped else "") + "\n" + shown)
                      if shown else f"exit={exit_code}, вывод пустой",
        }
        if killed:
            result["ok"] = False
            result["error"] = f"Процесс убит сигналом {-exit_code}"
        elif exit_code != 0:
            # Не нулевой код — НЕ ошибка цикла: модель должна прочитать вывод и
            # решить сама (grep без совпадений даёт exit=1 при нормальном итоге).
            result["error"] = f"Команда завершилась с кодом {exit_code}"
            result["nonzero_exit"] = True
        return result

    @staticmethod
    def _format_output(output: str) -> str:
        """Сжать вывод: начало + хвост, середина не нужна модели."""
        output = output.strip()
        if len(output) <= STDOUT_HEAD + STDOUT_TAIL + 20:
            return output
        return (output[:STDOUT_HEAD].rstrip()
                + f"\n… [пропущено {len(output) - STDOUT_HEAD - STDOUT_TAIL} симв.] …\n"
                + output[-STDOUT_TAIL:].lstrip())

    @staticmethod
    def _terminate(proc) -> None:
        """Остановить процесс и его детей (своя session-группа)."""
        if proc.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                return

        async def _force():
            # Дать долиться 3с, затем SIGKILL — иначе шаг зависнет намертво
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        try:
            asyncio.get_running_loop().create_task(_force())
        except RuntimeError:
            pass

    def cancel(self) -> None:
        """Экстренная остановка текущего процесса (по запросу пользователя)."""
        if not self._last_pid:
            return
        try:
            os.killpg(os.getpgid(self._last_pid), signal.SIGTERM)
        except Exception:
            pass

    async def _progress(self, text: str) -> None:
        if self.on_progress is None:
            return
        try:
            res = self.on_progress(text)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            logger.warning(f"Terminal progress callback error: {e}")


# ─── Завод из окружения ─────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except (ValueError, TypeError):
        return default


def terminal_from_env(on_progress: Optional[Callable[[str], Any]] = None) -> TerminalExecutor:
    """Собрать исполнитель из переменных окружения (.env)."""
    return TerminalExecutor(
        enabled=_env_bool("TERMINAL_ENABLED", True),
        cwd=os.environ.get("TERMINAL_CWD") or None,
        default_timeout=_env_float("TERMINAL_TIMEOUT", 120.0),
        max_timeout=_env_float("TERMINAL_MAX_TIMEOUT", 900.0),
        koda_bin=os.environ.get("KODA_BIN") or "koda",
        koda_model=os.environ.get("KODA_MODEL") or "koda-base",
        on_progress=on_progress,
    )
