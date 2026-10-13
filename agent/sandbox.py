"""
Sandbox — изолированный исполнитель для запуска произвольных программ.

Отличие от agent/terminal.py: terminal выполняет точечные команды пользователя
или агента в рабочем каталоге проекта, а sandbox — это «песочница»: отдельный
каталог, куда скрипт скачивает данные и пишет результаты, и из которого он НЕ
может ничего изменить снаружи. Реализация на macOS — штатный `sandbox-exec`
(профиль seatbelt), на Linux — `bwrap` при наличии, иначе запуск без
изоляции, но с рабочим каталогом внутри песочницы.

Что отдаётся наружу: каталог песочницы (файлы можно показать/скачать),
вывод команды, код возврата. Всё остальное — только внутри каталога.
"""

import asyncio
import logging
import os
import platform
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Вывод сжимается так же, как в terminal: голова + хвост, середина модели не нужна
STDOUT_HEAD = 1500
STDOUT_TAIL = 3500
CAPTURE_LIMIT = 400_000

# Команды, которые бессмысленно запускать даже в песочнице (они обращаются к
# самой системе, а не к её файлам).
DENY_PATTERNS: List[tuple] = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*(-rf?|--recursive)\s+/(?:\s|$)"),
     "рекурсивное удаление корня"),
    (re.compile(r"\bmkfs(\.|\s)"), "разметка диска"),
    (re.compile(r"\bdd\b[^\n]*of=/dev/"), "запись на блочное устройство"),
    (re.compile(r"\b(shutdown|reboot|halt)\b"), "выключение машины"),
    (re.compile(r":\(\)\s*\{.*\};\s*:"), "fork-bomb"),
]

# Разрешённые интерпретаторы для запуска готового скрифта из песочницы.
# Ограничение нужно, чтобы вызов interpreter=... не превратился в произвольный
# exec системного бинарника вне песочницы.
ALLOWED_INTERPRETERS = {
    "python": "python3", "python3": "python3", "sh": "sh", "bash": "bash",
    "node": "node", "perl": "perl", "ruby": "ruby", "php": "php",
}

DEFAULT_SANDBOX_ROOT = Path("ai_browser_sandbox")


def sandbox_root_from_env() -> Path:
    """Корень песочницы (терминал SANDBOX_ROOT). Всегда абсолютный, внутри проекта."""
    raw = os.environ.get("SANDBOX_ROOT", "").strip()
    root = Path(raw).expanduser() if raw else DEFAULT_SANDBOX_ROOT
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "нет", "off")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


class Sandbox:
    """Изолированная среда: каталог + запуск команд/скриптов с ограничением записи."""

    def __init__(self, root: Optional[Path] = None, enabled: Optional[bool] = None,
                 timeout: float = 120.0, max_timeout: float = 600.0,
                 isolate: Optional[bool] = None,
                 on_progress: Optional[Callable[[str], Any]] = None):
        # resolve обязателен: без него /tmp и /private/tmp считаются разными
        # путями и проверка выхода за песочницу ложно блокирует легитимную запись.
        self.root = (Path(root) if root else sandbox_root_from_env()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.enabled = True if enabled is None else bool(enabled)
        self.timeout = timeout
        self.max_timeout = max_timeout
        self.on_progress = on_progress
        self.isolate = True if isolate is None else bool(isolate)
        self._last_pid: Optional[int] = None
        self._runner = self._detect_runner()

    # ─── Профиль изоляции ───────────────────────────────────────────

    @staticmethod
    def _detect_runner() -> str:
        """Чем изолировать: sandbox-exec (macOS) → bwrap (Linux) → '' (без изоляции)."""
        system = platform.system().lower()
        if system == "darwin" and Path("/usr/bin/sandbox-exec").exists():
            return "sandbox-exec"
        if system == "linux" and shutil.which("bwrap"):
            return "bwrap"
        logger.warning("Изолятор не найден (sandbox-exec/bwrap) — песочница работает "
                       "только с рабочим каталогом, без ограничения записи")
        return ""

    def _seatbelt_profile(self) -> Path:
        """Профиль seatbelt: писать можно только в песочницу и во временные каталоги.

        Чтение оставлено открытым (скриптам нужны системные библиотеки и данные),
        но секреты спрятаны: ~/.ssh, ключи и куки недоступны даже на чтение.
        """
        root = str(self.root)
        home = str(Path.home())
        # Порядок важен: в seatbelt выигрывает ПОСЛЕДНЕЕ совпадение, поэтому
        # широкие запраты идут первыми, а разрешённые пути — перекрывать их.
        profile = rf"""(version 1)
(allow default)

; Снаружи писать нельзя нигде
(deny file-write*
 (subpath "/")
 (subpath "/tmp") (subpath "/private/tmp") (subpath "/var/tmp")
 (subpath "/Users") (subpath "/etc") (subpath "/usr") (subpath "/bin")
 (subpath "/sbin") (subpath "/opt") (subpath "/Library") (subpath "/System")
 (subpath "/Applications"))

; Разрешено: сама песочница и временные каталоги ОС (нужны рантаймам)
(allow file-write* (subpath "{root}"))
(allow file-write* (subpath "/private/var/folders"))

; Секреты не читаем даже внутри песочницы
(deny file-read* (subpath "{home}/.ssh") (subpath "{home}/.gnupg")
 (subpath "{home}/.aws") (subpath "{home}/.config/gcloud")
 (subpath "{home}/Library/Keychains"))
(deny file-read* (regex #"\.env$"))
"""
        # Профиль лежит ВНЕ песочницы, иначе скрипт мог бы сам себя разблокировать.
        path = self.root.parent / ".sandbox_profile.sb"
        path.write_text(profile, encoding="utf-8")
        return path

    def _wrap(self, argv: List[str]) -> List[str]:
        """Обернуть команду в изолятор (если он есть)."""
        if not self.isolate or not self._runner:
            return argv
        if self._runner == "sandbox-exec":
            return ["/usr/bin/sandbox-exec", "-f", str(self._seatbelt_profile()), *argv]
        if self._runner == "bwrap":
            root = str(self.root)
            return ["bwrap", "--ro-bind", "/", "/", "--bind", root, root,
                    "--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc",
                    "--unshare-user", "--unshare-pid", "--die-with-parent", *argv]
        return argv

    # ─── Публичное API ──────────────────────────────────────────────

    def path(self, rel: str = "") -> Path:
        """Путь внутри песочницы с защитой от выхода за её пределы (../..)."""
        target = (self.root / (rel or "")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"Путь вне песочницы: {rel}")
        return target

    def check_denied(self, command: str) -> Optional[str]:
        for rx, why in DENY_PATTERNS:
            if rx.search(command):
                return why
        return None

    async def run(self, command: str = "", script: str = "", interpreter: str = "python3",
                  filename: str = "", cwd: str = "", timeout: Optional[float] = None,
                  isolate: Optional[bool] = None) -> Dict[str, Any]:
        """Выполнить команду или готовый скрипт внутри песочницы.

        command  — shell-строка (запускается в корне песочницы);
        script   — текст скрипта: сохраняется в песочницу и запускается;
        filename — уже существующий файл песочницы (запуск по имени, без записи).
        """
        if not self.enabled:
            return {"ok": False,
                    "error": "Песочница отключена (SANDBOX_ENABLED=0)"}

        command = (command or "").strip()
        script = (script or "").strip()
        filename = (filename or "").strip()

        if script:
            name = filename or f"script_{int(time.time())}.py"
            res = self.write_file(name, script)
            if not res.get("ok"):
                return res
            filename = name
            command = ""

        if filename:
            try:
                target = self.path(filename)
            except ValueError as e:
                return {"ok": False, "error": str(e)}
            if not target.exists():
                return {"ok": False,
                        "error": f"Файла нет в песочнице: {filename}. Сначала запиши "
                                 f"его (script=...) или скачай командой."}
            interp = ALLOWED_INTERPRETERS.get((interpreter or "").strip().lower())
            if not interp:
                return {"ok": False,
                        "error": f"Интерпретатор '{interpreter}' не поддерживается. "
                                 f"Доступны: {', '.join(sorted(set(ALLOWED_INTERPRETERS)))}"}
            argv = [interp, str(target)]
            label = f"{interp} {filename}"
        elif command:
            denied = self.check_denied(command)
            if denied:
                return {"ok": False,
                        "error": f"Команда запрещена даже в песочнице ({denied}). "
                                 f"Она обращается к самой системе, а не к файлам."}
            argv = ["/bin/bash", "-lc", command]
            label = command
        else:
            return {"ok": False,
                    "error": "Нужно command (команда) или script (текст скрипта)"}

        workdir = self.root
        if cwd:
            try:
                workdir = self.path(cwd)
            except ValueError as e:
                return {"ok": False, "error": str(e)}
            if not workdir.is_dir():
                return {"ok": False, "error": f"Каталога нет в песочнице: {cwd}"}

        eff_isolate = self.isolate if isolate is None else bool(isolate)
        return await self._spawn(self._wrap(argv) if eff_isolate else argv,
                                 cwd=workdir, timeout=timeout, label=label)

    # ─── Файлы песочницы ────────────────────────────────────────────

    def write_file(self, name: str, content: str) -> Dict[str, Any]:
        """Записать файл в песочницу."""
        try:
            target = self.path(name)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "detail": f"Записан {name} ({len(content)} симв.)",
                "path": str(target)}

    def list_files(self, subdir: str = "") -> Dict[str, Any]:
        """Содержимое песочницы (имена, размеры, mtime) — для панели в чате."""
        try:
            base = self.path(subdir)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not base.is_dir():
            return {"ok": False, "error": f"Каталога нет: {subdir}"}
        items = []
        total = 0
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            total += st.st_size
            items.append({
                "name": str(p.relative_to(self.root)),
                "size": st.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            })
        return {"ok": True, "files": items[:500], "total_size": total,
                "root": str(self.root),
                "detail": f"{len(items)} файл(ов), {self._human(total)}"}

    def read_file(self, name: str, limit: int = 20_000) -> Dict[str, Any]:
        """Прочитать файл песочницы (текст). Для показа результата в чате."""
        try:
            target = self.path(name)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not target.is_file():
            return {"ok": False, "error": f"Файла нет: {name}"}
        raw = target.read_bytes()[:limit]
        return {"ok": True, "name": name, "size": target.stat().st_size,
                "content": raw.decode("utf-8", errors="replace"),
                "truncated": target.stat().st_size > limit}

    def delete(self, name: str = "") -> Dict[str, Any]:
        """Удалить файл, каталог или всё содержимое песочницы."""
        if not name:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root.mkdir(parents=True, exist_ok=True)
            return {"ok": True, "detail": "Песочница очищена"}
        try:
            target = self.path(name)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
        else:
            return {"ok": False, "error": f"Нет такого пути: {name}"}
        return {"ok": True, "detail": f"Удалено: {name}"}

    def download_url(self, url: str, name: str = "") -> Dict[str, Any]:
        """Скачать файл внутрь песочницы (скачивание — частый сценарий задачи)."""
        import urllib.parse
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "error": "Нужен http(s) URL"}
        if not name:
            name = Path(urllib.parse.urlparse(url).path).name or "download.bin"
        try:
            target = self.path(name)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "command": f'curl -L --fail -o "{name}" "{url}"',
                "target": name,
                "detail": f"Скачать {url[:120]} → {name} (выполни returned command)"}

    def info(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "isolation": self._runner or "нет (только рабочий каталог)",
            "root": str(self.root),
            "timeout": self.timeout,
        }

    @staticmethod
    def _human(n: int) -> str:
        for unit in ("Б", "КБ", "МБ", "ГБ"):
            if n < 1024:
                return f"{n:.0f} {unit}" if unit == "Б" else f"{n / 1:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} ТБ"

    # ─── Исполнение ─────────────────────────────────────────────────

    async def _spawn(self, argv: List[str], cwd: Path, timeout: Optional[float],
                     label: str) -> Dict[str, Any]:
        tmo = self.timeout if timeout is None else min(float(timeout), self.max_timeout)
        await self._progress(f"[SBX] $ {label[:1200]}")
        env = self._env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=str(cwd), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError:
            return {"ok": False, "error": f"Не найден исполняемый файл: {argv[0]}"}
        except Exception as e:
            return {"ok": False, "error": f"Не удалось запустить: {e}"}

        self._last_pid = proc.pid
        chunks: List[str] = []
        total = 0
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=tmo)
            chunks.append(out.decode("utf-8", errors="replace"))
            total = len(chunks[0])
        except asyncio.TimeoutError:
            self._terminate(proc)
            await self._progress(f"[SBX] ⏱ Таймаут {tmo:.0f}с — процесс остановлен")
            return {"ok": False, "timeout": True,
                    "error": f"Команда не завершилась за {tmo:.0f}с и остановлена. "
                             f"Уменьши объём работы или запусти с большим timeout.",
                    "output_partial": "".join(chunks)[:STDOUT_HEAD]}
        except Exception as e:
            self._terminate(proc)
            return {"ok": False, "error": f"Ошибка выполнения: {e}"}
        finally:
            self._last_pid = None

        if total > CAPTURE_LIMIT:
            chunks = [chunks[0][:CAPTURE_LIMIT]]
        output = self._format_output("".join(chunks))
        code = proc.returncode if proc.returncode is not None else -1
        result: Dict[str, Any] = {
            "ok": code == 0,
            "exit_code": code,
            "output": output,
            "detail": (f"exit={code}, вывод {total} симв.\n{output}" if output
                       else f"exit={code}, вывод пустой"),
            "sandbox": str(cwd),
        }
        if code != 0:
            result["error"] = f"Команда завершилась с кодом {code}"
            result["nonzero_exit"] = True
        files = self.list_files()
        if files.get("ok") and files["files"]:
            result["files"] = [f["name"] for f in files["files"][:20]]
        # Вывод — в терминальную панель UI (как в terminal.py): без него панель
        # показывала только команду и exit-код.
        if output:
            await self._progress(f"[SBX] {output[:8000]}")
        await self._progress(f"[SBX] exit={code}")
        return result

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        extra = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        env["PATH"] = env.get("PATH", "") + ":" + extra
        env["NO_COLOR"] = "1"
        env["HISTFILE"] = os.devnull
        env["HOME"] = str(self.root)  # скрипты пишут «домашние» файлы в песочницу
        env["TMPDIR"] = str(self.root)
        return env

    def _terminate(self, proc) -> None:
        if proc.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    @staticmethod
    def _format_output(output: str) -> str:
        output = output.strip()
        if len(output) <= STDOUT_HEAD + STDOUT_TAIL + 20:
            return output
        return (output[:STDOUT_HEAD].rstrip()
                + f"\n… [пропущено {len(output) - STDOUT_HEAD - STDOUT_TAIL} симв.] …\n"
                + output[-STDOUT_TAIL:].lstrip())

    def cancel(self) -> None:
        if self._last_pid:
            try:
                os.killpg(os.getpgid(self._last_pid), signal.SIGKILL)
            except Exception:
                pass

    async def _progress(self, text: str) -> None:
        logger.info(text)
        if self.on_progress:
            try:
                r = self.on_progress(text)
                if asyncio.iscoroutine(r):
                    await r
            except Exception as e:
                logger.warning(f"Sandbox progress error: {e}")


def sandbox_from_env(on_progress: Optional[Callable[[str], Any]] = None) -> Sandbox:
    """Собрать песочницу из окружения (SANDBOX_ENABLED/ROOT/TIMEOUT/ISOLATE)."""
    return Sandbox(
        enabled=_env_bool("SANDBOX_ENABLED", True),
        isolate=_env_bool("SANDBOX_ISOLATE", True),
        timeout=_env_float("SANDBOX_TIMEOUT", 120.0),
        max_timeout=_env_float("SANDBOX_MAX_TIMEOUT", 600.0),
        on_progress=on_progress,
    )
