"""Режим «Управление компьютером»: запрос → bash-команда → результат → ответ.

Модель (qwen3:8b) работает в tool-цикле: на каждый шаг она возвращает
ОДИН JSON-объект — либо {"action":"run","cmd":"..."} (выполнить в терминале),
либо {"action":"say","text":"..."} (ответить пользователю). Результат команды
возвращается модели как новая реплика, и она решает: ещё команда или итог.

Команды НЕ зашиты в промпт — модель сама подбирает их под задачу
(открыть программу, nmap, git, brew, файлы…). Опасные паттерны блокируются
гардом до выполнения.
"""
import json
import logging
import os
import re
import subprocess
import time
import urllib.request

log = logging.getLogger("assist.computer")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

MAX_STEPS = 8            # команд на один запрос пользователя
CMD_TIMEOUT = 120        # сек на одну команду
OUT_LIMIT = 4000         # сколько символов вывода показываем модели/пользователю

# Гард: эти паттерны не выполняем никогда (подстроки/regex, нижний регистр).
_DANGER = [
    r"rm\s+-[a-z]*r[a-z]*\s+/(?:\s|$)",   # rm -rf /
    r"rm\s+-[a-z]*r[a-z]*\s+~",           # rm -rf ~
    r"\bmkfs\b", r"\bdd\s+if=", r"\bshutdown\b", r"\breboot\b",
    r"diskutil\s+erase", r":\(\)\s*\{",   # fork-bomb
    r">\s*/dev/sd", r"\bsudo\b",          # sudo — только вручную
    r"chown\s+-R\s+.*\s+/$",
]
_DANGER_RE = [re.compile(p) for p in _DANGER]

SYSTEM_PROMPT = (
    "Ты — голосовой оператор терминала macOS (darwin, zsh). "
    "Ты можешь выполнять любые команды терминала через инструмент run. "
    "Отвечай СТРОГО одним JSON-объектом, без текста вокруг него:\n"
    '{"action":"run","cmd":"<одна bash-команда>"} — если нужно что-то сделать или узнать;\n'
    '{"action":"say","text":"<короткий ответ по-русски>"} — когда задача выполнена, '
    "невозможна, или нужен уточняющий вопрос.\n"
    "Правила:\n"
    "- одна команда за один шаг, без интерактивных флагов (-i, read, pause); "
    "долгие выводы ограничивай через head/tail/grep;\n"
    "- открытие программ: open -a \"Название\" или open https://…;\n"
    "- сначала проверь наличие утилиты (command -v nmap), потом используй;\n"
    "- после 2-3 неудачных попыток — say с объяснением проблемы;\n"
    "- в say пиши кратко (1-3 предложения), как голосовой ассистент, без markdown.\n"
    "Никаких меток вроде «Суть:», никаких инструкций в выводе."
)


def _danger_reason(cmd: str) -> str | None:
    low = cmd.lower()
    for rx in _DANGER_RE:
        if rx.search(low):
            return f"команда заблокирована гардом безопасности: {rx.pattern}"
    return None


def _call_ollama(messages: list[dict]) -> str:
    """Один шаг модели → текст ответа (JSON от модели или error-строка)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": 400, "temperature": 0.2},
        "think": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            content = data.get("message", {}).get("content", "")
            # счистить thinking-блоки qwen3, если вдруг включились
            content = re.sub(r"<think.*?>.*?</think\s*>", "", content,
                             flags=re.S | re.I).strip()
            log.info("computer: модель ответила за %.1fs: %r",
                     time.monotonic() - t0, content[:120])
            return content
    except Exception as e:
        log.error("computer: Ollama ошибка: %s", e)
        return json.dumps({"action": "say", "text": f"LLM недоступна: {e}"},
                          ensure_ascii=False)


def _extract_json(text: str) -> dict | None:
    """Достаёт первый JSON-объект из ответа (модель иногда добавляет текст)."""
    # часто оборачивают в ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    break
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _run_command(cmd: str) -> dict:
    """Выполнить команду с таймаутом; вернуть {exit, out}."""
    log.info("computer: выполняю: %r", cmd[:160])
    try:
        p = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=CMD_TIMEOUT, cwd=os.path.expanduser("~"),
            env={**os.environ, "TERM": "dumb", "NO_COLOR": "1",
                 "LANG": "en_US.UTF-8"},
        )
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        out = out.strip()
        if len(out) > OUT_LIMIT:
            out = out[:OUT_LIMIT // 2] + "\n…[вывод усечён]…\n" + out[-OUT_LIMIT // 2:]
        return {"exit": p.returncode, "out": out or "(пустой вывод)"}
    except subprocess.TimeoutExpired:
        return {"exit": -1, "out": f"команда прервана: превышен таймаут {CMD_TIMEOUT} сек"}
    except Exception as e:
        return {"exit": -1, "out": f"ошибка запуска: {e}"}


class ComputerAgent:
    """Сохраняет короткую историю между репликами пользователя,
    чтобы «и ещё раз, но без sudo» понималось в контексте."""

    def __init__(self):
        self.history: list[dict] = []

    def reset(self):
        self.history.clear()

    def turn(self, user_text: str) -> dict:
        """Один пользовательский запрос → {answer, events}.

        events — что реально произошло (для показа в UI):
        [{"cmd":..., "exit":..., "out":...} | {"blocked": reason}].
        Синхронный метод — вызывать через run_in_executor.
        """
        events: list[dict] = []
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-12:]

        answer = ""
        for step in range(MAX_STEPS):
            raw = _call_ollama(messages)
            obj = _extract_json(raw)

            if obj is None:
                # модель ответила обычным текстом — считаем это say
                answer = raw or "[пустой ответ]"
                break

            action = (obj.get("action") or "").lower()
            if action == "say":
                answer = str(obj.get("text") or "").strip() or "[пустой ответ]"
                break

            if action == "run":
                cmd = str(obj.get("cmd") or "").strip()
                if not cmd:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user",
                                     "content": 'Пустая команда. Верни JSON с "cmd".'})
                    continue
                reason = _danger_reason(cmd)
                if reason:
                    log.warning("computer: %s → %r", reason, cmd[:120])
                    events.append({"cmd": cmd, "blocked": reason})
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user",
                                     "content": f"Команда отклонена: {reason}. "
                                                "Предложи пользователю выполнить её вручную."})
                    continue
                res = _run_command(cmd)
                events.append({"cmd": cmd, "exit": res["exit"], "out": res["out"]})
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": f"Результат (exit={res['exit']}):\n{res['out']}"})
                continue

            # неизвестное действие
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": 'Неизвестное действие. Верни "run" или "say".'})
        else:
            answer = "Остановился по лимиту шагов — последние действия смотри выше."

        if not answer:
            last = [e for e in events if "out" in e]
            answer = "Готово." + (f"\n{last[-1]['out'][:600]}" if last else "")

        # в историю кладём только финальный ответ пользователя/ассистента —
        # шум выводов команд живёт в рамках одного turn
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > 24:
            self.history = self.history[-24:]
        return {"answer": answer, "events": events}


_agent: ComputerAgent | None = None


def get_agent() -> ComputerAgent:
    global _agent
    if _agent is None:
        _agent = ComputerAgent()
    return _agent