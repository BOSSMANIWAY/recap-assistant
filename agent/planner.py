"""
Agent Planner v2 - LLM-планировщик с tool calling

Отличия от v1:
- Вход: BrowserState (element_id) вместо сырого DOM
- Выход: {plan, thought, action} вместо плоского JSON
- Нормализация: любые синонимы tool'ов маппятся на канонические
- Устойчивый парсинг: markdown-обёртки, текст вокруг JSON, обрезанный JSON
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List

from prompts.system import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT_TEMPLATE
from agent.user_profile import profile_to_prompt

logger = logging.getLogger(__name__)

# Regex для извлечения чистого URL из Markdown-ссылок [text](url)
_MARKDOWN_URL_RE = re.compile(r"\[.*?\]\((https?://[^)]+)\)")


_MARKDOWN_INLINE_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")


def strip_markdown_urls(text: str) -> str:
    """[текст](https://url) → https://url во всей строке shell-команды.

    Модель системно оформляет URL markdown-ссылкой. Внутри команды скобки
    убивают curl (exit 3, malformed URL), а причина в stderr до модели не
    доходила — отсюда бесконечный повтор одной и той же команды.
    """
    if not text or "](" not in text:
        return text
    cleaned = _MARKDOWN_INLINE_RE.sub(lambda m: m.group(2), text)
    if cleaned != text:
        logger.info(f"terminal: markdown-URL раскатан: {text[:80]!r} -> {cleaned[:80]!r}")
    return cleaned


def _sanitize_url(raw: str) -> str:
    """Извлечь чистый URL из Markdown-ссылки [text](url) или вернуть как есть."""
    raw = str(raw).strip()
    m = _MARKDOWN_URL_RE.search(raw)
    if m:
        cleaned = m.group(1)
        logger.debug(f"Sanitized URL from markdown: {raw!r} -> {cleaned!r}")
        return cleaned
    return raw

# Канонические tool'ы
VALID_TOOLS = {
    "navigate", "click", "click_at", "hover", "drag", "type", "key", "scroll", "select",
    "new_tab", "switch_tab", "close_tab", "extract", "wait", "wait_user",
    "evaluate", "finish", "fill_form", "terminal", "sandbox",
}

# Синонимы → канонические имена
TOOL_SYNONYMS = {
    "navigate_to": "navigate", "navigate_to_url": "navigate", "go_to": "navigate",
    "go_to_url": "navigate", "open_url": "navigate", "open": "navigate", "goto": "navigate",
    "open_tab": "new_tab", "newtab": "new_tab", "create_tab": "new_tab",
    "open_link": "navigate", "go_to_site": "navigate", "go_url": "navigate",
    "change_url": "navigate", "goto_url": "navigate", "visit": "navigate",
    "fill": "type", "input": "type", "write": "type",
    "submit_form": "click", "submit": "click",
    "press": "key", "keypress": "key", "keyboard": "key",
    "complete": "finish", "done": "finish", "task_complete": "finish",
    "answer": "finish", "stop": "finish",
    "js": "evaluate", "javascript": "evaluate", "run_js": "evaluate",
    "screenshot": "wait",  # скриншот как отдельное действие не нужен — делаем всегда
    "confirm_action": "key", "request_confirmation": "key",
    "fetch_url": "navigate", "search": "navigate",
    # Терминал: модель может звать его по-разному
    "shell": "terminal", "bash": "terminal", "run": "terminal", "exec": "terminal",
    "execute": "terminal", "run_command": "terminal", "command": "terminal",
    "terminal_command": "terminal", "cli": "terminal", "koda": "terminal",
    "delegate": "terminal", "delegate_to_koda": "terminal", "subagent": "terminal",
    # Песочница: изолированный запуск скриптов/программ
    "sandbox_run": "sandbox", "run_sandbox": "sandbox", "isolate": "sandbox",
    "isolated": "sandbox", "box": "sandbox", "jail": "sandbox",
    "песочница": "sandbox", "safe_run": "sandbox", "isolated_run": "sandbox",
}


def _short_params(params: Dict[str, Any]) -> str:
    """Компактная строка параметров шага (длинные тексты — только длина)."""
    if not isinstance(params, dict) or not params:
        return ""
    parts = []
    for k, v in params.items():
        if v is None or v == "" or isinstance(v, bool):
            continue
        s = str(v)
        if len(s) > 40:
            parts.append(f"{k}=«{s[:30]}…» ({len(s)} симв.)")
        else:
            parts.append(f"{k}={s}")
    return " ".join(parts[:4])


def format_history(recent_history: List[Dict[str, Any]]) -> str:
    """
    История шагов для промпта: инструмент + параметры + ФАКТИЧЕСКИЙ результат
    + был ли прогресс. Результат живёт в details.result — раньше он не
    извлекался, и модель видела пустую историю, поэтому повторяла шаги.
    """
    if not recent_history:
        return "(это первый шаг — истории нет)"
    lines = []
    base = len(recent_history)
    for i, h in enumerate(recent_history, 1):
        details = h.get("details") or {}
        tool = h.get("action", "?")
        params = _short_params(details.get("params") or {})
        # Подпись элемента важнее element_id: id пересоздаются на каждом
        # снимке, и «click e1» в истории — два разных клика или один и тот же,
        # непонятно. «click e1 → button «Start»» снимает неоднозначность.
        target = str(details.get("target") or "").strip()
        if target:
            params = f"{params} → {target}".strip()
        res = details.get("result")
        if isinstance(res, dict):
            res_text = res.get("detail") or res.get("message") or res.get("error") or ""
            if res.get("ok") is False and res.get("error"):
                res_text = f"ОШИБКА: {res['error']}"
        else:
            res_text = str(res or "")
        progress = details.get("progress")
        mark = "прогресс: да" if progress else ("прогресс: НЕТ" if progress is False else "")
        head = f"[шаг {base - i + 1}] {tool}" + (f" {params}" if params else "")
        # Вывод терминала — это данные задачи (число из ответа, токен, путь):
        # обрезать до 120 символ. нельзя, модель потеряет результат работы.
        cap = 3000 if tool in ("terminal", "sandbox", "extract") else 120
        line = head + (f" → {str(res_text)[:cap]}" if res_text else "")
        if mark:
            line += f" | {mark}"
        lines.append(line)
    return "\n".join(lines)


class AgentPlanner:
    """Планировщик v2: BrowserState → LLM → {plan, thought, action}"""

    def __init__(self, llm_client: Any, model: str = ""):
        self.llm_client = llm_client
        self.model = model

    async def decide(
        self,
        task: str,
        browser_state_text: str,
        tabs_text: str,
        last_action_result: str,
        recent_history: List[Dict[str, Any]],
        step: int,
        rejected_warning: str = "",
        user_messages: str = "",
        screenshot_b64: str = "",
        sheet_order: Optional[List[str]] = None,
        current_plan: Optional[List[str]] = None,
        cycle_warning: str = "",
        screenshot_kind: str = "grid",
    ) -> Dict[str, Any]:
        """Принятие решения. Возвращает нормализованное действие.
        screenshot_b64: PNG-base64 (sheet уровня или viewport) — если задан,
        передаётся модели как изображение (vision). Нужен для визуальных
        уровней игры, где текстовое состояние не показывает картинки.
        sheet_order — порядок eID клеток на sheet (строки сверху вниз).
        current_plan — план с прошлого шага: модель обязана его продолжить,
        а не сочинять заново (иначе она зацикливается на первом пункте).
        cycle_warning — предупреждение рантайма о хождении по кругу."""
        # Модель без vision (например qwen3:8b в Ollama) физически не принимает
        # изображения: chat_with_image деградирует в текстовый chat, а цикл
        # ретраев ниже только сожжёт минуты на паузах. Убираем картинку сразу.
        if screenshot_b64 and not self.llm_client.accepts_vision():
            logger.info(f"Модель {self.llm_client.model} без vision — скриншот не отправляю")
            screenshot_b64 = ""
            sheet_order = None
        history_text = format_history(recent_history)

        plan_text = ""
        if current_plan:
            plan_text = (
                "=== ТЕКУЩИЙ ПЛАН (продолжай ЕГО, не переписывай с нуля) ===\n"
                + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(current_plan))
                + "\nЭто твой план с прошлого шага. Пункты, которые уже отражены "
                  "в истории действий, — ВЫПОЛНЕНЫ, не повторяй их. Верни этот же "
                  "план (или его остаток) и сделай СЛЕДУЮЩИЙ невыполненный пункт.\n"
            )

        user_prompt = AGENT_USER_PROMPT_TEMPLATE.format(
            task=task,
            user_profile=profile_to_prompt() + rejected_warning + user_messages + cycle_warning,
            last_action_result=last_action_result or "(первый шаг)",
            browser_state=browser_state_text,
            tabs=tabs_text,
            recent_history=history_text,
            step=step,
        )
        if plan_text:
            user_prompt = user_prompt.replace(
                "=== РЕЗУЛЬТАТ ПРЕДЫДУЩЕГО ДЕЙСТВИЯ ===",
                plan_text + "=== РЕЗУЛЬТАТ ПРЕДЫДУЩЕГО ДЕЙСТВИЯ ===",
            )
        if screenshot_b64:
            if sheet_order:
                rows: List[List[str]] = []
                per = 4
                for i in range(0, len(sheet_order), per):
                    rows.append(sheet_order[i:i + per])
                order_text = "\n".join(
                    f"ряд {i+1}: {', '.join(r)}" for i, r in enumerate(rows)
                )
                user_prompt += (
                    "\n\n=== КАРТИНКА УРОВНЯ (vision) ===\n"
                    "Приложено изображение: клетки игровой сетки, вырезанные из "
                    "фотографии и собранные в ряды. Под каждой клеткой белая "
                    "подпись — её element id (eN). Порядок рядов:\n"
                    f"{order_text}\n"
                    "Опиши, что видишь в каждой клетке (объекты, знаки, цвета), "
                    "и реши задачу уровня. НЕ придумывай содержимое: если картинка "
                    "не отображается у тебя — честно скажи 'не вижу изображение'."
                )
            elif screenshot_kind == "terminal":
                user_prompt += (
                    "\n\n=== СКРИНШОТ ТЕРМИНАЛА (vision) ===\n"
                    "Приложен PNG всего экрана. Вывод команд в этом терминале "
                    "отрисован в canvas и в текстовом состоянии его НЕТ — "
                    "единственный источник информации об output, пути, хостах "
                    "и флаге вот эта картинка. Прочитай строки промпта и "
                    "результата последней команды буквально, не додумывай. "
                    "Команды шли через type без element_id (в фокус), "
                    "submit=true отправляет Enter. Если ответа ещё нет — "
                    "поставь wait 2-3 сек и читай скриншот снова."
                )
            else:
                user_prompt += (
                    "\n\n=== СКРИНШОТ СТРАНИЧКИ (vision) ===\n"
                    "Приложен PNG-скриншот viewport. На нём видно то, что видит "
                    "человек: картинки в клетках сетки, знаки, фигуры, цвета. "
                    "Текстовое состояние выше может НЕ отражать содержимое "
                    "картинок — верь скриншоту. Координаты на скриншоте "
                    "совпадают с координатами center=(x,y) в списке элементов."
                )
        full_prompt = AGENT_SYSTEM_PROMPT + "\n\n" + user_prompt

        try:
            # LLM-вызов ограничен по времени: зависший curl/SSE не должен
            # блокировать шаг агента навсегда. Доставка картинки в pplx
            # НЕСТАБИЛЬНА (флап ~20-70%), поэтому при ответе "не вижу" —
            # до 8 попыток с паузами, смена формулировок не нужна: та же
            # картинка доходит со временем.
            import asyncio as _aio
            response = ""
            vision_ok = False
            vision_attempts = 8 if screenshot_b64 else 1
            for vision_attempt in range(vision_attempts):
                if screenshot_b64:
                    response = await _aio.wait_for(
                        self.llm_client.chat_with_image(full_prompt, screenshot_b64),
                        timeout=120,
                    )
                    resp_low = (response or "").lower()
                    lost_markers = (
                        "не вижу", "не могу напрямую", "не могу увидеть",
                        "нет доступа", "прикрепи", "пришлите изображение",
                        "пришлите картинку", "не был загружен", "не загружено",
                        "issue with the model", "не могу посмотреть",
                        "не могу определить", "не увидел", "нет изображения",
                        "не видно", "не отображается", "не отображаются",
                        "не могу описать", "не виден", "не видна",
                        "наугад", "наудачу", "угадыва", "не могу разглядеть",
                        "не могу рассмотреть", "картинка не", "фото не",
                    )
                    if response and not any(m in resp_low for m in lost_markers):
                        vision_ok = True
                        break
                    logger.warning(f"Vision lost (attempt {vision_attempt + 1}/{vision_attempts}): "
                                   f"{(response or '')[:120]}")
                    import asyncio as _s
                    await _s.sleep(3 * (vision_attempt + 1) if vision_attempt > 2 else 2)
                else:
                    response = await _aio.wait_for(self.llm_client.chat(full_prompt), timeout=90)
                    vision_ok = True
                    break
            if not vision_ok:
                # Все попытки слепые: ЗАПРЕЩАЕМ действия по угадыванию.
                # Следующий шаг заново соберёт sheet и повторит vision.
                logger.warning("Vision lost after all attempts — действие заблокировано")
                return {"tool": "wait", "seconds": 5,
                        "thought": "Изображение не доставлено — жду и повторю захват"}
            logger.info(f"LLM raw (first 300): {response[:300]}")

            action = self._parse_response(response)
            if action:
                logger.info(f"Decision: {action.get('tool')} — {action.get('thought', '')[:100]}")
                return action

            logger.warning("Failed to parse LLM response")
            return {"tool": "wait", "thought": "Не удалось распознать ответ LLM", "seconds": 2}

        except Exception as e:
            logger.error(f"LLM decision error: {e}", exc_info=True)
            return {"tool": "wait", "thought": f"Ошибка LLM: {e}", "seconds": 3}

    # ─── Парсинг ────────────────────────────────────────────────────

    _COERCE_WORD_RE = re.compile(r"[a-zа-я_]{2,}", re.IGNORECASE)
    _COERCE_URL_RE = re.compile(r"https?://[^\s\"`,)\]]+", re.IGNORECASE)

    def _coerce_tool(self, raw, action, data):
        """tool из предложения/вложенного плана — только если имя реально
        названо (слово=tool/синоним) или действие однозначно по параметрам.
        Никогда не угадываем: не распознали — None (шаг отбракуется, как и
        раньше), чтобы не выполнить лишнего."""
        if self._COERCE_URL_RE.search(raw or ""):
            return "navigate"
        low = (raw or "").lower().strip()
        # двухсловные синонимы (go_to → «go to …»): Regex режет их на слова,
        # поэтому проверяем фразу целиком до разбиения
        for k, v in TOOL_SYNONYMS.items():
            if "_" in k:
                phrase = k.replace("_", " ")
                if low == phrase or low.startswith(phrase + " "):
                    return v
        ns = str(action.get("next_step") or data.get("next_step") or "")
        for w in (self._COERCE_WORD_RE.findall(raw.lower())
                  + self._COERCE_WORD_RE.findall(ns.lower())):
            if w in VALID_TOOLS:
                return w
            if w in TOOL_SYNONYMS:
                return TOOL_SYNONYMS[w]
        def has(*keys):
            return any(action.get(k) not in (None, "", [], {}) for k in keys)
        if str(action.get("target", "")).startswith("http"):
            return "navigate"
        if has("fields", "form_data", "inputs"):
            return "fill_form"
        if has("key"):
            return "key"
        if has("text", "value") and has("element_id", "eid", "selector"):
            return "type"
        if has("element_id", "eid", "selector"):
            return "click"
        return None

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Устойчивый парсинг JSON из ответа LLM"""
        # 1. Пробуем прямой парсинг
        parsed = self._try_json(response.strip())
        if parsed is not None:
            return self._normalize(parsed)

        # 2. Ищем JSON в markdown-блоках ```json ... ```
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL):
            parsed = self._try_json(m.group(1))
            if parsed is not None:
                return self._normalize(parsed)

        # 3. Ищем первый { ... последний }
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end > start:
            parsed = self._try_json(response[start:end + 1])
            if parsed is not None:
                return self._normalize(parsed)

        # 4. Обрезанный JSON (LLM не закрыл скобки) — дополняем
        start = response.find("{")
        if start != -1:
            fragment = response[start:]
            # Пробуем дополнить закрывающими скобками
            for suffix in ("}", '"}}', '"}', "}}"):
                parsed = self._try_json(fragment + suffix)
                if parsed is not None:
                    return self._normalize(parsed)

        return None

    def _try_json(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    def _normalize(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Нормализация: синонимы tool'ов, вложенные структуры, параметры.
        Поддерживает и плоский формат ({"tool": "click", "plan": [...], "thought": "..."}),
        и вложенный ({"plan": [...], "thought": "...", "action": {"tool": "click", ...}})."""

        # action может быть вложен или на верхнем уровне
        action = data.get("action")
        if not isinstance(action, dict):
            # Локальные модели (qwen3) иногда кладут действие в "plan" объектом
            # вместо массива шагов: {"plan": {"next_action": "click_at", ...}}
            if isinstance(data.get("plan"), dict):
                action = data["plan"]
        if not isinstance(action, dict):
            # Модель объявляет задачу выполненной без действия:
            # {"status": "completed", "message": "Сайт открыт. Задача завершена."}
            # Раньше такой ответ уходил в «Unknown tool», цикл делал 3 пустых
            # retry и вставал — хотя цель была достигнута.
            status = str(data.get("status") or data.get("result") or "").lower()
            if status in ("completed", "complete", "done", "finished", "success", "ok"):
                return {
                    "tool": "finish",
                    "reason": str(data.get("message") or data.get("reason")
                                  or data.get("thought") or "Задача завершена"),
                    "thought": str(data.get("thought") or ""),
                }
            # Плоский формат: {"type": "click", ...} или {"tool": "click", ...}
            action = dict(data)
            # Убираем служебные поля
            for k in ("plan", "thought", "reasoning"):
                action.pop(k, None)

        # Имя tool'а: tool / type / action / name / next_action / command
        tool = (action.get("tool") or action.get("type") or action.get("action") or
                action.get("name") or data.get("next_action") or action.get("next_action") or
                action.get("command") or "")
        tool = str(tool).lower().strip()

        # Синонимы
        if tool in TOOL_SYNONYMS:
            tool = TOOL_SYNONYMS[tool]

        # submit_form/fill_form с полями → fill_form (композитное действие)
        has_fields = any(
            isinstance(action.get(k), dict) and action[k]
            for k in ("fields", "form_data", "inputs")
        )
        if has_fields and tool in ("click", "type"):
            tool = "fill_form"

        # search → navigate с google
        if tool == "search":
            query = action.get("query") or action.get("text") or ""
            tool = "navigate"
            action["url"] = f"https://www.google.com/search?q={query}"

        # Модель часто пишет в action целое предложение («navigate https://x»,
        # «click e5 и отправить», «искать в скриптах…») вместо имени tool'а.
        # Раньше ответ отбраковывался как «Unknown tool», и цикл вставал после
        # 3 попыток — даже когда действие было валидным. Извлекаем tool БЕЗ
        # выдумывания: по слову/синониму, словам next_step или параметрам.
        if tool not in VALID_TOOLS:
            coerced = self._coerce_tool(tool, action, data)
            if coerced:
                logger.info(f"tool «{tool[:60]}» → «{coerced}»")
                tool = coerced

        if tool not in VALID_TOOLS:
            logger.warning(f"Unknown tool: {tool}")
            return None

        # fill_form: композитное действие — заполняет несколько полей и опционально отправляет
        if tool == "fill_form":
            fields = action.get("fields") or action.get("form_data") or action.get("inputs") or {}
            if isinstance(fields, dict) and fields:
                normalized: Dict[str, Any] = {
                    "tool": "fill_form",
                    "fields": {str(k): str(v) for k, v in fields.items()},
                    "form_id": str(action.get("form_id") or action.get("form") or ""),
                    "submit": bool(action.get("submit", True)),  # по умолчанию отправляем
                }
                normalized["thought"] = str(data.get("thought") or action.get("thought") or "")
                plan = data.get("plan")
                if isinstance(plan, list):
                    normalized["plan"] = [str(p) for p in plan[:10]]
                return normalized
            # Пустые поля → обычный type
            tool = "type"

        # Нормализация параметров
        normalized: Dict[str, Any] = {"tool": tool}

        if tool == "navigate":
            url = action.get("url") or action.get("target") or ""
            if isinstance(url, list) and url:
                url = url[0]
            normalized["url"] = _sanitize_url(url)

        elif tool == "click":
            # element_id / eid / selector / target
            eid = action.get("element_id") or action.get("eid") or action.get("target") or action.get("selector") or ""
            eid = str(eid).lstrip("#")
            normalized["element_id"] = eid
            if not (eid.startswith("e") and eid[1:].isdigit()):
                normalized["text_fallback"] = True

        elif tool in ("click_at", "hover"):
            # локальные модели пишут координаты по-разному: x/y,
            # coordinates=[x,y], position={x,y}
            coords = action.get("coordinates") or action.get("coords") or action.get("position")
            x = action.get("x")
            y = action.get("y")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                x, y = coords[0], coords[1]
            elif isinstance(coords, dict):
                x, y = coords.get("x"), coords.get("y")
            try:
                normalized["x"] = float(x or 0)
                normalized["y"] = float(y or 0)
            except (TypeError, ValueError):
                normalized["x"] = normalized["y"] = 0.0

        elif tool == "drag":
            normalized["x1"] = float(action.get("x1", action.get("from_x", 0)))
            normalized["y1"] = float(action.get("y1", action.get("from_y", 0)))
            normalized["x2"] = float(action.get("x2", action.get("to_x", 0)))
            normalized["y2"] = float(action.get("y2", action.get("to_y", 0)))
            normalized["steps"] = int(action.get("steps", 12))

        elif tool == "type":
            eid = str(action.get("element_id") or action.get("eid") or action.get("selector") or "").lstrip("#")
            normalized["element_id"] = eid
            normalized["text"] = str(action.get("text") or action.get("value") or "")
            # Печать «в фокус» для canvas-терминалов (HTB Pwnbox, noVNC): там нет
            # DOM-полей, поэтому element_id отсутствует намеренно — координаты
            # лишь ставят фокус кликом перед вводом.
            if not eid:
                cx = action.get("click_x", action.get("x"))
                cy = action.get("click_y", action.get("y"))
                if cx is not None:
                    normalized["click_x"] = float(cx)
                if cy is not None:
                    normalized["click_y"] = float(cy)
            # JSON иногда содержит строковые boolean-значения; bool("false") == True.
            def _as_bool(value: Any, default: bool) -> bool:
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
                return str(value).strip().lower() not in ("false", "0", "no", "нет", "")
            normalized["clear"] = _as_bool(action.get("clear"), True)
            normalized["submit"] = _as_bool(action.get("submit"), False)

        elif tool == "key":
            normalized["key"] = str(action.get("key") or "Enter")

        elif tool == "scroll":
            normalized["direction"] = str(action.get("direction") or "down")
            normalized["amount"] = int(action.get("amount") or 600)
            # element_id = id панели (pN) — крутить конкретный контейнер, а не окно
            pane = str(action.get("element_id") or action.get("pane") or "").strip()
            if pane:
                normalized["element_id"] = pane

        elif tool == "select":
            normalized["element_id"] = str(action.get("element_id") or action.get("eid") or "").lstrip("#")
            normalized["value"] = str(action.get("value") or action.get("text") or "")

        elif tool in ("new_tab",):
            normalized["url"] = _sanitize_url(action.get("url") or "")

        elif tool in ("switch_tab", "close_tab"):
            normalized["tab_id"] = str(action.get("tab_id") or action.get("target") or "")

        elif tool == "extract":
            normalized["selector"] = str(action.get("selector") or "body")

        elif tool == "wait":
            try:
                normalized["seconds"] = min(float(action.get("seconds") or 2), 10)
            except (ValueError, TypeError):
                normalized["seconds"] = 2

        elif tool == "wait_user":
            normalized["reason"] = str(action.get("reason") or action.get("text") or "нужны данные от пользователя")
            try:
                normalized["timeout"] = min(float(action.get("timeout") or 300), 900)
            except (ValueError, TypeError):
                normalized["timeout"] = 300

        elif tool == "evaluate":
            normalized["expression"] = str(action.get("expression") or action.get("code") or "")

        elif tool == "terminal":
            # Два режима: command=готовая shell-команда, prompt=задание Koda-агенту.
            command = str(action.get("command") or action.get("cmd") or
                          action.get("shell") or "").strip()
            prompt = str(action.get("prompt") or action.get("task") or
                         action.get("request") or "").strip()
            # Модель часто кладёт текст задания в text — развести по режимам.
            if not command and not prompt:
                loose = str(action.get("text") or "").strip()
                # Короткая строка с shell-признаками → команда, иначе задание.
                if loose and any(ch in loose for ch in ("|", ">", ";", "$", "&", "`")):
                    command = loose
                elif loose:
                    prompt = loose
            # Раскатываем после обоих источников (command и text=loose).
            command = strip_markdown_urls(command)
            if not command and not prompt:
                logger.warning("terminal: пустые command и prompt — действие отклонено")
                return None
            via = action.get("via_koda", action.get("koda"))
            if isinstance(via, str):
                via = via.strip().lower() not in ("false", "0", "no", "нет", "")
            if via is None:
                via = bool(prompt) and not command
            normalized["command"] = command
            normalized["prompt"] = prompt
            normalized["via_koda"] = bool(via)
            normalized["cwd"] = str(action.get("cwd") or action.get("workdir") or "").strip()
            try:
                normalized["timeout"] = float(action.get("timeout") or 0)
            except (ValueError, TypeError):
                normalized["timeout"] = 0.0

        elif tool == "sandbox":
            # Песочница = изолированный каталог: запуск команд/скриптов, файлы.
            # Режим задаётся mode; без него — запуск того, что прислали.
            command = strip_markdown_urls(str(action.get("command") or
                                              action.get("cmd") or "").strip())
            script = str(action.get("script") or action.get("code") or
                         action.get("source") or "").strip()
            filename = str(action.get("filename") or action.get("file") or
                           action.get("path") or "").strip()
            content = action.get("content")
            content = "" if content is None else str(content)
            mode = str(action.get("mode") or action.get("op") or "").strip().lower()
            # Подкоманды файлового меню приходят и как отдельные поля — по ним
            # тоже восстанавливаем режим, иначе mode потеряется и уйдёт в run.
            if not mode:
                if action.get("list") or action.get("ls"):
                    mode = "files"
                elif action.get("read"):
                    mode = "read"
                    filename = filename or str(action["read"])
                elif action.get("write") or content:
                    mode = "write"
                    filename = filename or str(action.get("write") or "")
                elif action.get("delete") is not None or action.get("clear"):
                    mode = "delete"
                elif action.get("info"):
                    mode = "info"
            if not mode:
                mode = "run"
            # Текст без явных полей: многострочный со синтаксисом → скрипт,
            # однострочный с shell-признаками → команда, иначе нечего выполнять.
            if mode == "run" and not (command or script or filename):
                loose = str(action.get("text") or "").strip()
                if "\n" in loose or ("def " in loose and ":" in loose):
                    script, mode = loose, "run"
                elif loose:
                    command, mode = strip_markdown_urls(loose), "run"
            if mode == "run" and not (command or script or filename):
                logger.warning("sandbox: пусто — нечего выполнять, действие отклонено")
                return None
            interp = str(action.get("interpreter") or action.get("runtime") or
                         action.get("lang") or "").strip().lower()
            iso = action.get("isolate")
            if isinstance(iso, str):
                iso = iso.strip().lower() not in ("false", "0", "no", "нет", "")
            normalized["mode"] = mode
            normalized["command"] = command
            normalized["script"] = script
            normalized["filename"] = filename
            normalized["content"] = content
            normalized["interpreter"] = interp or ("bash" if mode == "run" and command else "python3")
            normalized["path"] = str(action.get("path") or "").strip()
            if isinstance(iso, bool):
                normalized["isolate"] = iso
            try:
                normalized["timeout"] = float(action.get("timeout") or 0)
            except (ValueError, TypeError):
                normalized["timeout"] = 0.0

        elif tool == "finish":
            reason = (action.get("reason") or action.get("text") or
                      action.get("answer") or data.get("thought") or "Задача выполнена")
            normalized["reason"] = str(reason)

        # Мысль и план — на верхний уровень
        normalized["thought"] = str(data.get("thought") or action.get("thought") or "")
        plan = data.get("plan")
        if isinstance(plan, list):
            normalized["plan"] = [str(p) for p in plan[:10]]

        # Пометка об отвергнутых данных (вердикт LLM, рантайм запоминает)
        if action.get("rejected_value"):
            normalized["rejected_value"] = str(action["rejected_value"])
            if action.get("rejected_message"):
                normalized["rejected_message"] = str(action["rejected_message"])

        return normalized