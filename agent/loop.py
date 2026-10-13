"""
Agent Loop v2.1 - Comet-подобный цикл: observe → decide → act → verify → goal-check

Ключевые механики:
- GoalDetector: runtime определяет успех по объективным сигналам (URL-параметры
  формы, success-сообщения) — LLM не нужно самому решать "задача выполнена?"
- State-based loop breaker: same state + same action + no progress → LLM НЕ
  спрашивается повторно, вместо этого применяется recovery-стратегия
- LM retry: пустой ответ → retry, а не искусственный wait
"""

import asyncio
import base64
import logging
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

from agent.memory import AgentMemory
from agent.planner import AgentPlanner
from agent.goal_detector import GoalDetector
from agent.state_diff import diff_states, progress_verdict, verify_typed_value
from agent.terminal import terminal_from_env
from agent.sandbox import sandbox_from_env
from browser.controller import BrowserController
from browser.browser_state import capture_browser_state, BrowserState

logger = logging.getLogger(__name__)

# ─── Game sheet (vision для визуальных уровней not-a-robot) ─────────────
# Клетки уровня — DIV с background-image (спрайт) + background-position %.
# Кроп каждого спрайта собирается в canvas: одна картинка-«простыня» с
# подписями eID. Закон доставки pplx: PNG доходит нестабильно (флап ~20-70%),
# поэтому обязателен retry с верификацией ответа.
GAME_SHEET_JS = r"""async (tile) => {
    const cells = [];
    for (const el of document.querySelectorAll('[data-eid]')) {
        const st = el.getAttribute('style') || '';
        if (!st.includes('background-image')) continue;
        const mp = st.match(/background-position:\s*([\d.]+)%\s+([\d.]+)%/);
        const mi = st.match(/background-image:\s*url\(["']?([^"')]+)["']?\)/);
        const ms = st.match(/background-size:\s*([\d.]+)%\s+([\d.]+)%/);
        if (!mp || !mi) continue;
        cells.push({id: el.getAttribute('data-eid'), x: parseFloat(mp[1]), y: parseFloat(mp[2]),
                    img: new URL(mi[1], location.href).href,
                    sx: ms ? parseFloat(ms[1]) : 400, sy: ms ? parseFloat(ms[2]) : 400});
    }
    if (!cells.length) return {error: 'no cells'};
    const uniq = [...new Set(cells.map(c => c.img))];
    const imgs = {};
    for (const u of uniq) {
        try {
            const resp = await fetch(u);
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const im = new Image();
            im.src = url;
            await new Promise((res, rej) => { im.onload = res; im.onerror = rej; });
            imgs[u] = im;
        } catch (e) { return {error: 'sprite load fail: ' + u}; }
    }
    // Порядок клеток: по строкам (y), внутри строки по x — как видит человек
    const byRow = {};
    for (const c of cells) (byRow[c.y + '|' + c.img] = byRow[c.y + '|' + c.img] || []).push(c);
    const rows = Object.values(byRow).map(r => r.sort((a, b) => a.x - b.x));
    rows.sort((r1, r2) => r1[0].y - r2[0].y);
    const LBL = Math.max(16, Math.round(tile * 0.36));
    const cv = document.createElement('canvas');
    cv.width = tile * rows[0].length;
    cv.height = rows.length * (tile + LBL);
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, cv.width, cv.height);
    const order = [];
    rows.forEach((row, ri) => {
        row.forEach((cell, ci) => {
            const im = imgs[cell.img];
            const cols = Math.round(cell.sx / 100 * 4) || 4;
            const rowsN = Math.round(cell.sy / 100 * 4) || 4;
            const tw = im.naturalWidth / cols, th = im.naturalHeight / rowsN;
            const fx = cell.x / 100 * (im.naturalWidth - tw);
            const fy = cell.y / 100 * (im.naturalHeight - th);
            const dx = ci * tile, dy = ri * (tile + LBL);
            try { ctx.drawImage(im, fx, fy, tw, th, dx, dy, tile, tile); } catch (e) {}
            ctx.fillStyle = '#fff';
            ctx.font = 'bold ' + Math.round(LBL * 0.62) + 'px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(cell.id, dx + tile / 2, dy + tile + LBL * 0.75);
            order.push(cell.id);
        });
    });
    // PNG до ~130KB b64 надёжен; при превышении — повтор с меньшим tile
    let b64 = cv.toDataURL('image/png').split(',')[1];
    return {b64, order, w: cv.width, h: cv.height};
}"""


async def _capture_game_sheet(controller, tile: int = 48) -> Dict[str, Any]:
    """Собрать sheet-картинку игрового уровня (PNG b64 + порядок eID)."""
    page = controller.page
    from browser.browser_state import STATE_JS
    await asyncio.wait_for(page.evaluate(STATE_JS, 1), timeout=10)
    r = await asyncio.wait_for(page.evaluate(GAME_SHEET_JS, tile), timeout=20)
    return r or {}


class AgentState(Enum):
    """Состояния автомата агента"""
    RUNNING = "running"            # обычный цикл observe→decide→act
    WAITING_USER = "waiting_user"  # пауза wait_user (SMS-код, 2FA)
    RECOVERING = "recovering"      # recovery после зацикливания
    COMPLETED = "completed"
    FAILED = "failed"


class AgentLoop:
    """Основной цикл агента v2.1"""

    def __init__(
        self,
        controller: BrowserController,
        memory: AgentMemory,
        llm_client: Any,
        max_steps: int = 40,
        model: str = "claude47opus",
        on_progress=None,
        on_finish=None,
    ):
        self.controller = controller
        self.memory = memory
        self.llm_client = llm_client
        self.max_steps = max_steps
        # ── Умный бюджет шагов (вместо жёсткого «лимит N») ──────────────
        # max_steps — стартовый бюджет, а не жёсткий потолок: пока задача
        # даёт прогресс, step_budget растёт и прогон не обрывается на ровном
        # месте. Реальный стоп — не число шагов, а застой: stall_limit шагов
        # подряд без изменения страницы. hard_cap — абсолютный предохранитель
        # от бесконечного цикла.
        self.step_budget = max_steps
        self.stall_limit = max(10, max_steps // 3)
        self.hard_cap = max(200, max_steps * 5)
        self._stall_run = 0
        self._stalled = False
        # Состояния страницы, которые уже встречались: возврат в них — это
        # цикл (пинг-понг вкладок/страниц), а не прогресс, каким бы «изменением
        # URL» его ни рисовал state diff.
        self._seen_states: set = set()
        self.model = model
        self.task = ""
        self.planner = AgentPlanner(llm_client, model)
        self.goal_detector = GoalDetector()
        # Терминал: shell-команды и делегирование заданий Koda-агенту
        # (TERMINAL_ENABLED=0 отключает; tool terminal тогда вернёт ошибку).
        self.terminal = terminal_from_env(on_progress=on_progress)
        # Песочница: изолированный запуск скриптов и программ (agent/sandbox.py)
        self.sandbox = sandbox_from_env(on_progress=on_progress)
        self.running = False
        self.step_count = 0
        self.state = AgentState.RUNNING
        self.on_progress = on_progress or (lambda text: asyncio.sleep(0))
        self.on_finish = on_finish or (lambda reason: asyncio.sleep(0))
        # Анти-луп: история (state_key, action_key)
        self._state_action_history: List[tuple] = []
        self._recovery_used = 0
        # Пинг-понг блокирующей модалки: сколько раз одно и то же окно
        # открылось заново после закрытия (агент жмёт крестик вместо «Сохранить»)
        self._dialog_open_counts: Dict[str, int] = {}
        self._dialog_prev_sig: Optional[str] = None
        self._dialog_hint_given: set = set()
        # После успешной записи кода запрещаем повторную печать в тот же редактор.
        # Это защищает от LLM-loop и сброса уже принятого решения.
        self._typed_editors: Dict[str, str] = {}
        self._rejected_credentials: Dict[str, str] = {}
        self._user_input_event: Optional[asyncio.Event] = None
        self._user_input_value: Optional[str] = None
        # Сообщения пользователя во время работы агента (код из SMS и т.п.)
        self._user_messages: List[str] = []
        # План, возвращённый LLM на прошлом шаге. Без его передачи модель
        # каждый шаг сочиняет план с нуля и снова берётся за первый пункт —
        # отсюда повторяющиеся «определить тип страницы» на 20-м шаге.
        self._current_plan: List[str] = []
        # Сколько шагов подряд модель называет тот же пункт плана «текущим»
        self._plan_stall: Dict[str, int] = {}
        # Для детекции план-зацикливания: последние N сигнатур планов (frozenset)
        self._plan_history: List[frozenset] = []
        self._plan_repeat_streak: int = 0
        # Повтор-детектор: сколько раз действие с тем же смысловым отпечатком
        # не дало прогресса. element_id в счёт не идёт — он пересоздаётся на
        # каждом снимке, и два разных клика выглядели бы одним повтором.
        self._no_progress_streak: Dict[str, int] = {}

    # ─── Повтор-детектор ────────────────────────────────────────────
    # Значимые параметры: те, что отличают одно действие другого по смыслу.
    _FP_PARAMS = ("url", "key", "command", "prompt", "script", "text", "path",
                  "filename", "mode", "direction", "tab_id", "field_name",
                  "form_id", "value")

    def _element_fingerprint(self, state: BrowserState, raw_eid: Any) -> str:
        """Смысловая подпись элемента вместо нестабильного element_id."""
        el = state.find_element(str(raw_eid or "").strip())
        if el is None:
            return f"eid?{raw_eid}"
        raw_text = (el.get("text") or el.get("aria") or el.get("placeholder")
                    or el.get("name") or "")
        text = " ".join(str(raw_text).split())[:40].lower()
        return f"{el.get('tag', '')}|{text}|{str(el.get('href') or '')[:60]}"

    def _action_fingerprint(self, action: Dict[str, Any],
                            state: BrowserState) -> str:
        """Отпечаток действия: tool + значимые параметры + url.

        element_id заменяется подписью элемента, координаты округляются до
        8px (в пределах одной кнопки) — иначе один и тот же клик каждый раз
        выглядел бы новым действием и детектор молчал бы.
        """
        parts = [str(action.get("tool", "?")), str(state.url)]
        for param in self._FP_PARAMS:
            if action.get(param) not in (None, ""):
                parts.append(f"{param}={str(action[param]).strip()[:120]}")
        for param in ("element_id",):
            if action.get(param) not in (None, ""):
                parts.append(self._element_fingerprint(state, action[param]))
        for param in ("x", "y", "x1", "y1", "x2", "y2", "click_x", "click_y"):
            val = action.get(param)
            if isinstance(val, (int, float)):
                # Бакет 32px ≈ размер типовой кнопки: клики в её пределах —
                # одно и то же действие, а не «новое» из-за дрожания на пиксель.
                parts.append(f"{param}={int(val // 32)}")
        return "|".join(parts).lower()

    def _repeat_guard(self, action: Dict[str, Any],
                      state: BrowserState) -> Optional[Dict[str, Any]]:
        """Блокировка действия, которое уже N раз подряд не дало прогресса.

        Возвращает результат-ошибку с подсказкой либо None (разрешено).
        Loop-breaker (_is_stuck) такое не ловит: он требует одинакового
        state_key, а при перерисовке DOM состояние «новое», хотя модель
        жмёт в ту же кнопку.
        """
        if action.get("tool") in ("finish", "wait", "wait_user", "extract"):
            return None
        fp = self._action_fingerprint(action, state)
        fails = self._no_progress_streak.get(fp, 0)
        # fails == 2 означает: два предыдущих таких действия не дали ничего,
        # третье — уже слепой повтор.
        if fails < 2:
            return None
        return {
            "ok": False,
            "error": (
                f"ПОВТОР ЗАБЛОКИРОВАН: это действие уже {fails} раза подряд "
                f"не дало никакого результата (страница не изменилась). "
                f"Оно не сработает и в третий раз — причина не в том, что "
                f"нужно нажать сильнее. Смени подход: перечитай состояние "
                f"страницы и выбери ДРУГОЙ элемент/другой инструмент "
                f"(extract, evaluate, scroll, navigate, terminal) либо "
                f"wait_user, если не хватает данных."
            ),
        }

    def _track_repeat(self, action: Dict[str, Any], state: BrowserState,
                      progress: Optional[bool]) -> None:
        """Обновить счётчик безрезультатных повторов по итогам шага."""
        if progress is True:
            self._no_progress_streak.clear()
            return
        if progress is False:
            fp = self._action_fingerprint(action, state)
            self._no_progress_streak[fp] = self._no_progress_streak.get(fp, 0) + 1

    def _element_label(self, state: BrowserState, raw_eid: Any) -> str:
        """Короткая человекочитаемая подпись элемента — для истории."""
        el = state.find_element(str(raw_eid or "").strip())
        if el is None:
            return ""
        text = " ".join(str(el.get("text") or el.get("aria")
                          or el.get("placeholder") or "").split())[:40]
        tag = el.get("tag") or el.get("role") or "el"
        return f"{tag} «{text}»" if text else str(tag)

    def stop(self):
        """Остановка цикла агента"""
        self.running = False
        self.state = AgentState.FAILED
        # Долгоиграющая терминальная команда не должна переживать остановку.
        try:
            self.terminal.cancel()
            self.sandbox.cancel()
        except Exception:
            pass
        logger.info("Agent loop stop requested")

    async def run(self, task: str):
        """Запуск агента с задачей"""
        self.task = task
        self.running = True
        self.step_count = 0
        self.state = AgentState.RUNNING
        self._state_action_history = []
        self._recovery_used = 0
        self._dialog_open_counts = {}
        self._dialog_prev_sig = None
        self._dialog_hint_given = set()
        self._typed_editors = {}
        self._rejected_credentials = {}
        self._no_progress_streak = {}
        self._user_input_event = None
        self._user_input_value = None
        self._user_messages = []
        self._current_plan = []
        self._plan_stall = {}
        self._plan_history = []
        self._plan_repeat_streak = 0
        self.step_budget = self.max_steps
        self._stall_run = 0
        self._stalled = False
        self._seen_states = set()
        self.memory.add_goal(task)

        logger.info(f"Agent loop started: {task[:100]}")
        await self._progress(f" Задача: {task}")

        last_result = "(первый шаг)"
        state_before: Optional[BrowserState] = None

        try:
            while (self.running and not self._stalled
                   and self.step_count < self.step_budget
                   and self.step_count < self.hard_cap):
                self.step_count += 1
                logger.info(
                    f"Step {self.step_count} (бюджет {self.step_budget}, "
                    f"застой {self._stall_run}/{self.stall_limit}, "
                    f"потолок {self.hard_cap})")

                # 1. OBSERVE
                state = await self._observe()
                if state is None:
                    # Страница не отвечает (CDP-канал умер/Chrome закрыт) —
                    # не убиваем задачу: пробуем переподключиться к вкладке.
                    await self._progress("[WARN] Страница не отвечает — попытка переподключения")
                    try:
                        # refresh_tabs/title могут зависнуть на мёртвом канале —
                        # ограничиваем всю попытку общим таймаутом
                        async def _light_reconnect():
                            await self.controller.refresh_tabs()
                            await self.controller._pick_active_page()
                            return await self._observe()
                        state = await asyncio.wait_for(_light_reconnect(), timeout=25)
                    except Exception as e:
                        logger.warning(f"Reconnect attempt failed: {e}")
                if state is None:
                    # Chrome мог умереть целиком (TAL/краш) ИЛИ быть живым без
                    # CDP (запущен вручную) — revive различит оба случая.
                    state = await self._revive_chrome()
                if state is None:
                    # Последний шанс: CDP-канал сломан на уровне Playwright
                    # (page.evaluate висит при живом CDP). Полный реконнект.
                    logger.warning("Capture failed with responsive CDP — full Playwright reconnect")
                    try:
                        await self.controller.disconnect()
                        await self.controller.connect(mode="attach")
                        if "not-a-robot" in self.task:
                            await self.controller.navigate("https://neal.fun/not-a-robot/")
                            await asyncio.sleep(2)
                        state = await self._observe()
                    except Exception as e:
                        logger.error(f"Full reconnect failed: {e}")
                if state is None:
                    await self._progress("[ERROR] Не удалось получить состояние страницы")
                    break
                state_before = state

                # 1.3 БЛОКИРУЮЩЕЕ МОДАЛЬНОЕ ОКНО: то же окно открылось заново
                # после того, как агент его «закрыл» → подсказка вместо шага впустую.
                dialog_hint = self._track_dialog(state)
                if dialog_hint:
                    last_result = dialog_hint

                # 1.5 GOAL CHECK (state-only, ДО LLM): цель могла быть достигнута
                # без агента — пользователь сам ввёл код во время wait_user,
                # сессия уже авторизована, произошёл redirect. Runtime решает сам.
                goal_reason = self.goal_detector.check_state(task, state)
                if goal_reason:
                    self.state = AgentState.COMPLETED
                    await self._progress(f"[GOAL] Цель достигнута (runtime, без LLM): {goal_reason}")
                    self.memory.complete_goal(task)
                    self.memory.add_action("goal_check", {"goal_completed": True, "reason": goal_reason})
                    await self.on_finish(goal_reason)
                    break

                # 2. LOOP BREAKER: то же состояние + то же действие → recovery без LLM
                if self._is_stuck(state):
                    self.state = AgentState.RECOVERING
                    recovery = await self._apply_recovery(state)
                    if recovery == "stop":
                        self.state = AgentState.FAILED
                        break
                    last_result = recovery  # подсказка для LLM на следующем шаге
                    self.state = AgentState.RUNNING
                    continue

                # 3. DECIDE (с retry при пустом ответе)
                # Визуальные уровни игры: текстовое состояние не показывает
                # содержимое картинок — собираем sheet (кропы спрайтов с
                # подписями eID) и даём модели vision.
                screenshot_b64 = ""
                sheet_order = []
                if state.game.get("detected") or state.game.get("terminal"):
                    # Терминал: только viewport (sheet клеток здесь бессмысленен)
                    screenshot_b64, sheet_order = await self._capture_vision_image(
                        use_sheet=bool(state.game.get("detected")))
                    if screenshot_b64:
                        logger.info(f"Vision image captured ({len(screenshot_b64)} b64 chars, "
                                    f"{len(sheet_order)} cells, "
                                    f"terminal={bool(state.game.get('terminal'))})")

                await self._progress("[THINKING] ")
                action = await self._decide_with_retry(
                    task, state, last_result,
                    rejected_warning=self._rejected_warning(),
                    user_messages=self._drain_user_messages(),
                    screenshot_b64=screenshot_b64,
                    sheet_order=sheet_order,
                    current_plan=self._current_plan,
                    cycle_warning=self._plan_cycle_warning(),
                    screenshot_kind=("grid" if state.game.get("detected")
                                     else "terminal"),
                )
                if action is None:
                    await self._progress("[ERROR] LLM недоступен после 3 попыток — остановка")
                    await self.on_finish("LLM не отвечает (пустые ответы после 3 retry)")
                    break

                tool = action.get("tool", "?")
                thought = action.get("thought", "")
                plan = action.get("plan", [])

                # LM может пометить значение как отвергнутое сайтом (её вердикт)
                if action.get("rejected_value"):
                    self.remember_rejected_value(
                        action["rejected_value"],
                        action.get("rejected_message") or thought,
                    )
                    await self._progress(f"   [REJECTED] Запомнено: {action['rejected_value']} отвергнут сайтом")

                await self._progress(f"[STEP] Шаг {self.step_count}: {tool}")
                if thought:
                    await self._progress(f"[THOUGHT] {thought[:2000]}")
                if plan:
                    await self._progress(f"[PLAN] {' → '.join(plan[:5])}")
                    # Сохраняем план и считаем повторы (анти-зацикливание)
                    self._track_plan(plan)

                # 4. ACT
                # 4.0 Анти-пинг-понг: отмена обязательного модального окна
                # запрещена. Дальше сработает обычный loop-breaker, если модель
                # продолжит настаивать на отмене.
                blocked = self._dialog_guard(action, state)
                if blocked is None:
                    blocked = self._repeat_guard(action, state)
                if blocked is not None:
                    result = blocked
                elif tool == "type" and str(action.get("element_id") or "").strip():
                    eid = action.get("element_id", "")
                    text = action.get("text", "")
                    previous = self._typed_editors.get(eid)
                    if previous is not None and previous == text:
                        result = {
                            "ok": False,
                            "error": f"редактор {eid} уже заполнен и проверен; повторная запись заблокирована — отправь решение",
                        }
                    elif previous is not None and state.url.startswith("https://assessment.hh.ru/code/"):
                        result = {
                            "ok": False,
                            "error": f"редактор {eid} уже изменён; повторная запись заблокирована до проверки решения",
                        }
                    else:
                        result = await self._execute(action, state)
                elif tool == "type":
                    # Ввод в фокус: защита от повторов не нужна — в терминале
                    # повторить команду (ls, cd ..) это нормальное действие.
                    result = await self._execute(action, state)
                else:
                    result = await self._execute(action, state)

                # 4.5 Цель достигнута прямо в _execute (wait_user → пользователь
                # сам завершил вход). Runtime завершает БЕЗ LLM.
                if result.get("goal_completed"):
                    self.state = AgentState.COMPLETED
                    reason = result.get("goal_reason") or "Цель достигнута во время ожидания"
                    await self._progress(f"[GOAL] {reason}")
                    self.memory.complete_goal(task)
                    self.memory.add_action(tool, {"result": result, "goal_completed": True})
                    await self.on_finish(reason)
                    break

                # 9. Завершение по инициативе LLM (до observe — страница не менялась)
                if tool == "finish":
                    self.state = AgentState.COMPLETED
                    reason = action.get("reason") or action.get("text") or "Задача выполнена"
                    self.memory.complete_goal(task)
                    await self._progress(f" {reason}")
                    self.memory.add_action("finish", {"reason": reason})
                    await self.on_finish(reason)
                    break

                # 5. OBSERVE (после навигационных действий ждём стабильности страницы)
                navigational = action.get("tool") in ("navigate", "click", "click_at", "hover", "drag", "type", "fill_form", "key", "select")
                state_after = await self._observe(wait_stable=navigational)

                # 5.5 STATE DIFF: изменилось ли состояние реально
                diff = diff_states(state, state_after)

                # Повторный diff с задержкой: UI-анимации (модалки) появляются не мгновенно
                if navigational and not diff.get("changed") and state_after is not None:
                    await asyncio.sleep(1.5)
                    state_after2 = await self._observe()
                    if state_after2 is not None:
                        diff2 = diff_states(state, state_after2)
                        if diff2.get("changed"):
                            diff = diff2
                            state_after = state_after2
                            logger.info("State diff: changes detected on delayed re-observe")

                verdict = progress_verdict(diff, action)

                # Терминал: страница не при чём — verdict по результату команды,
                # иначе успешная команда выглядела бы как NO_PROGRESS.
                if action.get("tool") in ("terminal", "sandbox"):
                    verdict = (
                        {"progress": True, "message": "команда выполнена"}
                        if result.get("ok") else
                        {"progress": False,
                         "message": result.get("error") or "команда завершилась с ошибкой"}
                    )

                # Ввод текста: прогресс = значение поля установилось (а не diff страницы).
                # Для clear=true проверяем полное совпадение, чтобы не принять дописывание
                # к исходному примеру за успешный ввод.
                # Печать в фокус (canvas-терминал) не имеет DOM-поля: сверять
                # значение негде, вывод подтверждается следующим скриншотом.
                if (action.get("tool") == "type" and result.get("ok")
                        and str(action.get("element_id") or "").strip()):
                    # Controller проверяет значение на том же locator до React-
                    # перерисовки. Повторный поиск по старому eid здесь может
                    # ложно провалиться, если редактор выдал новый data-eid.
                    if result.get("verified_on_same_locator"):
                        typed = {"ok": True, "message": "значение установлено и проверено в редакторе"}
                    else:
                        typed = await verify_typed_value(
                            self.controller,
                            action.get("element_id", ""),
                            action.get("text", ""),
                        )
                    if typed["ok"]:
                        self._typed_editors[action.get("element_id", "")] = action.get("text", "")
                        verdict = {"progress": True, "message": typed["message"]}
                    else:
                        result["ok"] = False
                        result["error"] = typed["message"]
                        verdict = {"progress": False, "message": typed["message"]}

                # 6. VERIFY
                verified = await self._verify(action, result, state, state_after)
                result_text = self._format_result(tool, result, verified, verdict, diff)
                await self._progress(f"   {result_text}")

                # 7. GOAL CHECK: runtime определяет успех объективно
                goal_reason = self.goal_detector.check(
                    task, action, result, state, state_after
                )
                if goal_reason:
                    self.state = AgentState.COMPLETED
                    await self._progress(f"[GOAL] Цель достигнута (runtime): {goal_reason}")
                    self.memory.complete_goal(task)
                    self.memory.add_action(tool, {"result": result, "goal_completed": True})
                    await self.on_finish(goal_reason)
                    break

                last_result = result_text

                # Данные от пользователя (код из SMS) → явно в результат,
                # чтобы LM на следующем шаге ввела их в поле
                if result.get("user_input"):
                    last_result = (
                        f" Пользователь передал данные: «{result['user_input']}». "
                        f"Введи их в поле ввода кода (type) и подтверди."
                    )

                # 7.5 Выделенные сообщения страницы → в результат действия.
                # Семантику (ошибка/успех/инфо) оценивает LLM сама по тексту и цвету.
                new_notices = (diff.get("notices_appeared") or []) if diff else []
                if new_notices:
                    last_result = result_text + (
                        " | [NOTICE] На странице появились выделенные сообщения: "
                        + " | ".join(new_notices)
                        + " — прочитай и учти в следующем действии."
                    )
                    await self._progress(f"   [NOTICE] Сообщения: {new_notices[0][:100]}")

                elif tool == "wait_user":
                    self.state = AgentState.RUNNING  # вышли из ожидания

                # 7.8 Скачанный файл = данные задачи. Кнопки вида «Download
                # Task Files» не ведут на ссылку: файл уходит в загрузку
                # браузера, DOM не меняется. Без явного сообщения агент видит
                # NO_PROGRESS и идёт искать материалы в чужих walkthrough.
                if tool in ("click", "click_at", "key", "type", "fill_form"):
                    dl = await self._collect_downloads()
                    if dl:
                        result["downloads"] = dl
                        files = ", ".join(
                            f"«{d['rel']}» ({d['size']} байт)" for d in dl)
                        await self._progress(f"[FILE] Скачано: {files}")
                        last_result = (
                            f"[FILE] Браузер скачал файл(ы) задачи: {files}. "
                            f"Это ДАННЫЕ задачи, они уже в песочнице. Дальше: "
                            f"{{\"tool\":\"sandbox\",\"mode\":\"files\"}} — список; "
                            f"для архива {{\"tool\":\"sandbox\",\"mode\":\"run\","
                            f"\"command\":\"unzip -o '{dl[0]['rel']}'\"}}; затем "
                            f"{{\"tool\":\"sandbox\",\"mode\":\"read\",\"path\":\"<файл>\"}}. "
                            f"Решай задание по содержимому этих файлов, а не по "
                            f"walkthrough из поиска."
                        )
                        verdict = {"progress": True,
                                   "message": f"скачаны файлы задачи: {files}"}

                # 8. Учёт для анти-лупа
                self._record(state, action, state_after)

                # 8.5 Повтор-детектор + подпись элемента в истории: без подписи
                # «click e1» и «click e1» — два разных элемента, и модель не
                # понимает, что уже жмёт то же самое.
                step_progress = verdict.get("progress") if verdict else None
                self._track_repeat(action, state, step_progress)
                self._update_budget(self._budget_progress(step_progress,
                                                          state_after))
                target_label = ""
                if action.get("element_id"):
                    target_label = self._element_label(
                        state, action.get("element_id"))

                self.memory.add_step(tool, plan, thought, action, result,
                                     verified=verified,
                                     progress=step_progress,
                                     target=target_label)

                await asyncio.sleep(0.5)

            else:
                if getattr(self, "_stalled", False):
                    reason = (f"Задача остановлена: {self._stall_run} шагов "
                              f"подряд без изменения страницы (застой). "
                              f"Прежний подход не работает — нужна смена "
                              f"стратегии или уточнение задачи.")
                elif self.step_count >= self.hard_cap:
                    reason = (f"Задача не решена: жёсткий потолок "
                              f"{self.hard_cap} шагов — признак "
                              f"бесконечного цикла.")
                elif not self.running:
                    reason = "Задача остановлена пользователем."
                else:
                    reason = (f"Задача не решена: бюджет {self.step_budget} "
                              f"шагов исчерпан без прогресса.")
                await self._progress(f" {reason}")
                await self.on_finish(reason)

        except Exception as e:
            logger.error(f"Agent loop error: {e}", exc_info=True)
            self.memory.add_error(str(e))
            await self._progress(f"[ERROR] Ошибка: {e}")
            await self.on_finish(f"Ошибка: {e}")

        finally:
            self.running = False
            self.memory.save()
            self.memory.save_history()

    # ─── Vision: сборка картинки уровня ─────────────────────────────

    async def _capture_vision_image(self, use_sheet: bool = True) -> tuple:
        """PNG-картинка уровня для vision. Сперва sheet клеток (кропы
        спрайтов + подписи eID); если клеток нет — viewport как PNG.
        Возвращает (b64, order) — order соответствует порядку клеток на sheet
        (строки сверху вниз, слева направо).

        use_sheet=False — сразу viewport целиком: так читают canvas-терминалы
        (Pwnbox/noVNC), где кроить спрайты не из чего и нужен весь экран."""
        if not use_sheet:
            try:
                shot = await asyncio.wait_for(
                    self.controller.page.screenshot(type="png"), timeout=10)
                return base64.b64encode(shot).decode("ascii"), []
            except Exception as e:
                logger.warning(f"Viewport screenshot failed (terminal): {e}")
                return "", []
        try:
            r = await _capture_game_sheet(self.controller, tile=48)
            if r.get("b64") and r.get("order"):
                return r["b64"], r["order"]
            logger.info(f"Sheet unavailable ({r.get('error')}), fallback to viewport PNG")
        except Exception as e:
            logger.warning(f"Sheet build failed: {e} — fallback to viewport PNG")
        try:
            shot = await asyncio.wait_for(
                self.controller.page.screenshot(type="png"),
                timeout=10,
            )
            b64 = base64.b64encode(shot).decode("ascii")
            # Viewport PNG может превышать порог доставки — уменьшать некем,
            # отдаём как есть (лучше шанс, чем ничего).
            return b64, []
        except Exception as e:
            logger.warning(f"Screenshot failed (продолжаю без vision): {e}")
            return "", []

    # ─── Решение с retry ────────────────────────────────────────────

    async def _decide_with_retry(self, task, state, last_result,
                                  rejected_warning: str = "",
                                  user_messages: str = "",
                                  screenshot_b64: str = "",
                                  sheet_order: Optional[List[str]] = None,
                                  current_plan: Optional[List[str]] = None,
                                  cycle_warning: str = "",
                                  screenshot_kind: str = "grid") -> Optional[Dict[str, Any]]:
        """Решение LLM с retry при пустом/нечитаемом ответе.
        При vision-запросе retry тоже идёт со скриншотом.
        sheet_order — порядок eID клеток на приложенной картинке.
        current_plan — план с прошлого шага (продолжить, а не сочинять заново).
        cycle_warning — текст-предупреждение о зацикленном плане."""
        tabs_text = self._format_tabs()
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            action = await self.planner.decide(
                task=task,
                browser_state_text=state.to_llm_text(),
                tabs_text=tabs_text,
                last_action_result=last_result,
                recent_history=self.memory.get_recent_actions(5),
                step=self.step_count,
                rejected_warning=rejected_warning,
                user_messages=user_messages,
                screenshot_b64=screenshot_b64,
                sheet_order=sheet_order or [],
                current_plan=current_plan,
                cycle_warning=cycle_warning,
                screenshot_kind=screenshot_kind,
            )
            # wait с пометкой "пустой ответ" = LM не сработал. Важно: мысль
            # вида "список элементов пуст" — легитимное решение, не сбой.
            thought = (action.get("thought") or "").lower()
            is_empty_fallback = (
                action.get("tool") == "wait"
                and ("пустой ответ" in thought or "error" in thought
                     or "ошибка llm" in thought or "ошибка lm" in thought
                     or "не удалось распознать" in thought
                     or "не настроен" in thought or "не установлен" in thought)
            )
            if not is_empty_fallback:
                return action

            logger.warning(f"LLM empty/invalid response (attempt {attempt}/{max_attempts})")
            await self._progress(f"    LLM вернул пустой ответ, повтор {attempt}/{max_attempts - 1}")
            await asyncio.sleep(1.5 * attempt)

        return None

    # ─── State-based loop breaker ───────────────────────────────────

    async def _attach_tabs(self, state: Optional[BrowserState]) -> None:
        """Прикрепить к снимку страницы список вкладок браузера.

        capture_browser_state читает одну страницу и про вкладки не знает.
        Без списка вкладок агент не видит, что клик открыл лабу в новой
        вкладке, и бесконечно жмёт ту же кнопку (PortSwigger ACCESS THE LAB).
        title() зависшей вкладки не должен вешать шаг — таймаут + откат к
        последнему снимку controller._tabs.
        """
        if state is None:
            return
        import asyncio as _aio
        try:
            tabs = await _aio.wait_for(self.controller.refresh_tabs(), timeout=5)
        except Exception as e:
            logger.debug(f"refresh_tabs для состояния не удалась: {e}")
            tabs = getattr(self.controller, "_tabs", []) or []
        state.tabs = [
            {"tab_id": t.tab_id, "url": t.url, "title": t.title,
             "is_active": t.is_active}
            for t in tabs
        ]

    def _state_key(self, state: BrowserState) -> str:
        """Ключ состояния: URL + scroll + позиции панелей + первые id элементов"""
        eids = ",".join(el["eid"] for el in state.elements[:15])
        game = state.game or {}
        # Прокрутка внутренних панелей: на split-screen (TryHackMe) window.scrollY
        # не меняется никогда, единственный признак движения — scrollTop панели.
        panes = ",".join(f"{p.get('id')}:{p.get('scroll_y')}"
                         for p in (getattr(state, "scroll_panes", []) or [])[:4])
        # Активная вкладка — часть состояния: переход tab1→tab2 (та же лаба,
        # тот же URL в другой вкладке) обязан выглядеть как изменение, иначе
        # анти-loop считает переключение вкладок застоем.
        active_tab = next((t.get("tab_id") for t in (state.tabs or [])
                           if t.get("is_active")), "")
        n_tabs = len(state.tabs or [])
        return (f"{state.url}|{state.scroll_y}|{panes}|{eids}|{game.get('level')}"
                f"|{game.get('terminal') and len(state.elements)}|{active_tab}|{n_tabs}")

    def _action_key(self, action: Dict[str, Any]) -> str:
        key = action.get("tool", "?")
        # text обязателен: в canvas-терминале DOM-состояние не меняется, и без
        # текста ls / cat / id выглядели бы как один и тот же повтор → анти-loop
        # прерывал бы работу терминала на третьей команде.
        for param in ("url", "element_id", "key", "tab_id", "direction", "x", "y", "x1", "y1", "x2", "y2",
                      "command", "prompt", "script", "filename", "path", "mode",
                      "text", "click_x", "click_y"):
            if param in action:
                key += f":{action[param]}"
        return key

    def _budget_progress(self, step_progress: Optional[bool],
                         state_after: Optional[BrowserState]) -> Optional[bool]:
        """Прогресс для бюджета = ШАГ ВПЕРЁД, а не «что-то мигало».

        Пинг-понг между двумя вкладками/страницами меняет URL на каждом
        шаге, и state diff честно даёт progress=True — но задача стоит на
        месте и бюджет рос бы вечно. Возврат в уже виденное состояние
        страницы понижается до застоя (False).
        """
        if state_after is None:
            return step_progress
        key_after = self._state_key(state_after)
        repeat = step_progress and key_after in self._seen_states
        self._seen_states.add(key_after)
        return False if repeat else step_progress

    def _update_budget(self, progress: Optional[bool]) -> None:
        """Умный бюджет: растёт на прогрессе, стопает на застое.

        progress=True  → задача двигается: расширяем бюджет (не даём
                         счётчику оборвать успешный прогон) и обнуляем
                         счётчик застоя.
        progress=False → шаг без изменения страницы: копить застой; после
                         stall_limit подряд помечаем _stalled — это реальный
                         стоп вместо произвольного «лимит N».
        """
        if progress is None:
            # Шаг без вердикта (wait/extract/wait_user) — не прогресс и не
            # застой: не трогаем счётчик, иначе чтение страницы «съело» бы
            # бюджет и остановило живую задачу.
            return
        if progress:
            self._stall_run = 0
            self.step_budget = max(self.step_budget,
                                   self.step_count + max(5, self.max_steps // 4))
        else:
            self._stall_run += 1
            if self._stall_run >= self.stall_limit:
                self._stalled = True

    def _record(self, state: BrowserState, action: Dict[str, Any],
                state_after: Optional[BrowserState]):
        """Записать (состояние, действие) и прогресс"""
        pair = (self._state_key(state), self._action_key(action))
        self._state_action_history.append(pair)
        self._state_action_history = self._state_action_history[-8:]

    def _is_stuck(self, state: BrowserState) -> bool:
        """
        Застряли? Последние 3 записи: то же состояние + то же действие.
        Это значит: действие выполняется, но состояние страницы не меняется.

        Второй случай — пинг-понг вкладок: четыре switch_tab подряд. Состояние
        каждый раз «новое» (вкладки-то разные), поэтому правило про одинаковые
        записи такое не ловит, а задача не двигается вообще.
        """
        if len(self._state_action_history) < 3:
            return False
        last3 = self._state_action_history[-3:]
        first_state, first_action = last3[0]
        if all(s == first_state and a == first_action for s, a in last3):
            return True
        actions = [a.split(":")[0] for _, a in self._state_action_history[-4:]]
        return len(actions) == 4 and all(a == "switch_tab" for a in actions)

    async def _apply_recovery(self, state: BrowserState) -> str:
        """
        Recovery-стратегия вместо повторного вопроса к LLM.
        Возвращает текст-подсказку для LLM на следующем шаге, или "stop".
        """
        self._recovery_used += 1
        if self._recovery_used > 3:
            reason = "Агент застрял: 3 recovery-попытки не дали прогресса"
            await self._progress(f"[STOP] {reason}")
            await self.on_finish(reason)
            return "stop"

        # Определяем какое действие застряло
        stuck_action = self._state_action_history[-1][1].split(":")[0]

        # Приоритет: обязательная модалка — самая частая причина «клика без
        # прогресса» (модель жмёт крестик, runtime его блокирует).
        if self._dialog_sig(state):
            hint = self._track_dialog(state) or (
                "RECOVERY: открыто обязательное модальное окно. Не отменяй его — "
                "выполни требование окна (заполни поле и нажми «Сохранить и продолжить»)."
            )
            await self._progress("[DIALOG] Требуется действие в модальном окне")
            self._state_action_history = []
            return hint

        await self._progress(f" Нет прогресса после 3× {stuck_action} — recovery #{self._recovery_used}")

        # Сбрасываем историю, чтобы дать LLM шанс с новой подсказкой
        self._state_action_history = []

        if stuck_action == "switch_tab":
            tabs = ", ".join(
                f"[{t.tab_id}] {t.url[:60]}" for t in self.controller._tabs[:4])
            lab_hint = ""
            off = [m for m in ((state.lab or {}).get("machines") or [])
                   if m.get("status") == "OFF"]
            if off:
                btn = ", ".join(
                    f"[{m['eid']}] «{m['button']}»" for m in off if m.get("eid"))
                lab_hint = (
                    " На странице карточек машин видно Status: Off — окружение "
                    f"НЕ запущено. Первым делом нажми кнопку запуска {btn} "
                    "и дождись Status: On (wait 20-30с). URL выключенной машины "
                    "открывать бессмысленно — будет 502 Bad Gateway."
                    if btn else
                    " На странице указано, что лабораторная машина выключена — "
                    "сначала запусти её, потом открывай URL."
                )
            hint = (
                "RECOVERY: ты переключал вкладки туда-сюда и это не действие — "
                "переключение ничего на странице не меняет. Останься на ОДНОЙ "
                "вкладке и сделай настоящее действие." + lab_hint +
                " Вкладки: " + tabs +
                " Если реальное действие уже выполнялось и результата нет — "
                "finish с честным итогом: что известно и где застопорился."
            )
        elif stuck_action == "scroll":
            panes = getattr(state, "scroll_panes", []) or []
            if panes:
                plist = ", ".join(
                    f"[{p['id']}] {p.get('scroll_y')}/{p.get('scroll_height')}px"
                    for p in panes[:3])
                hint = (
                    "RECOVERY: ordinary scroll не двигает эту страницу — "
                    "она скроллится НЕ окном браузера, а внутренними панелями. "
                    "Прокрути конкретную панель: scroll(direction=\"down\", "
                    f"element_id=\"<id>\"). Панели: {plist}."
                )
            else:
                # Скролл не помогает → подсказываем кликнуть по якорной ссылке
                anchor = self._find_anchor_link(state)
                if anchor:
                    hint = (f"RECOVERY: скролл не даёт прогресса. На странице есть ссылка "
                            f"[{anchor['eid']}] «{anchor.get('text', '')}» → {anchor.get('href', '')}. "
                            f"Кликни по ней вместо скролла.")
                else:
                    hint = ("RECOVERY: скролл не даёт прогресса. Проанализируй элементы заново "
                            "и выбери конкретное действие (click/type), а не scroll.")
        elif stuck_action in ("type", "fill_form"):
            hint = ("RECOVERY: ввод не подтверждён. НЕ повторяй type в тот же element_id "
                    "и НЕ дописывай код поверх текущего содержимого. Сначала заново "
                    "проверь актуальный element_id и текущее значение поля; для редактора "
                    "кода используй полную замену clear=true. Если значение уже совпадает, "
                    "переходи к кнопке отправки.")
        elif stuck_action in ("click", "click_at", "hover", "drag"):
            hint = ("RECOVERY: игровое действие не дало прогресса. Не повторяй его вслепую: "
                    "сделай новый screenshot/observe, проверь текущий номер уровня и текст задания, "
                    "затем выбери другой элемент или координаты. Для canvas используй click_at/drag, "
                    "для смены состояния — реальные key действия.")
        else:
            hint = f"RECOVERY: действие {stuck_action} повторяется без результата. Смени подход."

        return hint

    def _find_anchor_link(self, state: BrowserState) -> Optional[Dict[str, Any]]:
        """Якорная ссылка (#section) или ссылка с текстом — для recovery после скроллов"""
        for el in state.elements:
            if el.get("tag") == "a" and el.get("href"):
                href = el["href"]
                if "#" in href and not href.endswith("#"):
                    return el
        return None

    # ─── Блокирующие модальные окна (анти-пинг-понг) ────────────────
    # Типичный сбой: обязательная модалка («Уточните данные»), которую агент
    # закрывает крестиком/«Сбросить». Окно тут же открывается снова → шаги
    # уходят в пустую. Рантайм сам распознаёт окно и не даёт его отменять.

    _PROCEED_RE = re.compile(
        r"сохран|продолж|подтвер|примен|отправ|далее|дальше|готово|"
        r"заверш|выбрать|принять|agree|accept|continue|save|confirm|submit|next|apply|done|select|"
        r"(?:^|\s)ок(?:\s|$)|(?:^|\s)okay(?:\s|$)",
        re.I,
    )
    _CANCEL_RE = re.compile(
        r"закрыть|отмен|сброс|не сейчас|позже|close|cancel|dismiss|skip|later|×|✕|✖",
        re.I,
    )

    def _dialog_sig(self, state: BrowserState) -> Optional[str]:
        """Сигнатура открытого блокирующего окна (title + начало текста)."""
        d = state.dialog or {}
        if not d.get("open") or not d.get("blocking"):
            return None
        title = (d.get("title") or "").strip()
        text = re.sub(r"\s+", " ", (d.get("text") or "").strip())[:120]
        return f"{title}|{text}"

    def _track_dialog(self, state: BrowserState) -> Optional[str]:
        """
        Отследить открытия блокирующего окна. Возвращает подсказку, если то же
        окно открылось повторно (агент его уже закрывал, не выполнив требование).
        """
        sig = self._dialog_sig(state)
        prev = self._dialog_prev_sig
        self._dialog_prev_sig = sig
        if sig is None:
            return None
        if sig == prev:
            return None  # окно открыто непрерывно — это то же наблюдение
        # Окно появилось заново (или появилось впервые)
        self._dialog_open_counts[sig] = self._dialog_open_counts.get(sig, 0) + 1
        opens = self._dialog_open_counts[sig]
        if opens < 2:
            return None
        d = state.dialog or {}
        btns = d.get("buttons") or []
        proceed = [b for b in btns if self._PROCEED_RE.search(b.get("text") or "")]
        target = (f"Нажми [{proceed[0]['eid']}] «{proceed[0]['text']}»."
                  if proceed else
                  "Заполни обязательное поле окна и нажми его кнопку подтверждения.")
        hint = (
            f"[DIALOG-LOOP] Окно «{(d.get('title') or d.get('text', ''))[:80]}» "
            f"открылось уже {opens}-й раз. Его НЕЛЬЗЯ закрыть крестиком, «Сбросить», "
            f"Esc или «Не сейчас» — оно открывается снова, и задача не двигается. "
            f"{target} Если для подтверждения нужно значение, которого у тебя нет — "
            f"используй wait_user, а не закрытие окна."
        )
        if sig not in self._dialog_hint_given:
            self._dialog_hint_given.add(sig)
            logger.info(hint[:300])
        return hint

    def _cancel_buttons_in_dialog(self, state: BrowserState) -> List[Dict[str, Any]]:
        """Кнопки отмены внутри открытого блокирующего окна (их клик запрещён)."""
        d = state.dialog or {}
        if not d.get("open") or not d.get("blocking"):
            return []
        out = []
        for el in state.elements:
            # Берём всё, что помечено как элемент диалога: dialog.buttons может
            # не содержать крестик без подписи или кнопки за лимитом среза.
            if not el.get("in_dialog"):
                continue
            tag = el.get("tag")
            is_btn = (tag == "button" or el.get("role") == "button"
                      or (tag == "input" and el.get("type") in ("submit", "button")))
            if not is_btn:
                continue
            label = f"{el.get('text') or ''} {el.get('aria') or ''}"
            if self._CANCEL_RE.search(label):
                out.append(el)
        return out

    def _dialog_guard(self, action: Dict[str, Any], state: BrowserState) -> Optional[Dict[str, Any]]:
        """
        Блокировка клика по «отмене» в обязательной модалке.
        Возвращает результат-ошибку с подсказкой либо None (действие разрешено).
        """
        tool = action.get("tool")
        if tool not in ("click", "click_at"):
            return None
        cancels = self._cancel_buttons_in_dialog(state)
        if not cancels:
            return None

        hit = None
        if tool == "click":
            eid = str(action.get("element_id", ""))
            hit = next((c for c in cancels if c.get("eid") == eid), None)
        else:
            x, y = action.get("x"), action.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                for c in cancels:
                    cx, cy = c.get("x"), c.get("y")
                    w = max(c.get("width") or 0, 44)
                    h = max(c.get("height") or 0, 44)
                    if cx is not None and cy is not None and abs(x - cx) <= w and abs(y - cy) <= h:
                        hit = c
                        break
        if hit is None:
            return None

        d = state.dialog or {}
        proceed = [b for b in (d.get("buttons") or [])
                   if self._PROCEED_RE.search(b.get("text") or "")]
        target = (f"Вместо этого нажми [{proceed[0]['eid']}] «{proceed[0]['text']}»."
                  if proceed else
                  "Вместо этого заполни обязательное поле окна и нажми его кнопку подтверждения.")
        label = (hit.get("text") or hit.get("aria") or hit["eid"]).strip()[:40]
        return {
            "ok": False,
            "error": (
                f"Клик по «{label}» запрещён: это отмена обязательного модального окна — "
                f"оно закроется и откроется снова, задача не выполнится. {target} "
                f"Если нужных данных нет — wait_user."
            ),
        }

    # ─── Наблюдение / выполнение / верификация ──────────────────────

    # ─── Ожидание данных от пользователя ────────────────────────────

    async def _wait_for_user_input(self, timeout: float = 300,
                                    state_before: Optional[BrowserState] = None) -> Optional[str]:
        """
        Пауза цикла до передачи данных пользователем (web_backend →
        provide_user_input) или таймаут.
        Во время паузы каждые 2с проверяем состояние страницы: если пользователь
        сам ввёл код в браузере (состояние изменилось) — выходим сразу.
        Возвращает введённое значение или None.
        """
        self._user_input_event = asyncio.Event()
        self._user_input_value = None
        try:
            wait_task = asyncio.create_task(self._user_input_event.wait())
            poll_task = asyncio.create_task(self._poll_state_during_wait(state_before))
            done, pending = await asyncio.wait(
                {wait_task, poll_task},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if wait_task in done:
                return self._user_input_value
            if poll_task in done and poll_task.result():
                # Пользователь вмешался в браузер — выходим без данных
                return None
            return None  # таймаут
        finally:
            self._user_input_event = None

    async def _poll_state_during_wait(self, state_before: Optional[BrowserState],
                                       interval: float = 2.0) -> bool:
        """
        Поллинг состояния страницы во время wait_user.
        Возвращает True, если состояние изменилось (пользователь вмешался).
        """
        if state_before is None:
            # Без точки сравнения просто ждём вечно (прервётся по timeout)
            await asyncio.Event().wait()
            return False
        key_before = self._state_key(state_before)
        while True:
            await asyncio.sleep(interval)
            try:
                state_now = await self._observe()
                if state_now is not None and self._state_key(state_now) != key_before:
                    await self._progress(
                        "[INFO] Обнаружено: пользователь сам изменил страницу во время ожидания "
                        "(ввёл код / прошёл авторизацию). Продолжаю с новым состоянием."
                    )
                    return True
            except Exception:
                pass  # страница перезагружается — продолжаем поллинг

    async def provide_user_input(self, value: str):
        """
        Передача данных от пользователя (из web_backend).
        Если агент ждёт (wait_user) — будим его. Если работает — данные
        попадут в контекст модели на следующем шаге (user_messages).
        """
        self._user_input_value = value
        self._user_messages.append(value)
        if self._user_input_event is not None:
            self._user_input_event.set()
            await self._progress(f"[INFO] Получены данные от пользователя")
        else:
            await self._progress(f"[INFO] Данные от пользователя приняты — агент учтёт на следующем шаге")

    def _drain_user_messages(self) -> str:
        """Забрать накопленные сообщения пользователя для промпта"""
        if not self._user_messages:
            return ""
        msgs = list(self._user_messages)
        self._user_messages.clear()
        return (
            "=== СООБЩЕНИЯ ОТ ПОЛЬЗОВАТЕЛЯ (учти в решении) ===\n"
            + "\n".join(f"[USER] {m}" for m in msgs)
            + "\n"
        )

    # ─── Память об отвергнутых данных ───────────────────────────────
    # Семантику ("это ошибка, данные не подходят") оценивает LLM — она видит
    # текст сообщения. Рантайм только запоминает её вердикт и напоминает.

    def remember_rejected_value(self, value: str, message: str):
        """LM сообщила, что сайт отверг значение — запомнить для будущих шагов"""
        if value:
            self._rejected_credentials[str(value).strip()] = message

    def _rejected_warning(self) -> str:
        """Предупреждение для промпта: какие данные уже отвергнуты сайтом"""
        if not self._rejected_credentials:
            return ""
        lines = ["=== ОТВЕРГНУТЫЕ САЙТОМ ДАННЫЕ (НЕ ИСПОЛЬЗОВАТЬ ПОВТОРНО) ==="]
        for value, msg in self._rejected_credentials.items():
            lines.append(f"[REJECTED] {value} — сайт ответил: {msg[:120]}")
        lines.append("Используй альтернативу (email↔телефон) или finish с reason.")
        return "\n".join(lines) + "\n"

    # ─── Память о плане (анти-зацикливание на первом пункте) ─────────
    # Модель без видимого прошлого плана каждый шаг сочиняет его заново и
    # снова берётся за пункт 1 («определить тип страницы») — на 20-м шаге
    # это выглядит как дежавю. Рантайм хранит последний план и считает
    # повторы: три одинаковых плана подряд = модель ходит по кругу.

    def _track_plan(self, plan: List[str]):
        """Запомнить план шага и посчитать серию повторов одного и того же."""
        if not plan:
            return
        sig = frozenset(p.strip().lower()[:60] for p in plan)
        self._plan_history.append(sig)
        self._plan_history = self._plan_history[-6:]
        if len(self._plan_history) >= 2 and self._plan_history[-1] == self._plan_history[-2]:
            self._plan_repeat_streak += 1
        else:
            self._plan_repeat_streak = 1
        self._current_plan = list(plan)

    def _plan_cycle_warning(self) -> str:
        """Предупреждение в промпт, если модель повторяет один и тот же план."""
        if self._plan_repeat_streak < 3:
            return ""
        return (
            "\n=== [CYCLE] ТЫ ПОВТОРЯЕШЬ ОДИН И ТОТ ЖЕ ПЛАН "
            f"{self._plan_repeat_streak} ШАГОВ ПОДРЯД ===\n"
            "Ты зациклился: делаешь действие и пересочиняешь тот же план с нуля.\n"
            "1. Прочитай ИСТОРИЮ — какие пункты плана уже фактически выполнены.\n"
            "2. Верни план БЕЗ выполненных пунктов и сделай следующий невыполненный.\n"
            "3. Если следующий пункт уже выполнялся и страница не изменилась —\n"
            "   меняй подход (другой элемент, scroll, evaluate, другой путь),\n"
            "   а не повторяй то же действие.\n"
            "4. Если делать больше нечего — finish с итоговым отчётом.\n"
        )

    async def _collect_downloads(self) -> List[Dict[str, str]]:
        """Забрать файлы, скачанные браузером после последнего действия.

        Событие download приходит чуть позже возврата клика, а сам файл
        пишется асинхронно. Ждём долго ТОЛЬКО если скачивание началось:
        иначе каждый обычный клик замедлялся бы на полную паузу опроса."""
        pop = getattr(self.controller, "pop_downloads", None)
        if not callable(pop):
            return []
        files = pop()
        if files:
            return files
        await asyncio.sleep(0.3)
        files = pop()
        if files:
            return files
        in_progress = getattr(self.controller, "download_in_progress", None)
        if not (callable(in_progress) and in_progress()):
            return []
        for _ in range(40):  # крупный архив может грузиться долго
            await asyncio.sleep(0.5)
            files = pop()
            if files:
                return files
        return []

    async def _observe(self, wait_stable: bool = False) -> Optional[BrowserState]:
        """
        Захват BrowserState.
        wait_stable=True — сначала дождаться завершения навигации
        (после click/navigate контекст JS уничтожается и пересоздаётся).
        """
        page = self.controller.page
        if not page:
            return None

        if wait_stable:
            # Таймаут ожидания стабилизации — не блокируем цикл навсегда
            # (CDP-канал может зависнуть; наблюдали на neal.fun)
            import asyncio as _aio
            try:
                await _aio.wait_for(
                    self.controller.wait_for_stable_page(timeout=10), timeout=14
                )
            except _aio.TimeoutError:
                logger.warning("wait_for_stable_page общий таймаут — продолжаю без стабильности")

        # Несколько попыток: контекст может пересоздаваться.
        # Каждый capture ограничен по времени — зависший evaluate не убьёт шаг.
        import asyncio as _aio
        for attempt in range(3):
            try:
                state = await _aio.wait_for(capture_browser_state(page), timeout=15)
                await self._attach_tabs(state)
                return state
            except _aio.TimeoutError:
                logger.warning(f"Observe attempt {attempt + 1}: capture завис (>15с)")
                await asyncio.sleep(1.0 + attempt)
            except Exception as e:
                logger.warning(f"Observe attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1.0 + attempt)

        logger.error("Observe failed after 3 attempts")
        return None

    async def _revive_chrome(self) -> Optional[BrowserState]:
        """Chrome умер (TAL/краш/закрыт) — перезапускаем с CDP и переподключаемся.
        Возвращает свежий BrowserState или None, если оживить не удалось.
        Проверяет не только открытый сокет, но и реальный HTTP-ответ CDP:
        за мгновение до смерти порт успевает принять соединение (гонка 00:30)."""
        import socket
        import asyncio as _aio
        import urllib.request

        def _cdp_responsive() -> bool:
            # Сокет-проверка ложно-положительна при смерти Chrome — гоняем HTTP
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.controller.cdp_port}/json/version", timeout=3
                ) as resp:
                    return resp.status == 200
            except Exception:
                return False

        def _chrome_process_alive() -> bool:
            try:
                probe = subprocess.run(
                    ["pgrep", "-x", "Google Chrome"], capture_output=True, timeout=5
                )
                return probe.returncode == 0
            except Exception:
                return False

        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if not Path(chrome_path).exists():
            import shutil
            chrome_path = shutil.which("chrome") or shutil.which("google-chrome") or chrome_path

        profile_dir = Path(__file__).parent.parent / "ai_browser_profile"
        profile_dir.mkdir(exist_ok=True)

        if _cdp_responsive():
            # Порт отвечает HTTP 200 — Chrome жив, revive не нужен
            return None

        if _chrome_process_alive():
            # Классическая ловушка macOS: Chrome жив, но без CDP-флагов
            # (запущен вручную/LaunchServices). Одиночный процесс — второй
            # запуск просто откроет окно, порт не поднимется. Перезапускаем.
            await self._progress("[WARN] Chrome жив, но без CDP — перезапускаю с отладкой")
            logger.warning("Chrome alive without CDP — quitting via osascript")
            try:
                subprocess.run(
                    ["osascript", "-e", 'tell application "Google Chrome" to quit'],
                    capture_output=True, timeout=10,
                )
            except Exception as e:
                logger.warning(f"osascript quit failed: {e}")
            # Ждём фактического завершения процесса
            for _ in range(12):
                if not _chrome_process_alive():
                    break
                await _aio.sleep(1)
            if _chrome_process_alive():
                logger.error("Chrome did not quit in time — cannot relaunch")
                await self._progress("[ERROR] Chrome не завершился — перезапуск невозможен")
                return None

        await self._progress("[WARN] Chrome не отвечает — перезапускаю браузер")
        logger.warning("Chrome CDP port closed — relaunching browser")
        try:
            await self.controller.disconnect()
        except Exception:
            pass

        try:
            subprocess.Popen(
                [chrome_path,
                 f"--user-data-dir={profile_dir}",
                 f"--remote-debugging-port={self.controller.cdp_port}",
                 "--no-first-run",
                 "--no-default-browser-check"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(f"Failed to relaunch Chrome: {e}")
            return None

        # Ждём не просто открытого порта, а работающего CDP (HTTP 200)
        for _ in range(30):
            if _cdp_responsive():
                break
            await _aio.sleep(1)
        else:
            logger.error("Chrome CDP not ready after relaunch")
            return None

        try:
            await self.controller.connect(mode="attach")
            await self.controller.navigate("https://neal.fun/not-a-robot/")
            await _aio.sleep(2)
            return await self._observe()
        except Exception as e:
            logger.error(f"Revive reconnect failed: {e}")
            return None

    def _format_tabs(self) -> str:
        tabs = self.controller._tabs
        if not tabs:
            return "(нет вкладок)"
        return "\n".join(
            f"{'→ ' if t.is_active else '  '}[{t.tab_id}] {t.title[:60]} — {t.url[:80]}"
            for t in tabs[:10]
        )

    async def _execute(self, action: Dict[str, Any], state: BrowserState) -> Dict[str, Any]:
        tool = action.get("tool")
        try:
            if tool == "navigate":
                nav_result = await self.controller.navigate(action["url"])
                if not nav_result.get("ok"):
                    return nav_result  # {"ok": False, "error": "ERR_NAME_NOT_RESOLVED: ..."}
                return {"ok": True, "detail": f"Переход: {action['url']}"}

            elif tool == "click":
                eid, err = self._resolve_eid(
                    state, action.get("element_id", ""),
                    text_fallback=bool(action.get("text_fallback")),
                )
                if err:
                    return {"ok": False, "error": err}
                # Кладём актуальный id обратно: verify_typed_value и память
                # дальше работают с action, а не с локальной переменной.
                action["element_id"] = eid
                result = await self.controller.click_element(eid)
                return result

            elif tool in ("click_at", "hover", "drag"):
                # Мышинные действия с жёстким таймаутом: зависший mouse-вызов
                # навсегда блокирует шаг (наблюдали на игровых canvas).
                import asyncio as _aio

                async def _mouse():
                    if tool == "click_at":
                        return await self.controller.click_at(action.get("x", 0), action.get("y", 0))
                    if tool == "hover":
                        return await self.controller.hover_at(action.get("x", 0), action.get("y", 0))
                    return await self.controller.drag_at(
                        action.get("x1", 0), action.get("y1", 0),
                        action.get("x2", 0), action.get("y2", 0), action.get("steps", 12)
                    )

                try:
                    return await _aio.wait_for(_mouse(), timeout=20)
                except _aio.TimeoutError:
                    return {"ok": False, "error": f"{tool} превысил 20с — страница не отвечает на мышь"}

            elif tool == "type":
                # Canvas-терминалы (HTB Pwnbox, noVNC): DOM-поля нет, печать
                # идёт в текущий фокус страницы; координаты ставят фокус кликом.
                if not str(action.get("element_id") or "").strip():
                    cx = action.get("click_x")
                    cy = action.get("click_y")
                    if cx is None:
                        cx = action.get("x")
                    if cy is None:
                        cy = action.get("y")
                    return await self.controller.type_text(
                        action.get("text", ""),
                        click_x=(float(cx) if cx is not None else None),
                        click_y=(float(cy) if cy is not None else None),
                        press_enter=bool(action.get("submit")),
                    )
                eid, err = self._resolve_eid(
                    state, action.get("element_id", ""),
                    text_fallback=bool(action.get("text_fallback")),
                )
                if err:
                    return {"ok": False, "error": err}
                action["element_id"] = eid  # актуальный id для verify/памяти
                result = await self.controller.type_into_element(
                    eid, action.get("text", ""), clear=action.get("clear", True)
                )
                if result.get("ok") and action.get("submit"):
                    await self.controller.press_key("Enter")
                    result["submitted"] = True
                return result

            elif tool == "fill_form":
                return await self._execute_fill_form(action, state)

            elif tool == "key":
                await self.controller.press_key(action.get("key", "Enter"))
                return {"ok": True, "detail": f"Клавиша: {action.get('key')}"}

            elif tool == "scroll":
                res = await self.controller.scroll(
                    action.get("direction", "down"), action.get("amount", 600),
                    element_id=str(action.get("element_id") or ""),
                )
                pane = str(action.get("element_id") or "")
                if not res.get("ok"):
                    return res
                out = {"ok": True,
                       "detail": (f"Прокручена панель {pane}: {res.get('scroll_y')}px "
                                  f"из {res.get('scroll_height')}px"
                                  if pane else f"Прокрутка {action.get('direction')}")}
                if "pane_moved" in res:
                    out["pane_moved"] = res["pane_moved"]
                return out

            elif tool == "select":
                return await self.controller.select_option(
                    action.get("element_id", ""), action.get("value", "")
                )

            elif tool == "new_tab":
                tab = await self.controller.new_tab(action.get("url") or None)
                return {"ok": True, "detail": f"Новая вкладка [{tab.tab_id}]: {tab.url}"}

            elif tool == "switch_tab":
                await self.controller.switch_tab(action.get("tab_id", ""))
                return {"ok": True, "detail": f"Вкладка: {action.get('tab_id')}"}

            elif tool == "close_tab":
                await self.controller.close_tab(action.get("tab_id", ""))
                return {"ok": True, "detail": f"Закрыта: {action.get('tab_id')}"}

            elif tool == "extract":
                text = await self.controller.extract_text(action.get("selector", "body"),
                                                          limit=20000)
                return {"ok": True, "detail": f"Извлечено {len(text)} символов",
                        "extracted_text": text}

            elif tool == "wait":
                await asyncio.sleep(action.get("seconds", 2))
                return {"ok": True, "detail": f"Ожидание {action.get('seconds')}с"}

            elif tool == "wait_user":
                # Ожидание данных от пользователя (код из SMS, push, пароль).
                # Агент не может их получить сам — пауза до передачи через UI.
                # Во время паузы поллим состояние страницы: если пользователь
                # сам ввёл код в браузере — выходим из ожидания сразу.
                self.state = AgentState.WAITING_USER
                reason = action.get("reason") or "нужны данные от пользователя"
                await self._progress(
                    f"[WAIT] ОЖИДАНИЕ ДАННЫХ ОТ ПОЛЬЗОВАТЕЛЯ: {reason}\n"
                    f"   Передай данные через поле ввода (кнопка «Передать агенту»), "
                    f"или введи их в браузер сам — агент продолжит после этого."
                )
                got = await self._wait_for_user_input(
                    timeout=action.get("timeout", 300),
                    state_before=state,
                )
                if got:
                    return {"ok": True, "detail": f"Получены данные от пользователя ({len(got)} симв.)",
                            "user_input": got}
                # Данных не передано. Таймаут НЕ ошибка: пользователь мог сам
                # завершить вход. Goal check по новому состоянию решает —
                # цель достигнута (finish) или продолжаем цикл.
                state_now = await self._observe()
                if state_now is not None:
                    goal_reason = self.goal_detector.check_state(task, state_now)
                    if goal_reason:
                        return {"ok": True, "detail": f"Цель достигнута во время ожидания: {goal_reason}",
                                "goal_completed": True, "goal_reason": goal_reason}
                    if self._state_key(state_now) != self._state_key(state):
                        return {"ok": True, "detail": "Пользователь сам изменил страницу во время ожидания"}
                return {"ok": False, "error": "Пользователь не передал данные за отведённое время"}

            elif tool == "evaluate":
                result = await self.controller.evaluate_js(action.get("expression", ""))
                return {"ok": True, "detail": "JS выполнен", "result": str(result)[:500]}

            elif tool == "sandbox":
                # Изолированный каталог: команды и скрипты выполняются внутри
                # песочницы, файлы остаются там же — наружу writes запрещены.
                if self.sandbox is None or not self.sandbox.enabled:
                    return {"ok": False,
                            "error": "Песочница недоступна/отключена — используй terminal"}
                mode = action.get("mode") or "run"
                tmo = action.get("timeout") or None
                if mode == "files":
                    return self.sandbox.list_files(action.get("path") or "")
                if mode == "read":
                    return self.sandbox.read_file(action.get("path") or action.get("filename") or "")
                if mode == "write":
                    return self.sandbox.write_file(
                        action.get("path") or action.get("filename") or "",
                        action.get("content") or action.get("script") or "")
                if mode == "delete":
                    return self.sandbox.delete(action.get("path") or action.get("filename") or "")
                if mode == "info":
                    info = self.sandbox.info()
                    return {"ok": True, "detail": f"Песочница: {info['root']} "
                                                  f"(изоляция: {info['isolation']})"}
                return await self.sandbox.run(
                    command=action.get("command", ""),
                    script=action.get("script", ""),
                    filename=action.get("filename", ""),
                    interpreter=action.get("interpreter", "python3"),
                    cwd=action.get("path") or "",
                    timeout=tmo,
                    isolate=action.get("isolate"),
                )

            elif tool == "terminal":
                # Работа вне браузера: shell-команда или задание Koda-агенту.
                # Вывод возвращается модели целиком (он уже сжат в executor) —
                # это ее данные для следующих шагов (число, токен, путь файла).
                # Строку команды и вывод в UI печатает сам executor ([TERM] $…):
                # дубль отсюда раньше давал две одинаковые строки команды и
                # ни одной строки ответа в терминальной панели.
                if self.terminal is None or not self.terminal.enabled:
                    return {"ok": False,
                            "error": "Терминал недоступен/отключён — работай только браузером"}
                return await self.terminal.run(
                    command=action.get("command", ""),
                    prompt=action.get("prompt", ""),
                    cwd=action.get("cwd") or None,
                    timeout=action.get("timeout") or None,
                    via_koda=action.get("via_koda"),
                )

            else:
                return {"ok": False, "error": f"Неизвестный tool: {tool}"}

        except Exception as e:
            logger.error(f"Execute {tool} error: {e}")
            return {"ok": False, "error": str(e)}

    async def _execute_fill_form(self, action: Dict[str, Any], state: BrowserState) -> Dict[str, Any]:
        """Композитное действие: заполнить поля формы и отправить"""
        fields: Dict[str, str] = action.get("fields", {})
        form_id = action.get("form_id", "")
        filled, missed = [], []

        for field_name, value in fields.items():
            eid = self._find_field_element(state, field_name, form_id)
            if eid:
                r = await self.controller.type_into_element(eid, str(value), clear=True)
                if r.get("ok"):
                    filled.append(field_name)
                else:
                    missed.append(field_name)
            else:
                missed.append(field_name)

        if missed:
            return {"ok": len(filled) > 0, "detail": f"Заполнено: {filled}, не найдено: {missed}"}

        if action.get("submit", True):
            submit_eid = self._find_submit_button(state, form_id)
            if submit_eid:
                await asyncio.sleep(0.3)
                r = await self.controller.click_element(submit_eid)
                if r.get("ok"):
                    return {"ok": True, "detail": f"Форма заполнена ({', '.join(filled)}) и отправлена", "submitted": True}
            if filled:
                await self.controller.press_key("Enter")
                return {"ok": True, "detail": f"Форма заполнена ({', '.join(filled)}), отправлена через Enter", "submitted": True}

        return {"ok": True, "detail": f"Заполнено: {', '.join(filled)} (без отправки)"}

    def _find_field_element(self, state: BrowserState, field_name: str, form_id: str = "") -> Optional[str]:
        name_lower = field_name.lower()
        candidates = []
        for el in state.elements:
            if el.get("tag") not in ("input", "textarea", "select"):
                continue
            if form_id and el.get("form") and el["form"] != form_id:
                continue
            score = 0
            if (el.get("name") or "").lower() == name_lower:
                score = 4
            elif (el.get("id_attr") or "").lower() == name_lower:
                score = 3
            elif name_lower in (el.get("name") or "").lower():
                score = 2
            elif name_lower in (el.get("placeholder") or "").lower():
                score = 1
            elif name_lower in (el.get("aria") or "").lower():
                score = 1
            if score:
                candidates.append((score, el["eid"]))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return None

    def _find_submit_button(self, state: BrowserState, form_id: str = "") -> Optional[str]:
        submit_keywords = ("submit", "send", "отправить", "отправка", "sign in", "log in",
                           "войти", "register", "next", "далее", "ok", "поиск")
        best = None
        for el in state.elements:
            if el.get("tag") not in ("button", "input"):
                continue
            if el.get("type") == "submit":
                if not form_id or not el.get("form") or el["form"] == form_id:
                    return el["eid"]
            text = (el.get("text") or "").lower()
            if any(kw in text for kw in submit_keywords):
                if not form_id or not el.get("form") or el["form"] == form_id:
                    best = best or el["eid"]
        return best

    def _find_element_by_text(self, state: BrowserState, text: str) -> Optional[Dict[str, Any]]:
        text_lower = text.lower().strip()
        for el in state.elements:
            el_text = (el.get("text") or "").lower()
            el_aria = (el.get("aria") or "").lower()
            if text_lower and (text_lower in el_text or text_lower in el_aria):
                return el
        return None

    _EID_RE = re.compile(r"^e\d+$", re.I)
    # Заглушка вместо id: 'e?', 'e', 'e__', 'e*' — модель не подобрала номер.
    _EID_PLACEHOLDER_RE = re.compile(r"^e[\W_]*$", re.I)

    def _resolve_eid(self, state: BrowserState, raw: Any,
                     text_fallback: bool = False) -> tuple:
        """
        Элемент из ответа LLM → реальный eid. Возвращает (eid, error).
        Модель иногда шлёт плейсхолдер ('e?') или подпись кнопки вместо id.
        Различаем: похож на id → его нет в снимке; иначе ищем по тексту.
        """
        raw = str(raw or "").strip()
        if not raw:
            return None, "Не указан element_id — возьми id из списка элементов состояния."
        if state.find_element(raw) is not None:
            return raw, None
        if self._EID_PLACEHOLDER_RE.match(raw):
            return None, (f"element_id '{raw}' — незаполненная заглушка. "
                          f"Возьми точный id (e1, e2, ...) из списка элементов состояния.")
        cleaned = raw.strip("?*# ").strip()
        if self._EID_RE.match(cleaned) and not text_fallback:
            return None, (f"Элемента с id '{raw}' нет в текущем состоянии. "
                          f"Id пересоздаются после каждого действия — возьми актуальный "
                          f"из списка элементов.")
        found = self._find_element_by_text(state, cleaned)
        if found:
            return found["eid"], None
        return None, (f"Элемент '{raw}' не найден (ни по id, ни по тексту). "
                      f"Используй точный id из списка элементов.")

    async def _verify(self, action, result, state_before, state_after) -> str:
        tool = action.get("tool")
        if not result.get("ok"):
            return f"[ERROR] {result.get('error', 'действие не выполнено')}"
        if tool == "navigate":
            if state_after and state_after.url and not state_after.url.startswith("about:"):
                return f" Страница загружена: {state_after.title[:60]}"
            return " Навигация выполнена, страница пустая"
        if tool == "click":
            if state_after is None:
                return " Клик выполнен (страница в навигации)"
            if state_after.url != state_before.url:
                return f" Клик привёл к переходу: {state_after.url[:80]}"
            return " Клик выполнен"
        if tool == "type":
            return " Текст введён" + (" + Enter" if result.get("submitted") else "")
        return " Выполнено"

    def _format_result(self, tool: str, result: Dict[str, Any], verified: str,
                       verdict: Optional[Dict[str, Any]] = None,
                       diff: Optional[Dict[str, Any]] = None) -> str:
        # Ошибка действия (DNS, элемент не найден) — важнее всего
        if not result.get("ok"):
            err = result.get("error", "действие не выполнено")
            # Для терминала вывод и есть причина: без stderr ("URL using
            # bad/illegal format") модель видит только код возврата и
            # повторяет ту же команду.
            if tool in ("terminal", "sandbox"):
                out = (result.get("output") or result.get("detail")
                       or result.get("output_partial") or "")
                if out:
                    return f"[ERROR] {err}\n{out[:2500]}"
            return f"[ERROR] {err}"
        text = verified
        detail = result.get("detail") or ""
        if tool == "extract" and result.get("extracted_text"):
            # Текст страницы и есть данные задачи: без него модель получает
            # отчёт «извлечено N символов» и решает по памяти.
            return f"{verified}\n{result['extracted_text'][:12000]}"
        if tool in ("terminal", "sandbox"):
            # Вывод команды = данные задачи (числа, токены, пути): отдаём detail
            # целиком (уже сжат в executor), verdict/diff про страницу не нужны.
            return f"{verified}\n{detail[:2500]}" if detail else verified
        if detail and detail not in verified:
            text += f" ({detail[:100]})"
        # Вердикт о прогрессе: LLM должен знать, сработало ли действие реально
        if verdict and not verdict["progress"]:
            text += f" |  NO_PROGRESS: {verdict['message']}"
        elif verdict and diff and diff.get("changed"):
            text += f" |  {diff.get('summary', '')[:150]}"
        return text

    async def _progress(self, text: str):
        logger.info(text)
        try:
            await self.on_progress(text)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")
