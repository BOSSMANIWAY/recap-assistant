"""
State Diff - сравнение BrowserState до и после действия

"Успешный click" ≠ "состояние изменилось". Для SPA и модалок URL может
не меняться, поэтому diff смотрит на:
- URL, title
- видимые интерактивные элементы (появились/исчезли)
- формы, inputs
- dialog/modal
- текст страницы

Результат используется для progress_check: действие без изменений состояния
= no_progress, и LLM получает это явно.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def diff_states(before, after) -> Dict[str, Any]:
    """
    Сравнить два BrowserState. Возвращает:
    {
        "changed": bool,
        "url_changed": bool,
        "title_changed": bool,
        "elements_appeared": [...],   # описания новых интерактивных элементов
        "elements_disappeared": [...],
        "inputs_delta": int,          # изменение числа input/textarea/select
        "forms_delta": int,
        "dialog_opened": bool,
        "dialog_closed": bool,
        "text_changed": bool,
        "summary": "человекочитаемое описание для LLM"
    }
    """
    if after is None:
        return {"changed": False, "summary": "страница не читается после действия"}

    if before is None:
        return {"changed": True, "summary": "первое наблюдение"}

    result: Dict[str, Any] = {
        "changed": False,
        "url_changed": False,
        "title_changed": False,
        "elements_appeared": [],
        "elements_disappeared": [],
        "inputs_delta": 0,
        "forms_delta": 0,
        "dialog_opened": False,
        "dialog_closed": False,
        "text_changed": False,
        "tabs_changed": False,      # изменился набор вкладок
        "tabs_appeared": [],        # [{tab_id, url, title}] — новые вкладки
        "active_tab_changed": "",   # было → стало
        "summary": "",
    }

    # URL
    url_b = (before.url or "").split("?")[0].rstrip("/")
    url_a = (after.url or "").split("?")[0].rstrip("/")
    if url_b != url_a:
        result["url_changed"] = True

    # Title
    if (before.title or "") != (after.title or ""):
        result["title_changed"] = True

    # Интерактивные элементы: сравниваем по (tag, name/id, текст)
    def el_key(el: Dict[str, Any]) -> str:
        return "|".join([
            el.get("tag", ""),
            (el.get("name") or el.get("id_attr") or ""),
            (el.get("text") or el.get("aria") or "")[:40],
        ])

    interactive_b = {el_key(el) for el in before.elements
                     if el.get("tag") in ("input", "textarea", "select", "button", "a")}
    interactive_a = {el_key(el) for el in after.elements
                     if el.get("tag") in ("input", "textarea", "select", "button", "a")}

    appeared = interactive_a - interactive_b
    disappeared = interactive_b - interactive_a
    result["elements_appeared"] = sorted(appeared)[:10]
    result["elements_disappeared"] = sorted(disappeared)[:10]

    # Счётчики полей
    def count_inputs(state) -> int:
        return sum(1 for el in state.elements if el.get("tag") in ("input", "textarea", "select"))

    def count_forms(state) -> int:
        return len({el.get("form") for el in state.elements if el.get("form")})

    result["inputs_delta"] = count_inputs(after) - count_inputs(before)
    result["forms_delta"] = count_forms(after) - count_forms(before)

    # Dialog (из STATE_JS: включает modal/popup классы)
    dialog_b = bool(getattr(before, "dialog", {}).get("open"))
    dialog_a = bool(getattr(after, "dialog", {}).get("open"))
    result["dialog_opened"] = dialog_a and not dialog_b
    result["dialog_closed"] = dialog_b and not dialog_a
    if result["dialog_opened"]:
        result["dialog_text"] = (getattr(after, "dialog", {}) or {}).get("text", "")[:200]

    # Текст и значения полей. Для редакторов кода page_text часто не меняется,
    # поэтому сравниваем value/текст каждого input и textarea отдельно.
    text_b = (before.page_text or "")[:2000]
    text_a = (after.page_text or "")[:2000]
    result["text_changed"] = text_b != text_a

    def field_values(state) -> Dict[str, str]:
        values = {}
        for el in state.elements:
            if el.get("tag") in ("input", "textarea"):
                values[el.get("eid", "")] = str(el.get("value") or "")
        return values

    fields_b = field_values(before)
    fields_a = field_values(after)
    result["field_values_changed"] = fields_b != fields_a
    result["fields_changed"] = [eid for eid in set(fields_b) | set(fields_a)
                                 if fields_b.get(eid, "") != fields_a.get(eid, "")][:10]

    # Выбор клеток (игровые сетки): изменился ли набор [ВЫБРАНО] элементов.
    # Клик по клетке CAPTCHA меняет только подсветку — это прогресс.
    def selected_keys(state) -> set:
        return {el.get("eid", "") for el in state.elements if el.get("selected")}
    sel_b, sel_a = selected_keys(before), selected_keys(after)
    result["selection_changed"] = sel_b != sel_a
    result["selected_count"] = len(sel_a)

    # Прокрутка внутренних панелей (split-screen TryHackMe): окно браузера
    # не двигается вообще, единственный признак скролла — scrollTop панели.
    def pane_pos(state) -> Dict[str, int]:
        return {p.get("id", ""): int(p.get("scroll_y") or 0)
                for p in (getattr(state, "scroll_panes", []) or [])}
    panes_b, panes_a = pane_pos(before), pane_pos(after)
    result["pane_scrolled"] = panes_b != panes_a
    if result["pane_scrolled"]:
        moved = [f"{pid} {panes_b.get(pid, 0)}→{pos}px"
                 for pid, pos in panes_a.items() if panes_b.get(pid, 0) != pos]
        result["pane_moves"] = moved[:3]

    # Заметные сообщения: появились новые или исчезли старые (по тексту)
    def notice_texts(state) -> set:
        return {n.get("text", "") for n in (getattr(state, "notices", []) or []) if n.get("text")}
    notices_b = notice_texts(before)
    notices_a = notice_texts(after)
    result["notices_appeared"] = sorted(notices_a - notices_b)[:3]
    result["notices_disappeared"] = bool(notices_b - notices_a)

    # Вкладки. Ключевой сигнал: клик открыл лабу в НОВОЙ вкладке — на
    # активной странице при этом не меняется НИЧЕГО, и без этого сравнения
    # шаг выглядит как NO_PROGRESS (агент жмёт кнопку по кругу).
    def tab_sig(t: Dict[str, Any]) -> str:
        return f"{(t.get('url') or '').split('#')[0]}|{(t.get('title') or '')[:40]}"

    tabs_b = [t if isinstance(t, dict) else t.__dict__
              for t in (getattr(before, "tabs", []) or [])]
    tabs_a = [t if isinstance(t, dict) else t.__dict__
              for t in (getattr(after, "tabs", []) or [])]
    if tabs_b and tabs_a:  # нет данных о вкладках — не считаем изменением
        sig_b = {tab_sig(t) for t in tabs_b}
        sig_a = {tab_sig(t) for t in tabs_a}
        new_tabs = [t for t in tabs_a if tab_sig(t) not in sig_b]
        closed = [t for t in tabs_b if tab_sig(t) not in sig_a]
        if new_tabs or closed:
            result["tabs_changed"] = True
            result["tabs_appeared"] = new_tabs[:5]
            result["tabs_disappeared"] = len(closed)
        act_b = next((t.get("tab_id") for t in tabs_b if t.get("is_active")), "")
        act_a = next((t.get("tab_id") for t in tabs_a if t.get("is_active")), "")
        if act_b != act_a:
            result["active_tab_changed"] = f"{act_b} → {act_a}"

    # Итог
    changed = any([
        result["url_changed"], result["title_changed"],
        appeared, disappeared,
        result["inputs_delta"] != 0, result["forms_delta"] != 0,
        result["dialog_opened"], result["dialog_closed"],
        result["text_changed"], result["field_values_changed"],
        result["notices_appeared"], result["notices_disappeared"],
        result["selection_changed"], result["pane_scrolled"],
        result["tabs_changed"], bool(result["active_tab_changed"]),
    ])
    result["changed"] = changed

    # Человекочитаемая сводка
    parts = []
    if result["url_changed"]:
        parts.append(f"URL изменился → {after.url[:80]}")
    if result["title_changed"]:
        parts.append(f"заголовок: «{(after.title or '')[:50]}»")
    if result["dialog_opened"]:
        parts.append(f"открылось модальное окно: {(result.get('dialog_text') or '')[:100]}")
    if result.get("notices_appeared"):
        parts.append("появились выделенные сообщения: " + "; ".join(a[:80] for a in result["notices_appeared"]))
    if result.get("notices_disappeared"):
        parts.append("выделенные сообщения исчезли")
    if result["dialog_closed"]:
        parts.append("модальное окно закрылось")
    if result["inputs_delta"] > 0:
        parts.append(f"появилось полей ввода: +{result['inputs_delta']}")
    if result["inputs_delta"] < 0:
        parts.append(f"исчезло полей ввода: {result['inputs_delta']}")
    if result["forms_delta"] != 0:
        parts.append(f"форм: {result['forms_delta']:+d}")
    if appeared:
        sample = [k.replace("|", " ")[:50] for k in list(appeared)[:3]]
        parts.append(f"новые элементы: {'; '.join(sample)}")
    if disappeared:
        sample = [k.replace("|", " ")[:50] for k in list(disappeared)[:3]]
        parts.append(f"исчезли элементы: {'; '.join(sample)}")
    if result.get("pane_scrolled"):
        parts.append("прокручена панель: " + ", ".join(result.get("pane_moves") or []))
    for t in (result.get("tabs_appeared") or []):
        parts.append(
            f"ОТКРЫЛАСЬ НОВАЯ ВКЛАДКА [{t.get('tab_id')}]: "
            f"{(t.get('url') or '')[:90]} «{(t.get('title') or '')[:50]}» — "
            f"переключись switch_tab(tab_id=\"{t.get('tab_id')}\")"
        )
    if result.get("tabs_disappeared"):
        parts.append(f"закрылось вкладок: {result['tabs_disappeared']}")
    if result.get("active_tab_changed"):
        parts.append(f"активная вкладка: {result['active_tab_changed']}")
    if not parts and result["text_changed"]:
        parts.append("изменился текст страницы")

    result["summary"] = "; ".join(parts) if parts else "состояние не изменилось"
    return result


def progress_verdict(diff: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Вердикт о прогрессе после действия.
    Возвращает {"progress": bool, "message": str} — message идёт в LLM
    как last_action_result на следующем шаге.
    """
    tool = action.get("tool", "")

    # Навигация: успех = URL изменился или страница загрузилась
    if tool == "navigate":
        if diff.get("url_changed") or diff.get("title_changed"):
            return {"progress": True, "message": "навигация выполнена"}
        return {"progress": False, "message": "навигация не изменила страницу"}

    # Скролл: прогресс = появились новые элементы или изменился текст
    if tool == "scroll":
        if diff.get("elements_appeared") or diff.get("text_changed"):
            return {"progress": True, "message": "прокрутка открыла новый контент"}
        return {"progress": False,
                "message": "прокрутка не открыла нового контента — достигнут край страницы или контент не подгрузился"}

    # Клик/ввод: прогресс = любое изменение состояния, включая value редактора.
    if diff.get("tabs_appeared"):
        t = (diff.get("tabs_appeared") or [{}])[0]
        return {"progress": True,
                "message": (f"действие сработало: открылась НОВАЯ ВКЛАДКА "
                            f"[{t.get('tab_id')}] {(t.get('url') or '')[:90]} "
                            f"«{(t.get('title') or '')[:50]}». Активная страница "
                            "не изменилась — это НОРМАЛЬНО (ссылка открылась в "
                            f"новой вкладке). Дальше: switch_tab(tab_id=\"{t.get('tab_id')}\"). "
                            "НЕ нажимай ту же кнопку/ссылку повторно.")}
    if diff.get("changed"):
        if diff.get("selection_changed"):
            return {"progress": True,
                    "message": f"выбор клеток изменился (сейчас выбрано: {diff.get('selected_count', 0)})"}
        return {"progress": True, "message": f"состояние изменилось: {diff.get('summary', '')}"}
    return {"progress": False,
            "message": f"действие {tool} выполнено, но состояние страницы НЕ изменилось — элемент не сработал, "
                       f"нужен другой подход (другой элемент, wait, navigate)"}


async def verify_typed_value(controller, element_id: str, expected: str) -> Dict[str, Any]:
    """
    Верификация ввода: значение поля реально установилось?
    Ввод текста не обязан менять состояние страницы — проверяем value поля.
    """
    actual = await controller.get_element_value(element_id)
    if actual is None:
        return {"ok": False, "message": f"поле {element_id} не найдено для проверки значения"}
    if actual.strip() == expected.strip():
        return {"ok": True, "message": f"значение установлено в {element_id}"}
    # Фактическое значение — данные: без него LLM вслепую перепечатывает то же
    # самое, хотя поле уже приняло другое (маска ввода, автоскоупинг скобок).
    return {"ok": False,
            "message": (f"в {element_id} сейчас «{actual[:200]}», ожидалось "
                        f"«{expected[:200]}» — поле переписало ввод либо оно в "
                        f"другом фрейме/только для чтения. Перепечатывание того "
                        f"же текста не поможет: сверься со списком элементов и, "
                        f"если значение уже верное, отправь форму")}
