"""
Goal Completion Detector - определение достижения цели на уровне runtime

LLM не должен сам решать "задача выполнена?" — runtime проверяет объективные
сигналы успеха и останавливает цикл автоматически:

1. Форма отправлена: URL получил query-параметры с данными формы
   (пример: ?name=Test+User&email=...&message=...)
2. На странице появилось сообщение об успехе (спасибо / thank you / sent...)
   — сравнивается с текстом ДО действия, чтобы не ловить ложные срабатывания
3. Навигация на целевой URL, если задача содержала конкретный адрес
"""

import logging
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

# Ключевые слова успеха (lowercase, ru + en)
SUCCESS_KEYWORDS = [
    "спасибо", "thank you", "thanks for", "message sent", "сообщение отправлено",
    "отправлено", "заявка принята", "заявка отправлена", "мы свяжемся", "свяжемся с вами",
    "form submitted", "successfully sent", "успешно отправлено", "your message has been sent",
    "request received", "запрос отправлен", "письмо отправлено",
    "order confirmed", "заказ оформлен", "заказ принят", "payment successful",
    "оплата прошла", "registration complete", "регистрация завершена",
    "account created", "аккаунт создан", "вы вошли", "logged in", "welcome back",
    "добро пожаловать", "subscription confirmed", "подписка оформлена",
]

# Имена полей, считающиеся данными формы в query-параметрах
FORM_PARAM_NAMES = {
    "name", "email", "message", "phone", "subject", "comment", "text",
    "firstname", "first_name", "lastname", "last_name", "company", "role",
    "username", "login", "query", "search", "q", "s",
}

# Слова, которые могут встречаться в задаче "просто открой сайт" (ru + en).
# Универсальный подход: убираем их из задачи и смотрим, что осталось.
NAVIGATION_FILLER_WORDS = {
    # ru
    "открой", "открыть", "откройте", "зайди", "зайти", "зайдите", "перейди",
    "перейти", "перейдите", "на", "в", "во", "сайт", "страницу", "страница",
    "пожалуйста", "давай", "давайте", "мне", "нужно", "надо", "хочу",
    "и", "а", "но", "же", "бы", "ли", "там", "туда", "тут", "здесь",
    "посмотри", "посмотреть", "проверь", "проверить", "что", "там",
    # en
    "open", "go", "to", "navigate", "visit", "browse", "please", "the", "a",
    "an", "site", "website", "page", "and", "check", "look", "at", "let's",
    "lets", "me", "i", "want", "need", "just", "up",
}


class GoalDetector:
    """Детектор достижения цели по объективным сигналам"""

    def check_state(self, task: str, state: Any) -> Optional[str]:
        """
        State-only проверка цели: БЕЗ действия и результата.
        Вызывается в начале каждого цикла ДО обращения к LLM —
        если цель уже достигнута (например, пользователь сам всё сделал
        во время wait_user), runtime завершает задачу без LLM.
        """
        if state is None:
            return None
        return self._check_target_url(task, state)

    def check(
        self,
        task: str,
        action: Dict[str, Any],
        result: Dict[str, Any],
        state_before: Any,   # BrowserState
        state_after: Any,    # BrowserState
    ) -> Optional[str]:
        """
        Возвращает reason если цель достигнута, иначе None.
        Вызывается после каждого действия.
        """
        if state_after is None:
            return None

        tool = action.get("tool", "")
        interactive_action = tool in ("click", "type", "fill_form", "key", "select")

        # ─── Сигнал 0: HH.ru принял решение практики ───
        # Экран результата содержит эти кнопки и сбрасывает редактор при
        # переходе к следующей тренировке. Для задачи пользователя это уже
        # объективное завершение — больше ничего не нажимаем.
        if interactive_action and result.get("ok") and self._check_hh_practice_result(state_before, state_after):
            return "Решение практики принято — задание завершено"

        # ─── Сигнал 1: форма отправлена через GET (параметры в URL) ───
        if interactive_action and state_before is not None and result.get("ok"):
            completion = self._check_form_submit_via_url(state_before, state_after)
            if completion:
                return completion

        # ─── Сигнал 2: сообщение об успехе появилось ПОСЛЕ действия ───
        if interactive_action and result.get("ok"):
            completion = self._check_success_message(state_before, state_after)
            if completion:
                return completion

        # ─── Сигнал 3: задача требовала конкретный URL и мы на нём ───
        if tool == "navigate":
            completion = self._check_target_url(task, state_after)
            if completion:
                return completion

        # ─── Сигнал 4: после wait_user сессия авторизована без агента ───
        # wait_user означает "жду учётные данные". Если во время ожидания
        # форма входа исчезла или URL сменился — пользователь завершил вход сам.
        # Это объективные сигналы состояния, LLM не нужен.
        if tool == "wait_user" and state_before is not None:
            completion = self._check_authenticated_during_wait(state_before, state_after)
            if completion:
                return completion

        return None

    def _check_hh_practice_result(self, before, after) -> bool:
        """Распознать экран результата hh.ru после отправки кода."""
        if after is None:
            return False
        text = (after.page_text or "").lower()
        elements = " ".join((el.get("text") or "").lower() for el in after.elements)
        markers = ("продолжить тренировку", "перейти к настоящим задачам")
        return any(marker in text or marker in elements for marker in markers)

    def _check_authenticated_during_wait(self, before, after) -> Optional[str]:
        """Во время wait_user форма логина исчезла / URL сменился"""
        login_before = bool((before.flags or {}).get("login_form"))
        login_after = bool((after.flags or {}).get("login_form"))
        url_before = before.url or ""
        url_after = after.url or ""

        if login_before and not login_after:
            logger.info("Goal detected: login form disappeared during wait_user")
            return "Форма входа исчезла во время ожидания — авторизация пройдена"
        if (not login_before and not login_after) and url_after and url_before \
                and url_after != url_before:
            logger.info(f"Goal detected: URL changed during wait_user: {url_before} → {url_after}")
            return "Страница сменилась во время ожидания (пользователь завершил вход сам)"
        return None

    def _check_form_submit_via_url(self, before, after) -> Optional[str]:
        """URL изменился и получил query-параметры с данными формы"""
        url_before = before.url or ""
        url_after = after.url or ""
        if url_after == url_before:
            return None

        try:
            query = parse_qs(urlparse(url_after).query)
        except Exception:
            return None

        if not query:
            return None

        form_params = {k for k in query if k.lower() in FORM_PARAM_NAMES}
        # На странице до действия была форма?
        had_form = any(
            el.get("tag") in ("input", "textarea", "select") or el.get("form")
            for el in before.elements
        )

        if form_params and had_form:
            params_str = ", ".join(f"{k}={query[k][0][:30]}" for k in sorted(form_params)[:4])
            logger.info(f"Goal detected: form submitted via URL params: {params_str}")
            return f"Форма отправлена — URL содержит данные формы ({params_str})"
        return None

    def _check_success_message(self, before, after) -> Optional[str]:
        """Сообщение об успехе появилось после действия (не было до)"""
        text_after = (after.page_text or "").lower()
        text_before = (before.page_text or "").lower() if before else ""

        for kw in SUCCESS_KEYWORDS:
            if kw in text_after and kw not in text_before:
                logger.info(f"Goal detected: success message '{kw}'")
                return f"На странице появилось сообщение об успехе: «{kw}»"
        return None

    def _check_target_url(self, task: str, state) -> Optional[str]:
        """
        Задача требовала открыть конкретный сайт, и мы на нём.

        Универсальный подход (работает на любых задачах и опечатках):
        убираем из задачи домены и "навигационные" слова. Если остаётся
        содержательный текст — в задаче есть действия помимо навигации
        (залогиниться, заполнить, купить...), авто-финиш НЕ срабатывает.
        """
        # Ищем домены в задаче
        domains = re.findall(r"(?:https?://)?([a-z0-9-]+\.[a-z0-9.-]+)", task.lower())
        if not domains:
            return None

        current = (state.url or "").lower()
        if not current.startswith("http"):
            return None

        matched_domain = None
        for domain in domains:
            domain = domain.strip(".")
            if "." in domain and len(domain) >= 4 and domain in current:
                matched_domain = domain
                break
        if not matched_domain:
            return None

        # Убираем домены и навигационный "наполнитель" — что осталось?
        remainder = task.lower()
        for d in domains:
            remainder = remainder.replace(d.strip("."), " ")
        words = [
            w.strip(".,!?;:()\"'«»-")
            for w in remainder.split()
        ]
        meaningful = [w for w in words if w and w not in NAVIGATION_FILLER_WORDS]

        if not meaningful:
            logger.info(f"Goal detected: navigation-only task, site {matched_domain} opened")
            return f"Открыт целевой сайт: {matched_domain}"

        # В задаче есть содержательные действия — не финишим
        logger.debug(f"Task has actions beyond navigation: {meaningful[:5]}")
        return None
