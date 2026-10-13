"""
Browser State v2 - Снимок состояния страницы со стабильными element_id

Каждый интерактивный элемент получает стабильный id (e1, e2, ...).
LLM ссылается на элементы по id: click(e17), type(e5, "text").
Селекторы больше не нужны — агент не зависит от хрупких CSS/XPath.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# JS: размечает интерактивные элементы data-eid и собирает компактный снимок.
# Параметр startCounter — продолжение нумерации eid для дочерних фреймов.
# ВАЖНО: сырая строка (r"""...""") — внутри JS-регулярки с \d, \s и т.п.
STATE_JS = r"""
(startCounter) => {
    // 1. Снимаем старую разметку
    document.querySelectorAll('[data-eid]').forEach(el => el.removeAttribute('data-eid'));
    document.querySelectorAll('[data-epane]').forEach(el => el.removeAttribute('data-epane'));

    // 2. Находим интерактивные элементы — ВКЛЮЧАЯ shadow DOM и same-origin iframe
    const selector = [
        'a[href]',
        'button',
        'input:not([type="hidden"])',
        'textarea',
        'select',
        '[role="button"]',
        '[role="link"]',
        '[role="tab"]',
        '[role="checkbox"]',
        '[role="radio"]',
        '[role="combobox"]',
        '[contenteditable="true"]',
        '[onclick]',
        '[tabindex]:not([tabindex="-1"])',
        'canvas',
        'svg [fill]:not([fill="none"]), svg [stroke]:not([stroke="none"])'
    ].join(', ');

    // Игровые страницы часто используют div-области с pointer-событиями
    // вместо семантических button/input. Берём только видимые кликабельные области.
    const pointerCandidates = [];
    try {
        for (const el of document.querySelectorAll('div, section, span, li')) {
            const style = window.getComputedStyle(el);
            if (style.cursor === 'pointer' || style.touchAction === 'none') {
                pointerCandidates.push(el);
            }
        }
    } catch (e) {}

    // Рекурсивный обход: document -> shadow roots
    // (кросс-origin iframe обрабатываются снаружи через Playwright frames)
    function deepQueryAll(sel, root, out) {
        try {
            out.push(...root.querySelectorAll(sel));
        } catch (e) {}
        // Shadow roots
        let children = [];
        try { children = root.querySelectorAll('*'); } catch (e) {}
        for (const el of children) {
            if (el.shadowRoot) deepQueryAll(sel, el.shadowRoot, out);
        }
        return out;
    }

    const candidates = deepQueryAll(selector, document, []);
    for (const el of pointerCandidates) {
        if (!candidates.includes(el)) candidates.push(el);
    }

    // Прокручиваемые контейнеры (split-screen панели TryHackMe, чаты, IDE).
    // У таких страниц window.scrollY не двигается вообще: скроллится панель.
    // Элемент ниже видимой области панели достижим её скроллом, поэтому
    // выбрасывать его нельзя — иначе агент слепнет к половине интерфейса
    // (так и было: кнопки «Start AttackBox»/«Start Lab Machine» пропадали).
    const panes = new Map();   // элемент панели -> её id (p1, p2, ...)
    function scrollableAncestor(el) {
        let n = el.parentElement;
        for (let depth = 0; n && n !== document.body && depth < 10; depth++, n = n.parentElement) {
            const s = window.getComputedStyle(n);
            const oy = s.overflowY;
            if ((oy === 'auto' || oy === 'scroll') && n.scrollHeight > n.clientHeight + 8) {
                if (!panes.has(n)) {
                    panes.set(n, 'p' + (panes.size + 1));
                    n.setAttribute('data-epane', panes.get(n));
                }
                return n;
            }
        }
        return null;
    }

    const interactive = [];
    let counter = startCounter;

    for (const el of candidates) {
        // Видимость: в DOM, имеет размеры, не display:none
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        const visible = rect.width > 0 && rect.height > 0
            && style.display !== 'none'
            && style.visibility !== 'hidden'
            && style.opacity !== '0';
        if (!visible) continue;

        // В viewport-окрестности (включая чуть за пределами для scroll)
        const inWindowBand = !(rect.bottom < -200 || rect.top > window.innerHeight + 200);
        const pane = scrollableAncestor(el);
        if (!inWindowBand) {
            // Оставляем элемент, только если его панель сама на экране:
            // значит он достижим прокруткой этой панели.
            const pr = pane ? pane.getBoundingClientRect() : null;
            const paneOnScreen = !!pr && pr.bottom > 0 && pr.top < window.innerHeight
                && pr.width > 50 && pr.height > 50;
            if (!paneOnScreen) continue;
        }

        const eid = 'e' + (counter++);
        el.setAttribute('data-eid', eid);

        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute('type') || '';
        let text = '';

        if (tag === 'input') {
            if (type === 'submit' || type === 'button') {
                text = el.value || '';
            } else {
                text = el.placeholder || el.getAttribute('aria-label') || el.name || '';
            }
        } else {
            text = (el.innerText || el.textContent || el.value || '').trim().substring(0, 120);
        }

        // Родительская форма (для контекста)
        const form = el.closest('form');
        const formId = form ? (form.id || form.getAttribute('name') || 'form') : null;

        // Картинка внутри клетки (игровые сетки): src + цветовой профиль.
        // Имя файла часто семантично (stop-sign-3.jpg), цвет — для визуальных CAPTCHA.
        function imgProfile(node) {
            let img = node.tagName === 'IMG' ? node : node.querySelector('img');
            let src = img ? img.src : '';
            if (!src) {
                const bg = getComputedStyle(node).backgroundImage || '';
                let m = bg.match(/url\("?([^")]+)"?\)/);
                if (!m && node.parentElement) {
                    const pbg = getComputedStyle(node.parentElement).backgroundImage || '';
                    m = pbg.match(/url\("?([^")]+)"?\)/);
                }
                if (m) src = m[1];
            }
            if (!src) return null;
            const base = src.split('/').pop().split('?')[0].substring(0, 60);
            const info = { img: base };
            try {
                if (img && img.complete && img.naturalWidth > 0) {
                    const c = document.createElement('canvas');
                    c.width = 24; c.height = 24;
                    const ctx = c.getContext('2d');
                    ctx.drawImage(img, 0, 0, 24, 24);
                    const d = ctx.getImageData(0, 0, 24, 24).data;
                    let red = 0, green = 0, blue = 0, yellow = 0, dark = 0, total = 0;
                    for (let i = 0; i < d.length; i += 4) {
                        const r = d[i], g = d[i+1], b = d[i+2];
                        total++;
                        if (r > 120 && r > g * 1.4 && r > b * 1.4) red++;
                        if (g > 100 && g > r * 1.25 && g > b * 1.25) green++;
                        if (b > 110 && b > r * 1.3 && b > g * 1.1) blue++;
                        if (r > 150 && g > 120 && b < 100) yellow++;
                        if (r < 60 && g < 60 && b < 60) dark++;
                    }
                    const pct = (n) => Math.round(n / total * 100);
                    info.colors = { red: pct(red), green: pct(green), blue: pct(blue), yellow: pct(yellow), dark: pct(dark) };
                }
            } catch (e) { /* CORS canvas taint — цвет недоступен */ }
            return info;
        }
        const ip = imgProfile(el);

        // Состояние выбора (игровые клетки): aria + классы
        const clsAll = ((el.className + ' ' + (el.parentElement ? el.parentElement.className : '')) + '').toLowerCase();
        const selected = el.getAttribute('aria-pressed') === 'true'
            || el.getAttribute('aria-checked') === 'true'
            || /(^|[\s_-])(selected|checked|chosen|active)([\s_-]|$)/.test(clsAll);

        // Внутри модального окна? (помогает отличить submit-кнопку формы
        // от похожих кнопок-опций на фоне)
        const inDialog = !!el.closest('dialog[open], [role="dialog"], [role="alertdialog"], [class*="modal" i], [class*="Modal"], [class*="popup" i]');

        interactive.push({
            eid: eid,
            tag: tag,
            type: type || undefined,
            text: text || undefined,
            x: Math.round(rect.left + rect.width / 2),
            y: Math.round(rect.top + rect.height / 2),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            cursor: style.cursor || undefined,
            img: ip ? ip.img : undefined,
            colors: ip && ip.colors ? ip.colors : undefined,
            selected: selected || undefined,
            href: tag === 'a' ? el.href.substring(0, 200) : undefined,
            name: el.name || undefined,
            id_attr: el.id || undefined,
            placeholder: el.placeholder || undefined,
            value: ((tag === 'input' || tag === 'textarea') && el.value && type !== 'password') ? el.value.substring(0, 4000) : undefined,
            required: el.required || undefined,
            disabled: el.disabled || undefined,
            checked: el.checked || undefined,
            form: formId || undefined,
            aria: el.getAttribute('aria-label') || undefined,
            role: el.getAttribute('role') || undefined,
            in_dialog: inDialog || undefined,
            pane: pane ? panes.get(pane) : undefined,
            in_viewport: rect.top >= 0 && rect.bottom <= window.innerHeight
                && (!pane || (rect.top >= pane.getBoundingClientRect().top
                              && rect.bottom <= pane.getBoundingClientRect().bottom))
        });
    }

    return {
        url: window.location.href,
        title: document.title,
        readyState: document.readyState,
        scrollY: Math.round(window.scrollY),
        pageHeight: document.documentElement.scrollHeight,
        viewportHeight: window.innerHeight,
        viewportWidth: window.innerWidth,
        // Прокручиваемые панели: на split-screen страницах (TryHackMe) окно
        // браузера не скроллится вообще — скроллится каждая панель отдельно.
        scroll_panes: [...panes.entries()].map(([el, pid]) => {
            const r = el.getBoundingClientRect();
            return {
                id: pid,
                x: Math.round(r.left + r.width / 2),
                y: Math.round(r.top + r.height / 2),
                width: Math.round(r.width),
                height: Math.round(r.height),
                scroll_y: Math.round(el.scrollTop),
                scroll_height: el.scrollHeight,
                client_height: el.clientHeight,
                on_screen: r.bottom > 0 && r.top < window.innerHeight && r.width > 50,
                at_bottom: el.scrollTop + el.clientHeight >= el.scrollHeight - 30,
                label: (el.innerText || '').trim().substring(0, 60)
            };
        }).filter(p => p.on_screen),
        elements: interactive,
        // Структура страницы: заголовки — помогают понять тип и назначение страницы
        headings: [...document.querySelectorAll('h1, h2, h3')]
            .map(h => (h.innerText || '').trim().substring(0, 100))
            .filter(t => t.length > 0)
            .slice(0, 10),
        // Компактный текст страницы для контекста.
        // 3000 симв. теряли середину длинных страниц: на TryHackMe блок
        // задания с кнопкой «Download Task Files» оказывался за обрезкой, и
        // агент решал по чужим walkthrough вместо файлов задачи.
        page_text: document.body ? document.body.innerText.substring(0, 12000) : '',
        // Inline <script> страницы: innerText их НЕ содержит, а во многих
        // лабах (PortSwigger «unpredictable URL» и подобные) путь к админке,
        // API-токен или скрытый URL лежат именно в JS. Берём только inline
        // (без src — внешний файл здесь не читается), режем шум и лимитим.
        scripts: (() => {
            const parts = [...document.querySelectorAll('script:not([src])')]
                .map(s => (s.textContent || '').trim())
                .filter(t => t.length > 3 && !/^\s*window\.__|dataLayer|google_tag/i.test(t))
                .map(t => t.substring(0, 1500));
            return parts.join('\n;\n').substring(0, 8000);
        })(),
        // Детект типовых препятствий
        game: (() => {
            const body = document.body ? document.body.innerText : '';
            const level = body.match(/Level\s+(\d+)\s*:\s*([^\n]+)/i);
            const canvases = [...document.querySelectorAll('canvas')].map((c, i) => {
                const r = c.getBoundingClientRect();
                return {index: i, x: Math.round(r.left), y: Math.round(r.top), width: Math.round(r.width), height: Math.round(r.height)};
            }).filter(c => c.width > 0 && c.height > 0);
            // Canvas-терминалы: HTB Pwnbox (guacamole), noVNC, ttyd, xterm.
            // Полей ввода в DOM у них нет — вывод читается только со скриншота.
            const hay = location.href + ' ' + document.title + ' ' + body;
            const termEl = document.querySelector(
                '.terminal, .xterm, #terminal, .guac-layer, .display, canvas[tabindex]');
            const termWord = /pwnbox|guacamole|guac-|noVNC|novnc|ttyd|wetty|xterm|\/terminal|kasmvnc/i.test(hay);
            // Одно лишь слово «Pwnbox» в тексте страницы (карточка модуля, меню) —
            // ещё не терминал: иначе агент ждёт вывод там, где его нет. Терминал
            // подтверждается только DOM-признаком либо словом + canvas на экране.
            const term = !!termEl || (termWord && canvases.length > 0);
            return {
                detected: /neal\.fun\/not-a-robot|I'm Not a Robot/i.test(hay),
                terminal: term,
                level: level ? Number(level[1]) : null,
                name: level ? level[2].trim().substring(0, 120) : null,
                canvases,
                interactive_count: document.querySelectorAll('button, input, [role="button"], canvas, [onclick], [tabindex]:not([tabindex="-1"])').length
            };
        })(),
        // Лабораторное окружение (TryHackMe / HTB): карточки AttackBox и Lab
        // machine со статусом и кнопкой запуска. Без этого блока модель
        // «видела» включённую машину, которой нет: открывала URL лаборатории,
        // получала 502 Bad Gateway и ходила между вкладками до лимита шагов.
        lab: (() => {
            const body = document.body ? document.body.innerText : '';
            if (!/attack\s*box|lab\s*machine|virtual machine/i.test(body)) return {detected: false};
            const startRe = /start\s*(attack\s*box|lab\s*machine|machine|vm|environment)/i;
            const machines = [];
            // Карточка = ближайший контейнер-предок кнопки запуска, в тексте
            // которого есть «Status» и название машины.
            for (const b of document.querySelectorAll('button, [role="button"], a')) {
                const t = (b.innerText || b.value || b.getAttribute('aria-label') || '').trim();
                if (!startRe.test(t) && !/^(stop|shutdown)$/i.test(t)) continue;
                let card = b;
                for (let d = 0; d < 6 && card.parentElement; d++) {
                    card = card.parentElement;
                    if (/status/i.test(card.innerText || '')) break;
                }
                const txt = (card.innerText || '').replace(/\s+/g, ' ').trim().substring(0, 220);
                const off = /\bstatus:?\s*(off|stopped|not started|остановл)/i.test(txt);
                const on = /\bstatus:?\s*(on|running|active|запущ)/i.test(txt);
                machines.push({
                    button: t.substring(0, 60),
                    eid: b.getAttribute('data-eid') || '',
                    status: off ? 'OFF' : (on ? 'ON' : '?'),
                    card: txt
                });
            }
            // Реальные URL запущенной лаборатории. Иначе модель берёт из текста
            // страницы ШАБЛОН «LAB_WEB_URL.p.thmlabs.com» (или «{...}», «<...>»)
            // и подставляет его буквально — навигация висит 30с по таймауту.
            // Настоящий адрес появляется отдельной ссылкой после старта:
            // THM — поддомен от IP «10-x-x-x.reverse-proxy…vm.tryhackme.com»,
            // HTB/прочие — «*.machines.htb», «thmlabs.com» и т.п.
            const realUrlRe = /(?:https?:\/\/(?:\d{1,3}[-.]){3}\d{1,3}[^\s"'<>]*|(?:https?:\/\/)[\w.-]*(?:thmlabs\.com|vm\.tryhackme\.com|hackthebox\.machines\.htb|machines\.htb|reverse-proxy)[^\s"'<>]*)/gi;
            const labLinks = [];
            const seenUrl = new Set();
            // 1) href ссылок — надёжнее текста: это DOM, а не выдержка из prose
            for (const a of document.querySelectorAll('a[href]')) {
                const h = a.href || '';
                if (!/thmlabs\.com|vm\.tryhackme\.com|hackthebox\.machines\.htb|machines\.htb|reverse-proxy|(?:\d{1,3}[-.]){3}\d{1,3}/i.test(h)) continue;
                // Отбрасываем шаблон-плейсхолдеры LAB_WEB_URL / {…} / <…>.
                if (/lab[_-]?web[_-]?url|[<>{}]/i.test(h)) continue;
                if (seenUrl.has(h)) continue;
                seenUrl.add(h);
                labLinks.push({
                    url: h.substring(0, 200),
                    eid: a.getAttribute('data-eid') || '',
                    label: (a.innerText || a.getAttribute('aria-label') || '').trim().substring(0, 40)
                });
            }
            // 2) URL прямо в тексте карточек/заданий (THM печатает адрес текстом)
            for (const u of (body.match(realUrlRe) || [])) {
                if (/lab[_-]?web[_-]?url|[<>{}]/i.test(u)) continue;
                if (seenUrl.has(u)) continue;
                seenUrl.add(u);
                labLinks.push({ url: u.substring(0, 200), eid: '', label: '' });
            }
            // IP машины из карточек (для терминала/AttackBox, не для браузера).
            const ips = [...new Set((body.match(/\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g) || []))].slice(0, 6);
            const offCount = machines.filter(m => m.status === 'OFF').length;
            return {
                detected: machines.length > 0,
                all_on: machines.length > 0 && offCount === 0,
                off_count: offCount,
                machines: machines.slice(0, 6),
                lab_links: labLinks.slice(0, 6),
                ips
            };
        })(),
        flags: {
            captcha: !!document.querySelector('iframe[src*="recaptcha"], iframe[src*="hcaptcha"], .g-recaptcha, .h-captcha'),
            cookie_banner: !!Array.from(document.querySelectorAll('div, section, dialog')).find(el => {
                const id = (el.id + ' ' + el.className).toLowerCase();
                return /cookie|consent|gdpr/.test(id) && el.getBoundingClientRect().height > 30;
            }),
            login_form: !!document.querySelector('input[type="password"]')
        },
        // Подписи и подсказки рядом с полями: ВСЕ тексты рядом с input,
        // БЕЗ фильтра по ключевым словам — семантику оценивает LLM
        // по тексту и цвету (как человек глазами)
        field_context: (() => {
            const out = [];
            const seen = new Set();
            for (const input of document.querySelectorAll('input, textarea, select')) {
                const r = input.getBoundingClientRect();
                if (r.width < 30) continue;
                const eid = input.getAttribute('data-eid') || '?';
                // Маска/placeholder/текущее значение: человек видит "+7 (___) ___"
                // прямо в поле — даём модели то же самое
                const mask = input.placeholder || input.getAttribute('data-mask') || '';
                const curValue = (input.value || '').substring(0, 40);
                const label = input.closest('label');
                const labelText = label ? (label.innerText || '').trim().substring(0, 80) : '';
                const info = { eid };
                if (mask) info.mask = mask;
                if (curValue) info.current_value = curValue;
                if (labelText) info.label = labelText;
                if (mask || curValue || labelText) out.push(info);
                for (const sib of (input.closest('label') || input.closest('div, fieldset, p') || input.parentElement || document.body).querySelectorAll('label, div, span, p, small, strong, em')) {
                    if (sib.contains(input) || sib.querySelector('input, textarea, select')) continue;
                    const sr = sib.getBoundingClientRect();
                    if (sr.width < 30 || sr.height < 8) continue;
                    // Рядом с полем (ниже или выше на 60px)
                    if (Math.abs(sr.top - r.bottom) > 60 && Math.abs(r.top - sr.bottom) > 60) continue;
                    const text = (sib.innerText || '').trim();
                    if (text.length < 2 || text.length > 200 || seen.has(text)) continue;
                    seen.add(text);
                    out.push({ eid, text, color: window.getComputedStyle(sib).color });
                }
            }
            return out.slice(0, 12);
        })(),
        // Открытые модалки/dialog — для state_diff и для решения LLM.
        // Кнопки внутри диалога критичны: без них модель видит только текст
        // и пытается закрыть окно крестиком, попадая в цикл «открылось→закрылось».
        dialog: (() => {
            const dialogs = [...document.querySelectorAll(
                'dialog[open], [role="dialog"], [role="alertdialog"], [class*="modal" i], [class*="Modal"], [class*="popup" i]'
            )].filter(d => {
                const r = d.getBoundingClientRect();
                const s = window.getComputedStyle(d);
                return r.width > 100 && r.height > 50 && s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0';
            });
            // Кнопка, которая ПРОДОЛЖАЕТ работу (а не отменяет её).
            // ВАЖНО: \b в JS не работает с кириллицей (ASCII word boundary),
            // поэтому для «ок» границы задаём явно через \s и якоря.
            const proceedRe = /сохран|продолж|подтвер|примен|отправ|далее|дальше|готово|заверш|выбрать|выбрана|принять|agree|accept|continue|save|confirm|submit|next|apply|done|select|(?:^|\s)ok(?:\s|$)|(?:^|\s)okay(?:\s|$)|(?:^|\s)ок(?:\s|$)/i;
            const cancelRe = /закрыть|отмен|сброс|не сейчас|позже|крест|close|cancel|dismiss|skip|later|×|✕|✖/i;
            const buttons = [];
            let hasProceed = false, hasCancel = false, hasRequired = false, hasInput = false;
            for (const d of dialogs) {
                for (const b of d.querySelectorAll('[data-eid]')) {
                    const tag = b.tagName.toLowerCase();
                    const txt = (b.innerText || b.value || b.getAttribute('aria-label') || '').trim().substring(0, 60);
                    const isBtn = tag === 'button' || b.getAttribute('role') === 'button'
                        || tag === 'input' && ['submit', 'button'].includes((b.getAttribute('type') || '').toLowerCase());
                    if (isBtn) {
                        if (proceedRe.test(txt)) hasProceed = true;
                        if (cancelRe.test(txt)) hasCancel = true;
                        if (txt || b.getAttribute('aria-label')) {
                            buttons.push({ eid: b.getAttribute('data-eid'), text: txt || b.getAttribute('aria-label') });
                        }
                    }
                    const isField = tag === 'input' || tag === 'select' || tag === 'textarea'
                        || b.getAttribute('role') === 'combobox'
                        || b.getAttribute('contenteditable') === 'true';
                    if (isField && (b.getAttribute('type') || '') !== 'hidden') hasInput = true;
                    if (isField && b.required) hasRequired = true;
                }
            }
            const titleEl = dialogs.length ? dialogs[0].querySelector('h1, h2, h3, [role="heading"]') : null;
            return {
                open: dialogs.length > 0,
                count: dialogs.length,
                text: dialogs.map(d => (d.innerText || '').substring(0, 300)).join(' | ').substring(0, 500),
                title: titleEl ? (titleEl.innerText || '').trim().substring(0, 120) : '',
                buttons: buttons.slice(0, 12),
                // blocking: окно ТРЕБУЕТ действия, а не информирует.
                // Обязательное поле — точно блокирует. Просто поле + кнопка
                // подтверждения — тоже (форма данных). Промо-окно с одной
                // кнопкой «ОК» и без полей остаётся закрываемым.
                blocking: dialogs.length > 0 && (hasRequired || (hasInput && hasProceed)),
                has_proceed: hasProceed,
                has_cancel: hasCancel,
                has_input: hasInput,
                has_required_field: hasRequired
            };
        })(),
        // Заметные сообщения: элементы с ARIA-ролями ИЛИ выделенные цветом
        // (красный/оранжевый/зелёный текст) — БЕЗ фильтра по словам.
        // Человек замечает такое глазами; здесь даём модели то же самое:
        // текст + цвет, семантику (ошибка/успех/инфо) оценивает LLM.
        notices: (() => {
            const out = [];
            const seen = new Set();
            const push = (el) => {
                const r = el.getBoundingClientRect();
                const s = window.getComputedStyle(el);
                if (r.width < 30 || r.height < 10 || s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') return;
                const text = (el.innerText || '').trim();
                if (text.length < 3 || text.length > 300 || seen.has(text)) return;
                seen.add(text);
                out.push({ text, color: s.color, role: el.getAttribute('role') || undefined });
            };
            // 1. ARIA-роли уведомлений (стандартный способ)
            for (const el of document.querySelectorAll('[role="alert"], [role="status"], [aria-live="assertive"], [aria-live="polite"]')) {
                push(el);
            }
            // 2. Цветной текст: красные/оранжевые/зелёные оттенки = визуальное выделение
            for (const el of document.querySelectorAll('div, span, p, li, td, strong, b')) {
                if (el.closest('[role="alert"], [role="status"]')) continue;
                const s = window.getComputedStyle(el);
                const m = s.color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
                if (!m) continue;
                const [rr, gg, bb] = [+m[1], +m[2], +m[3]];
                const isReddish = rr > 130 && rr > gg * 1.4 && rr > bb * 1.4;
                const isOrange = rr > 150 && gg > 80 && gg < rr * 0.85 && bb < 100;
                const isGreen = gg > 100 && gg > rr * 1.3 && gg > bb * 1.3;
                if (isReddish || isOrange || isGreen) push(el);
            }
            return out.slice(0, 8);
        })()
    };
}
"""


class BrowserState:
    """Снимок состояния страницы для LLM"""

    def __init__(self, data: Dict[str, Any]):
        self.url: str = data.get("url", "")
        self.title: str = data.get("title", "")
        self.ready_state: str = data.get("readyState", "")
        self.scroll_y: int = data.get("scrollY", 0)
        self.page_height: int = data.get("pageHeight", 0)
        self.viewport_height: int = data.get("viewportHeight", 0)
        self.viewport_width: int = data.get("viewportWidth", 0)
        self.elements: List[Dict[str, Any]] = data.get("elements", [])
        self.page_text: str = data.get("page_text", "")
        # Inline-скрипты страницы: исходный код, которого нет в
        # innerText. Без него агент слеп к подсказкам вида
        # «посмотри исходный код страницы» (PortSwigger lab).
        self.page_scripts: str = data.get("scripts", "")
        self.flags: Dict[str, bool] = data.get("flags", {})
        self.headings: List[str] = data.get("headings", [])
        self.notices: List[Dict[str, Any]] = data.get("notices", [])
        self.field_context: List[Dict[str, Any]] = data.get("field_context", [])
        self.scroll_panes: List[Dict[str, Any]] = data.get("scroll_panes", [])
        self.dialog: Dict[str, Any] = data.get("dialog", {"open": False, "count": 0, "text": ""})
        self.game: Dict[str, Any] = data.get("game", {})
        self.lab: Dict[str, Any] = data.get("lab", {})
        # Список вкладок браузера: [{tab_id, url, title, is_active}].
        # Без него агент видит только активную страницу и не знает,
        # что лаба/ссылка открылась в НОВОЙ вкладке (зацикливание на
        # кнопке, которая уже сработала). Заполняет loop._attach_tabs.
        self.tabs: List[Dict[str, Any]] = data.get("tabs", [])

    def at_bottom(self) -> bool:
        return self.scroll_y + self.viewport_height >= self.page_height - 50

    def find_element(self, eid: str) -> Optional[Dict[str, Any]]:
        for el in self.elements:
            if el.get("eid") == eid:
                return el
        return None

    def to_llm_text(self, max_elements: int = 120) -> str:
        """Компактное текстовое представление для промпта LLM"""
        lines = [
            f"URL: {self.url}",
            f"Заголовок: {self.title}",
            f"Прокрутка: {self.scroll_y}/{self.page_height}px" + (" (внизу страницы)" if self.at_bottom() else ""),
            f"Окно браузера: {self.viewport_width}×{self.viewport_height}px — координаты click_at/type не должны выходить за эти пределы",
        ]

        # Вкладки: критично для лаб, открывающихся в новой вкладке
        # (PortSwigger ACCESS THE LAB, target=_blank). Без этого блока
        # агент считает, что кнопка «не сработала», и жмёт её повторно.
        if self.tabs:
            lines.append(
                f"🗂 ВКЛАДКИ БРАУЗЕРА ({len(self.tabs)} шт). Ты сейчас работаешь "
                "только с АКТИВНОЙ (⭐); содержимое и элементы других вкладок тебе "
                "не видны, чтобы на них перейти — switch_tab(tab_id):"
            )
            for t in self.tabs:
                active = " ⭐(активна — её элементы ты видишь ниже)" if t.get("is_active") else ""
                title = (t.get("title") or "").strip()[:60]
                lines.append(f"   • [{t.get('tab_id')}] {(t.get('url') or '')[:100]} «{title}»{active}")
            if len(self.tabs) > 1:
                lines.append(
                    "   ⚠️ Если цель задания (лабораторная страница, форма ответа, "
                    "терминал) находится в НЕактивной вкладке — это НЕ значит, что "
                    "кнопка не сработала: она уже сработала и открыла вкладку. "
                    "Переключись switch_tab, а не нажимай ту же кнопку заново."
                )

        if self.headings:
            lines.append("Заголовки страницы: " + " | ".join(self.headings[:6]))

        if self.scroll_panes:
            lines.append(
                "🪟 ПАНЕЛИ С ВНУТРЕННИМ СКРОЛЛОМ: окно браузера здесь не крутится, "
                "скроллится каждая панель. Элемент помечен (панель pN) — чтобы "
                "достать скрытое, прокрути ЭТУ панель: "
                "scroll(direction=\"down\", element_id=\"pN\")."
            )
            for p in self.scroll_panes:
                state = "внизу" if p.get("at_bottom") else \
                    f"прокручено {p.get('scroll_y')} из {p.get('scroll_height')}px"
                lines.append(
                    f"   • [{p['id']}] {p['width']}×{p['height']}px центр=({p['x']},{p['y']}) "
                    f"{state}" + (f" начало текста: «{p['label']}»" if p.get("label") else "")
                )

        if self.lab.get("detected"):
            off = [m for m in (self.lab.get("machines") or []) if m.get("status") == "OFF"]
            if off:
                names = ", ".join(
                    f"«{m['button']}»" + (f" (id {m['eid']})" if m.get("eid") else "")
                    for m in off)
                lines.append(
                    f"⚠️ ЛАБОРАТОРНОЕ ОКРУЖЕНИЕ НЕ ЗАПУЩЕНО ({len(off)} шт: {names}). "
                    "Машины ВЫКЛЮЧЕНЫ (Status: Off) — это факт со страницы, не "
                    "предположение. Целевой IP/URL машины сейчас недоступен, "
                    "обращение к нему даст 502 Bad Gateway. СНАЧАЛА нажми кнопку "
                    "запуска (click по id выше), дождись Status: On (wait 10-30с, "
                    "при необходимости повтори wait), и только потом открывай "
                    "URL машины. НЕ придумывай, что машина уже включена."
                )
            else:
                lines.append("🖥 Лабораторное окружение: все машины запущены (Status: On).")
            # Реальные адреса лаборатории, собранные из DOM (href/текст).
            # Критично: в тексте страницы всегда есть ШАБЛОН вида
            # LAB_WEB_URL.p.thmlabs.com / {…} / <…> — его открывать нельзя,
            # он не резолвится и вешает навигацию на 30с. Модель должна брать
            # адрес из этого списка, а не выдумывать из prose.
            real_links = self.lab.get("lab_links") or []
            if real_links:
                lines.append(
                    "🔗 РЕАЛЬНЫЕ АДРЕСА ЛАБОРАТОРИИ (брать отсюда, НЕ из текста задания):"
                )
                for l in real_links:
                    tag = f" [id {l['eid']}]" if l.get("eid") else ""
                    lbl = f" — «{l['label']}»" if l.get("label") else ""
                    lines.append(f"   • {l['url']}{tag}{lbl}")
                lines.append(
                    "   ⛔ Не подставляй в URL шаблоны LAB_WEB_URL, {…}, <…> — "
                    "это заглушки из описания, их навигация завернётся по таймауту."
                )
            ips = self.lab.get("ips") or []
            if ips:
                lines.append(
                    "🧭 IP машины (для nmap/терминала, не для браузера): " + ", ".join(ips)
                )

        if self.game.get("detected"):
            level = self.game.get("level")
            name = self.game.get("name") or "неизвестный уровень"
            lines.append(f"🎮 ИГРА NEAL.FUN: уровень {level or '?'} — {name}")
            lines.append(f"Игровых canvas: {len(self.game.get('canvases', []))}; интерактивных областей: {self.game.get('interactive_count', 0)}")

        if self.game.get("terminal"):
            lines.append(
                "⌨️ CANVAS-ТЕРМИНАЛ (Pwnbox/noVNC): полей ввода в DOM нет. "
                "Вывод читай ТОЛЬКО со скриншота; команды печатай через "
                "type без element_id (текст уйдёт в фокус), при необходимости "
                "сначала click_at по центру окна терминала, submit=true = Enter."
            )
        if self.flags.get("captcha"):
            lines.append(" На странице CAPTCHA")
        if self.flags.get("cookie_banner"):
            lines.append("🍪 Обнаружен cookie-баннер (найди кнопку accept и кликни)")
        if self.flags.get("login_form"):
            lines.append("🔐 На странице есть форма входа (input[type=password])")
        if self.dialog.get("open"):
            lines.append(f"🪟 ОТКРЫТО МОДАЛЬНОЕ ОКНО ({self.dialog.get('count')}): {self.dialog.get('text', '')[:300]}")
            if self.dialog.get("blocking"):
                why = []
                if self.dialog.get("has_proceed"):
                    why.append("есть кнопка подтверждения")
                if self.dialog.get("has_required_field"):
                    why.append("есть обязательное поле")
                lines.append(
                    "   ⛔ БЛОКИРУЮЩЕЕ ОКНО (" + ", ".join(why) + "): оно требует действия. "
                    "Закрытие крестиком/«Сбросить»/Esc/«Не сейчас» НЕ подходит — "
                    "окно откроется снова. Нажми кнопку подтверждения (Сохранить и продолжить)."
                )
            btns = self.dialog.get("buttons") or []
            if btns:
                lines.append("   Кнопки окна: " + ", ".join(
                    f"[{b['eid']}] «{b['text']}»" for b in btns[:10]
                ))
        if self.notices:
            lines.append(" ВЫДЕЛЕННЫЕ СООБЩЕНИЯ НА СТРАНИЦЕ (цвет/роль — оцени семантику сам):")
            for n in self.notices:
                color = n.get("color", "")
                role = f", role={n['role']}" if n.get("role") else ""
                lines.append(f"   • [{color}{role}] {n['text'][:200]}")
        if self.field_context:
            lines.append("📝 КОНТЕКСТ ПОЛЕЙ ВВОДА (маска, текущее значение, подписи, подсказки):")
            for fc in self.field_context:
                bits = []
                if fc.get("mask"):
                    bits.append(f"маска/placeholder: «{fc['mask']}»")
                if fc.get("current_value"):
                    bits.append(f"уже введено: «{fc['current_value']}»")
                if fc.get("label"):
                    bits.append(f"подпись: «{fc['label']}»")
                if fc.get("text"):
                    bits.append(f"[{fc.get('color', '')}] {fc['text'][:150]}")
                if bits:
                    lines.append(f"   • поле {fc.get('eid')}: " + "; ".join(bits))

        lines.append("")
        lines.append("=== ИНТЕРАКТИВНЫЕ ЭЛЕМЕНТЫ (id: описание) ===")

        for el in self.elements[:max_elements]:
            parts = [f"[{el['eid']}]"]
            parts.append(f"<{el['tag']}>")
            if el.get("type"):
                parts.append(f"type={el['type']}")
            if el.get("text"):
                parts.append(f'"{el["text"]}"')
            if el.get("placeholder"):
                parts.append(f'placeholder="{el["placeholder"]}"')
            if el.get("name"):
                parts.append(f"name={el['name']}")
            if el.get("href"):
                parts.append(f"href={el['href']}")
            if el.get("x") is not None and el.get("y") is not None:
                parts.append(f"center=({el['x']},{el['y']})")
            if el.get("width") and el.get("height"):
                parts.append(f"size={el['width']}x{el['height']}")
            if el.get("cursor") == "pointer":
                parts.append("cursor=pointer")
            if el.get("img"):
                parts.append(f"img={el['img']}")
            if el.get("colors"):
                c = el["colors"]
                top = sorted(c.items(), key=lambda kv: -kv[1])[:3]
                parts.append("цвета:" + ",".join(f"{k}={v}%" for k, v in top if v > 5))
            if el.get("selected"):
                parts.append("[ВЫБРАНО]")
            if el.get("form"):
                parts.append(f"form={el['form']}")
            if el.get("value") and el.get("tag") in ("input", "textarea"):
                parts.append(f'value="{el["value"]}"')  # заполненные поля видны явно
            if el.get("required"):
                parts.append("*required")
            if el.get("disabled"):
                parts.append("[disabled]")
            if el.get("checked"):
                parts.append("[checked]")
            if el.get("in_dialog"):
                parts.append("(в модальном окне)")
            if not el.get("in_viewport"):
                if el.get("pane"):
                    parts.append(f"(ниже видимой части панели {el['pane']})")
                else:
                    parts.append("(вне экрана, нужен scroll)")
            lines.append(" ".join(parts))

        if len(self.elements) > max_elements:
            lines.append(f"... и ещё {len(self.elements) - max_elements} элементов (прокрути страницу)")

        scripts = (self.page_scripts or "").strip()
        if scripts:
            lines.append("")
            lines.append(
                "=== ИСХОДНЫЙ КОД СТРАНИЦЫ: inline <script> (в ТЕКСТЕ страницы "
                "этого НЕТ) ==="
            )
            lines.append(scripts[:6000])
            lines.append(
                "   ⚠️ Если задание говорит «посмотри исходный код / JavaScript "
                "страницы» — путь к цели обычно здесь: ищи строки вида "
                "/administrator-panel…, /admin/…, var url = «/…», fetch(«/…»). "
                "Нашёл — открывай этот путь через navigate, а не ищи кнопку на "
                "странице и не лезь в robots.txt."
            )

        lines.append("")
        # 2000 симв. обрезали середину длинных страниц: условие задания и
        # кнопка «Download Task Files» оставались непрочитанными, и агент
        # решал по чужим walkthrough. Полный текст — через extract.
        full_text = self.page_text or ""
        cap = 8000
        lines.append("=== ТЕКСТ СТРАНИЦЫ (фрагмент) ===")
        lines.append(full_text[:cap])
        if len(full_text) > cap:
            lines.append(
                f"… текст усечён: показано {cap} из {len(full_text)} символов. "
                f"Если нужного фрагмента здесь нет — НЕ додумывай его, "
                f"прочитай страницу целиком: extract(selector=\"body\")."
            )

        return "\n".join(lines)


async def capture_browser_state(page) -> BrowserState:
    """
    Захват BrowserState со страницы Playwright.

    ВАЖНО: обходит ВСЕ фреймы, включая кросс-origin (Ozon ID, виджеты оплаты,
    reCAPTCHA и т.п.) — через Playwright frames API, которому same-origin policy
    не мешает. Элементы фреймов получают продолжение нумерации eid (eN+1...),
    а data-eid ставится в DOM фрейма — click_element/type_into_element работают
    с ними через frame locator.
    """
    data = await page.evaluate(STATE_JS, 1)  # нумерация с e1
    elements = list(data.get("elements", []))
    counter = 1 + len(elements)

    # Кросс-origin и дочерние фреймы
    for frame in page.frames:
        if frame == page.main_frame:
            continue
        try:
            frame_data = await frame.evaluate(STATE_JS, counter)
        except Exception as e:
            logger.debug(f"Frame {frame.url[:60]} skipped: {e}")
            continue
        frame_elements = frame_data.get("elements", [])
        if not frame_elements:
            continue
        # Пометка фрейма для последующих действий
        frame_url = (frame.url or "")[:100]
        for el in frame_elements:
            el["frame_url"] = frame_url
        elements.extend(frame_elements)
        counter += len(frame_elements)
        # Скрипты фрейма — там же, где и текст лабы (iframe-стенды)
        fscripts = (frame_data.get("scripts") or "").strip()
        if fscripts:
            data["scripts"] = ((data.get("scripts") or "") + "\n" + fscripts[:2000])[:8000]
        # Текст фрейма добавляем к page_text
        ftext = frame_data.get("page_text", "")
        if ftext:
            data["page_text"] = (data.get("page_text", "")) + "\n--- [содержимое фрейма " + frame_url[:60] + "] ---\n" + ftext[:1500]
        # Флаги из фрейма (captcha часто в iframe)
        fflags = frame_data.get("flags", {})
        for k, v in fflags.items():
            if v:
                data.setdefault("flags", {})[k] = True
        # Заметные сообщения и контекст полей из фрейма (Ozon ID и т.п.)
        for key in ("notices", "field_context"):
            fitems = frame_data.get(key, [])
            if fitems:
                data.setdefault(key, []).extend(fitems)
        # Game/terminal из фрейма: Pwnbox и noVNC живут внутри iframe,
        # их canvas и признак терминала поднимаем в главный документ.
        fgame = frame_data.get("game", {})
        if fgame:
            g = data.setdefault("game", {})
            if fgame.get("canvases"):
                g.setdefault("canvases", []).extend(fgame["canvases"])
            for k in ("detected", "terminal"):
                if fgame.get(k):
                    g[k] = True
        # Dialog из фрейма
        fdialog = frame_data.get("dialog", {})
        if fdialog.get("open") and not data.get("dialog", {}).get("open"):
            data["dialog"] = fdialog

    data["elements"] = elements
    return BrowserState(data)
