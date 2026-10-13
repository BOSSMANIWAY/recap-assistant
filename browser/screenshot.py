"""
Screenshot & DOM Capture - Скриншоты и снимки DOM для анализа LLM
"""

import base64
import logging
from typing import Dict, Any, Optional
from browser.controller import BrowserController

logger = logging.getLogger(__name__)


class ScreenshotCapture:
    """Захват скриншотов и DOM-снимков страницы"""

    def __init__(self, controller: BrowserController):
        self.controller = controller

    async def capture_full_page(self, quality: int = 80) -> Dict[str, Any]:
        """
        Полный скриншот страницы (включая прокрутку).
        Возвращает base64-encoded PNG + метаданные.
        """
        if not self.controller._page:
            raise RuntimeError("Browser not connected")

        page = self.controller._page

        # Полный скриншот
        screenshot_b64 = await page.screenshot(full_page=True, type='jpeg', quality=quality)
        screenshot_data = base64.b64encode(screenshot_b64).decode("utf-8")

        # Метаданные
        metrics = await self.controller.get_page_metrics()

        return {
            "screenshot": screenshot_data,
            "format": "jpeg",
            "metrics": metrics,
            "url": page.url,
            "title": await page.title()
        }

    async def capture_viewport(self, quality: int = 80) -> Dict[str, Any]:
        """Скриншот только видимой области (viewport)"""
        if not self.controller._page:
            raise RuntimeError("Browser not connected")

        page = self.controller._page
        screenshot_b64 = await page.screenshot(full_page=False, type='jpeg', quality=quality)
        screenshot_data = base64.b64encode(screenshot_b64).decode("utf-8")

        return {
            "screenshot": screenshot_data,
            "format": "jpeg",
            "viewport": page.viewport_size,
            "url": page.url
        }

    async def capture_dom_snapshot(self) -> Dict[str, Any]:
        """
        Снимок DOM-структуры страницы.
        Включает текст, ссылки, формы, кнопки, мета-данные, контекст.
        """
        if not self.controller._page:
            raise RuntimeError("Browser not connected")

        page = self.controller._page

        dom_data = await page.evaluate("""
            () => {
                const result = {
                    title: document.title,
                    url: window.location.href,
                    meta: {},
                    text: document.body.innerText.substring(0, 10000),
                    links: [],
                    buttons: [],
                    forms: [],
                    inputs: [],
                    selects: [],
                    tables: [],
                    images: [],
                    iframes: [],
                    alerts: [],
                    nav: [],
                    main_content: '',
                    header: '',
                    footer: ''
                };

                // Мета-данные
                const metas = document.querySelectorAll('meta');
                metas.forEach(m => {
                    if (m.name || m.property) {
                        result.meta[m.name || m.property] = m.content;
                    }
                });
                const ogTitle = document.querySelector('meta[property="og:title"]');
                if (ogTitle) result.meta.og_title = ogTitle.content;
                const ogDesc = document.querySelector('meta[property="og:description"]');
                if (ogDesc) result.meta.og_description = ogDesc.content;

                // Хедер
                const header = document.querySelector('header, [role="banner"]');
                if (header) result.header = header.innerText.substring(0, 2000);

                // Футер
                const footer = document.querySelector('footer, [role="contentinfo"]');
                if (footer) result.footer = footer.innerText.substring(0, 2000);

                // Навигация
                const nav = document.querySelector('nav, [role="navigation"]');
                if (nav) result.nav = nav.innerText.substring(0, 2000);

                // Основной контент
                const main = document.querySelector('main, [role="main"], article');
                if (main) result.main_content = main.innerText.substring(0, 5000);

                // Ссылки с контекстом
                document.querySelectorAll('a').forEach((a, i) => {
                    if (i < 50) {
                        const rect = a.getBoundingClientRect();
                        result.links.push({
                            text: a.innerText.substring(0, 100),
                            href: a.href,
                            aria_label: a.getAttribute('aria-label'),
                            role: a.getAttribute('role'),
                            visible: rect.width > 0 && rect.height > 0,
                            classes: a.className.substring(0, 100)
                        });
                    }
                });

                // Кнопки с контекстом
                document.querySelectorAll('button, [role="button"], input[type="submit"], [onclick]').forEach((b, i) => {
                    if (i < 30) {
                        const rect = b.getBoundingClientRect();
                        result.buttons.push({
                            text: (b.innerText || b.value || b.textContent || '').substring(0, 100),
                            type: b.type || 'button',
                            aria_label: b.getAttribute('aria-label'),
                            role: b.getAttribute('role'),
                            disabled: b.disabled,
                            visible: rect.width > 0 && rect.height > 0,
                            classes: b.className.substring(0, 100),
                            data_attrs: {}
                        });
                        // Data-атрибуты
                        Array.from(b.attributes).forEach(attr => {
                            if (attr.name.startsWith('data-')) {
                                result.buttons[result.buttons.length-1].data_attrs[attr.name] = attr.value;
                            }
                        });
                    }
                });

                // Формы
                document.querySelectorAll('form').forEach((f, i) => {
                    if (i < 10) {
                        result.forms.push({
                            action: f.action,
                            method: f.method,
                            id: f.id,
                            classes: f.className.substring(0, 100),
                            inputs: f.querySelectorAll('input, select, textarea').length
                        });
                    }
                });

                // Поля ввода
                document.querySelectorAll('input:not([type="submit"]):not([type="button"]), textarea').forEach((inp, i) => {
                    if (i < 30) {
                        const rect = inp.getBoundingClientRect();
                        result.inputs.push({
                            type: inp.type,
                            name: inp.name,
                            id: inp.id,
                            placeholder: inp.placeholder,
                            value: inp.value ? '[HIDDEN]' : '',
                            required: inp.required,
                            disabled: inp.disabled,
                            aria_label: inp.getAttribute('aria-label'),
                            visible: rect.width > 0 && rect.height > 0,
                            classes: inp.className.substring(0, 100)
                        });
                    }
                });

                // Dropdown/select
                document.querySelectorAll('select').forEach((sel, i) => {
                    if (i < 10) {
                        const options = [];
                        sel.querySelectorAll('option').forEach(opt => {
                            options.push({
                                value: opt.value,
                                text: opt.text.substring(0, 50),
                                selected: opt.selected
                            });
                        });
                        result.selects.push({
                            name: sel.name,
                            id: sel.id,
                            multiple: sel.multiple,
                            options: options
                        });
                    }
                });

                // Таблицы
                document.querySelectorAll('table').forEach((tbl, i) => {
                    if (i < 5) {
                        const rows = [];
                        tbl.querySelectorAll('tr').forEach((row, ri) => {
                            if (ri < 10) {
                                const cells = [];
                                row.querySelectorAll('td, th').forEach(cell => {
                                    cells.push(cell.innerText.substring(0, 100));
                                });
                                rows.push(cells);
                            }
                        });
                        result.tables.push({
                            headers: rows[0] || [],
                            rows: rows.slice(1),
                            caption: tbl.caption ? tbl.caption.innerText : ''
                        });
                    }
                });

                // Изображения
                document.querySelectorAll('img').forEach((img, i) => {
                    if (i < 20) {
                        result.images.push({
                            src: img.src.substring(0, 200),
                            alt: img.alt,
                            width: img.width,
                            height: img.height,
                            aria_label: img.getAttribute('aria-label')
                        });
                    }
                });

                // Iframes
                document.querySelectorAll('iframe').forEach((frame, i) => {
                    if (i < 5) {
                        result.iframes.push({
                            src: frame.src,
                            width: frame.width,
                            height: frame.height,
                            title: frame.title
                        });
                    }
                });

                // Alert/Modal detection
                const bodyText = document.body.innerText;
                if (bodyText.includes('Sign in') || bodyText.includes('Log in') ||
                    bodyText.includes('Войти') || bodyText.includes('Авторизация')) {
                    result.alerts.push('login_detected');
                }
                if (bodyText.includes('captcha') || bodyText.includes('reCAPTCHA')) {
                    result.alerts.push('captcha_detected');
                }
                if (bodyText.includes('Cookie') || bodyText.includes('cookie')) {
                    result.alerts.push('cookie_banner_detected');
                }

                return result;
            }
        """)

        return dom_data

    async def capture_step_by_step(self, step_number: int, action: str, url: str) -> Dict[str, Any]:
        """
        Скриншот для пошагового отображения прогресса.
        """
        screenshot = await self.capture_viewport(quality=60)

        return {
            "step": step_number,
            "action": action,
            "url": url,
            "screenshot": screenshot["screenshot"],
            "timestamp": self._get_timestamp()
        }

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    async def highlight_element(self, selector: str) -> Dict[str, Any]:
        """
        Подсветка элемента на скриншоте.
        """
        if not self.controller._page:
            raise RuntimeError("Browser not connected")

        page = self.controller._page

        # Подсвечиваем элемент
        await page.evaluate("""
            (selector) => {
                const el = document.querySelector(selector);
                if (el) {
                    el.style.outline = '3px solid red';
                    el.style.outlineOffset = '3px';
                    el.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
                }
            }
        """, selector)

        # Делаем скриншот
        screenshot = await self.capture_viewport()

        # Убираем подсветку
        await page.evaluate("""
            (selector) => {
                const el = document.querySelector(selector);
                if (el) {
                    el.style.outline = '';
                    el.style.outlineOffset = '';
                    el.style.backgroundColor = '';
                }
            }
        """, selector)

        return screenshot
