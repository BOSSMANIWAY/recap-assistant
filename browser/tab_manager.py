"""
Tab Manager - Управление вкладками, группировка, порядок
"""

import logging
from typing import Dict, List, Optional, Tuple
from browser.controller import TabInfo, BrowserController

logger = logging.getLogger(__name__)


class TabManager:
    """Менеджер вкладок с группировкой и сортировкой"""

    def __init__(self, controller: BrowserController):
        self.controller = controller

    async def list_tabs(self, verbose: bool = False) -> str:
        """Список всех вкладок"""
        tabs = await self.controller.get_tabs()
        if not tabs:
            return "Нет открытых вкладок"

        lines = ["\n📑 Открытые вкладки:", ""]
        for i, tab in enumerate(tabs, 1):
            active = " ⭐" if tab.is_active else ""
            lines.append(f"  {i}. {tab.title}")
            lines.append(f"     URL: {tab.url[:80]}{'...' if len(tab.url) > 80 else ''}{active}")
        lines.append("")
        return "\n".join(lines)

    async def auto_group_tabs(self) -> Dict[str, List[TabInfo]]:
        """
        Автоматическая группировка вкладок по доменам/категориям.
        """
        tabs = await self.controller.get_tabs()
        groups: Dict[str, List[TabInfo]] = {}

        for tab in tabs:
            # Извлекаем домен
            try:
                domain = tab.url.split("//")[1].split("/")[0].split(":")[0]
                # Определяем категорию по домену
                category = self._categorize_domain(domain)
                if category not in groups:
                    groups[category] = []
                groups[category].append(tab)
            except Exception:
                if "Uncategorized" not in groups:
                    groups["Uncategorized"] = []
                groups["Uncategorized"].append(tab)

        logger.info(f"Auto-grouped {len(tabs)} tabs into {len(groups)} categories")
        return groups

    def _categorize_domain(self, domain: str) -> str:
        """Категоризация домена"""
        categories = {
            "Social": ["facebook.com", "twitter.com", "x.com", "instagram.com",
                       "linkedin.com", "reddit.com", "tiktok.com"],
            "Work": ["gmail.com", "google.com", "outlook.com", "slack.com",
                     "zoom.us", "teams.microsoft.com", "notion.so", "github.com"],
            "Shopping": ["amazon.com", "aliexpress.com", "ebay.com", "ozon.ru",
                         "wildberries.ru"],
            "Entertainment": ["youtube.com", "netflix.com", "twitch.tv",
                              "spotify.com", "vimeo.com"],
            "News": ["habr.com", "medium.com", "bbc.com", "cnn.com",
                     "news.ycombinator.com"],
            "Dev": ["stackoverflow.com", "dev.to", "medium.com", "docs.python.org"],
        }

        for category, domains in categories.items():
            for d in domains:
                if d in domain:
                    return category

        return "General"

    async def reorder_tabs(self, order: List[str]):
        """
        Переупорядочивание вкладок.
        order: список tab_id в нужном порядке
        """
        tabs = await self.controller.get_tabs()
        tab_map = {t.tab_id: t for t in tabs}

        # Сохраняем активную вкладку
        active_id = next((t.tab_id for t in tabs if t.is_active), None)

        # Переходим по вкладкам в новом порядке
        for tab_id in order:
            if tab_id in tab_map:
                await self.controller.switch_tab(tab_id)

        # Восстанавливаем активную
        if active_id:
            await self.controller.switch_tab(active_id)

        logger.info(f"Reordered {len(order)} tabs")

    async def close_duplicates(self) -> int:
        """Закрытие дублирующихся вкладок (одинаковый URL)"""
        tabs = await self.controller.get_tabs()
        url_counts: Dict[str, List[TabInfo]] = {}

        for tab in tabs:
            if tab.url not in url_counts:
                url_counts[tab.url] = []
            url_counts[tab.url].append(tab)

        closed = 0
        for url, url_tabs in url_counts.items():
            if len(url_tabs) > 1:
                # Закрываем все кроме первой
                for tab in url_tabs[1:]:
                    try:
                        await self.controller.close_tab(tab.tab_id)
                        closed += 1
                    except Exception as e:
                        logger.warning(f"Failed to close duplicate: {e}")

        logger.info(f"Closed {closed} duplicate tabs")
        return closed

    async def summarize_all_tabs(self) -> str:
        """Суммаризация всех открытых вкладок"""
        tabs = await self.controller.get_tabs()
        summaries = []

        for tab in tabs:
            summary = await self._summarize_tab(tab)
            summaries.append(f"[TAB] {tab.title}\n   URL: {tab.url}\n   {summary}\n")

        return "\n".join(summaries)

    async def _summarize_tab(self, tab: TabInfo) -> str:
        """Суммаризация одной вкладки"""
        # Получаем текст страницы
        text = await self.controller._page.evaluate("""
            () => {
                // Получаем основной текст, убирая навигацию и скрипты
                const content = document.querySelector('main') || document.body;
                const text = content.innerText || content.textContent;
                // Обрезаем до разумного размера
                return text.substring(0, 5000).trim();
            }
        """)

        # Убираем лишние пробелы
        text = " ".join(text.split())
        if len(text) > 500:
            text = text[:500] + "..."

        return f"Текст: {text}"