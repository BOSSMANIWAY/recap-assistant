"""
Agent Overlay - визуализация действий агента в браузере
Показывает что делает агент прямо на странице Chrome
"""

import asyncio
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# CSS для overlay
OVERLAY_CSS = """
#agent-overlay {
    position: fixed;
    top: 10px;
    right: 10px;
    width: 380px;
    max-height: 80vh;
    background: rgba(0, 0, 0, 0.9);
    color: #00ff00;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    z-index: 999999;
    border-radius: 8px;
    border: 2px solid #00ff00;
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

#agent-overlay-header {
    background: #00ff00;
    color: #000;
    padding: 8px 12px;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
}

#agent-overlay-header:hover {
    background: #00cc00;
}

#agent-overlay-close {
    background: none;
    border: none;
    color: #000;
    font-size: 16px;
    cursor: pointer;
    font-weight: bold;
}

#agent-overlay-body {
    padding: 10px;
    overflow-y: auto;
    max-height: calc(80vh - 40px);
}

#agent-overlay-log {
    list-style: none;
    padding: 0;
    margin: 0;
}

#agent-overlay-log li {
    padding: 4px 0;
    border-bottom: 1px solid rgba(0, 255, 0, 0.2);
    animation: fadeIn 0.3s ease-in;
}

#agent-overlay-log li:last-child {
    border-bottom: none;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}

.agent-action {
    color: #00ff00;
    font-weight: bold;
}

.agent-thought {
    color: #88ff88;
    margin-left: 10px;
    font-style: italic;
}

.agent-result {
    color: #ffff00;
    margin-left: 10px;
}

.agent-error {
    color: #ff4444;
}

.agent-step {
    color: #00ffff;
    font-weight: bold;
}
"""

# HTML для overlay
OVERLAY_HTML = """
<div id="agent-overlay">
    <div id="agent-overlay-header">
        <span> AI Agent</span>
        <button id="agent-overlay-close">×</button>
    </div>
    <div id="agent-overlay-body">
        <ul id="agent-overlay-log"></ul>
    </div>
</div>
"""


class AgentOverlay:
    """Визуализация действий агента на странице"""

    def __init__(self, controller):
        self.controller = controller
        self._initialized = False

    async def initialize(self):
        """Добавляет overlay в страницу"""
        if self._initialized:
            return
        
        page = self.controller._page
        if not page:
            return
        
        try:
            # Добавляем CSS
            await page.add_style_tag(content=OVERLAY_CSS)
            # Добавляем HTML
            await page.evaluate(OVERLAY_HTML)
            
            # Добавляем обработчик закрытия
            await page.evaluate("""
                document.getElementById('agent-overlay-close').addEventListener('click', () => {
                    document.getElementById('agent-overlay').style.display = 'none';
                });
            """)
            
            self._initialized = True
            logger.info("Agent overlay initialized")
        except Exception as e:
            logger.error(f"Failed to initialize overlay: {e}")

    async def log_action(self, step: int, action_type: str, thought: str = "", result: str = ""):
        """Добавляет запись в лог overlay"""
        if not self._initialized:
            await self.initialize()
        
        page = self.controller._page
        if not page:
            return
        
        try:
            # Экранируем одинарные кавычки для JS
            thought_escaped = thought.replace("'", "\\'").replace("\n", "\\n")
            result_escaped = result.replace("'", "\\'").replace("\n", "\\n")
            action_escaped = action_type.replace("'", "\\'")
            
            await page.evaluate(f"""
                () => {{
                    const log = document.getElementById('agent-overlay-log');
                    if (!log) return;
                    
                    const li = document.createElement('li');
                    li.innerHTML = '<span class="agent-step">[Шаг {step}]</span> ' +
                                   '<span class="agent-action">{action_escaped}</span>' +
                                   {'\'<br><span class="agent-thought"> \' + \'{thought_escaped}\' + \'</span>\'' if thought else ''} +
                                   {'\'<br><span class="agent-result"> \' + \'{result_escaped}\' + \'</span>\'' if result else ''};
                    
                    log.appendChild(li);
                    
                    // Автоскролл вниз
                    const body = document.getElementById('agent-overlay-body');
                    body.scrollTop = body.scrollHeight;
                    
                    // Максимум 50 записей
                    while (log.children.length > 50) {{
                        log.removeChild(log.firstChild);
                    }}
                }}
            """)
        except Exception as e:
            logger.error(f"Failed to log action to overlay: {e}")

    async def clear(self):
        """Очищает лог overlay"""
        if not self._initialized:
            return
        
        page = self.controller._page
        if not page:
            return
        
        try:
            await page.evaluate("""
                () => {
                    const log = document.getElementById('agent-overlay-log');
                    if (log) log.innerHTML = '';
                }
            """)
        except Exception as e:
            logger.error(f"Failed to clear overlay: {e}")

    async def hide(self):
        """Скрывает overlay"""
        if not self._initialized:
            return
        
        page = self.controller._page
        if not page:
            return
        
        try:
            await page.evaluate("""
                () => {
                    const overlay = document.getElementById('agent-overlay');
                    if (overlay) overlay.style.display = 'none';
                }
            """)
        except Exception as e:
            logger.error(f"Failed to hide overlay: {e}")
