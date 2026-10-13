"""
LLM Client Adapter - Универсальный адаптер для LLM.

Транспорты выбираются по реестру agent/providers.py (kind провайдера):
  • openai — OpenAI-совместимый /chat/completions (OpenAI, Kimi, DeepSeek,
             Groq, Mistral, OpenRouter, Gemini, xAI, LM Studio, LocalAI …)
  • ollama — нативный /api/chat локального Ollama
  • pplx   — Perplexity через cookies-сессию (curl + SSE)

Новый вендор подключается записью в PROVIDERS (или register_provider), без
правок в этом файле.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid as _uuid
from typing import Optional, List

from agent.providers import PROVIDERS, ProviderSpec, get_provider

logger = logging.getLogger(__name__)

# ─── Perplexity credentials ────────────────────────────────────────────────────────────────
# Пир-доступ к Perplexity работает через cookies авторизованной сессии.
# Ничего из этого не хранится в репозитории: значения приходят из
# переменных окружения или из локального файла .pplx_cookies.txt
# (перечислен в .gitignore). Без них провайдер pplx просто не будет
# выбран — остальные (kimi/ollama/openai) работают без cookies.

PPLX_ACCOUNT_ID = os.environ.get("PPLX_ACCOUNT_ID", "")

# Cookie-файл искорем в самом проекте, а не по относительному пути наружу:
# так клон репозитория остаётся самодостаточным.
_COOKIE_FILE = os.environ.get(
    "PPLX_COOKIES_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".pplx_cookies.txt"),
)
try:
    with open(_COOKIE_FILE, "r") as _f:
        PPLX_COOKIES = _f.read().strip()
except OSError:
    PPLX_COOKIES = ""

# Идентификаторы запроса/сессии — одноразовые, генерируются на каждый запуск.
# Захардкоженные значения позволили бы ассоциировать запросы с конкретным
# аккаунтом, а заодно утекли бы в публичный репозиторий.
REQUEST_UUID = os.environ.get("PPLX_REQUEST_UUID") or str(_uuid.uuid4())
RUM_SESSION_ID = os.environ.get("PPLX_RUM_SESSION_ID") or str(_uuid.uuid4())
PPLX_CONTEXT_UUID = os.environ.get("PPLX_CONTEXT_UUID") or str(_uuid.uuid4())

# Список моделей для каждого провайдера собирается из реестра providers.PROVIDERS,
# а не дублируется здесь. Формат ({id,name,best?}) сохранён для совместимости со
# старым кодом; источник правды теперь один — agent/providers.py.
MODELS = {key: [m.to_dict() for m in spec.models] for key, spec in PROVIDERS.items()}


class LLMClient:
    """
    Адаптер для LLM через прямые API-вызовы.

    Транспорт выбирается по spec.kind из реестра providers.PROVIDERS, а не по
    имени провайдера, поэтому подключение нового вендора не требует правок
    здесь. Поддерживаются:
      • kind="pplx"   — Perplexity через cookies (curl + SSE)
      • kind="ollama" — нативный /api/chat локального Ollama
      • kind="openai" — OpenAI-совместимый /chat/completions (OpenAI, Kimi,
                        DeepSeek, Groq, Mistral, OpenRouter, Gemini, xAI, …)
    """

    def __init__(self, provider: str = "pplx", model: str = ""):
        self.provider = provider
        self.spec: Optional[ProviderSpec] = get_provider(provider)
        if self.spec is None:
            logger.warning(f"Неизвестный провайдер '{provider}' — транспорт не определён")
        # Модель: явная → дефолт реестра. Сохраняем прежнее поведение по умолчанию.
        self.model = model or (self.spec.default_model_id() if self.spec else "") or "claude47opus"

    def accepts_vision(self) -> bool:
        """Поддерживает ли текущая модель изображения.

        Нужно планировщику заранее, до вызова: цикл ретраев «не вижу картинку»
        бессилен, когда у модели глаз нет вовсе (qwen3:8b в Ollama), и лишь
        сжигает минуты на паузах.
        """
        if self.spec is None or not self.spec.vision:
            return False
        if self.spec.kind == "ollama":
            # Ollama принимает images только для VL-моделей.
            return any(kw in self.model.lower() for kw in ("vl", "vision", "moondream"))
        return True

    async def chat(self, message: str) -> str:
        """Отправить сообщение и получить ответ."""
        kind = self.spec.kind if self.spec else ""
        if kind == "pplx":
            return await self._chat_pplx(message)
        if kind == "ollama":
            return await self._chat_ollama(message)
        if kind == "openai":
            return await self._chat_openai_compat(message)
        raise ValueError(f"Unknown provider: {self.provider}")

    async def chat_with_image(self, message: str, image_b64: str,
                              filename: str = "image.png",
                              mime: str = "image/png") -> str:
        """Сообщение + изображение (vision).

        Транспорт vision зависит от kind:
          • pplx   — file_view-attachment (PNG надёжнее JPEG/WebP);
          • openai — content-part {type:image_url, data: URL};
          • ollama — поле images:[base64].
        Провайдер без vision (spec.vision=False) деградирует в текстовый chat.
        """
        if self.spec is None or not self.spec.vision or not image_b64:
            return await self.chat(message)
        kind = self.spec.kind
        if kind == "pplx":
            return await self._chat_pplx(message, image_b64=image_b64,
                                         image_filename=filename, image_mime=mime)
        if kind == "openai":
            return await self._chat_openai_compat(message, image_b64=image_b64, image_mime=mime)
        if kind == "ollama":
            # Ollama vision работает только для VL-моделей (qwen3-vl, llama3.2-vision и т.п.)
            # qwen3:8b, llama3:8b и другие текстовые модели вернут 400 при images.
            m_lower = self.model.lower()
            if not any(kw in m_lower for kw in ("vl", "vision", "moondream")):
                logger.info(f"Модель {self.model} не vision — деградация к текстовому chat")
                return await self.chat(message)
            return await self._chat_ollama(message, image_b64=image_b64)
        # fallback
        logger.info(f"kind={kind} не поддерживает vision — деградация к текстовому chat")
        return await self.chat(message)

    async def _chat_pplx(self, message: str, image_b64: str = "",
                         image_filename: str = "screenshot.jpg",
                         image_mime: str = "image/jpeg") -> str:
        """Вызов Perplexity AI через cookies и curl (как pplx_caller.py).
        image_b64 — чистый base64 (без data:-префикса); пустая строка = текстовый запрос."""
        try:
            model_pref = self.model or "turbo"

            attachments = []
            if image_b64:
                attachments = [{
                    "file_view": {
                        "kind": "image",
                        "mime_type": image_mime,
                        "base64": image_b64,
                    },
                    "filename": image_filename,
                }]

            payload = {
                "params": {
                    "attachments": attachments,
                    "language": "ru-RU",
                    "timezone": "Europe/Kaliningrad",
                    "search_focus": "internet",
                    "sources": ["web"],
                    "frontend_uuid": REQUEST_UUID,
                    "mode": "copilot",
                    "model_preference": model_pref,
                    "is_related_query": False,
                    "is_sponsored": False,
                    "frontend_context_uuid": PPLX_CONTEXT_UUID,
                    "prompt_source": "user",
                    "query_source": "home",
                    "is_incognito": False,
                    "local_search_enabled": True,
                    "use_schematized_api": True,
                    "send_back_text_in_streaming_api": False,
                    "supported_block_use_cases": [
                        "answer_modes", "media_items", "knowledge_cards",
                        "inline_entity_cards", "place_widgets", "finance_widgets",
                        "sports_widgets", "news_widgets", "shopping_widgets",
                        "jobs_widgets", "search_result_widgets", "inline_images",
                        "inline_assets", "placeholder_cards", "diff_blocks",
                        "inline_knowledge_cards", "entity_group_v2",
                        "refinement_filters", "canvas_mode", "maps_preview",
                        "answer_tabs", "preserve_latex", "in_context_suggestions",
                        "pending_followups", "inline_claims", "unified_assets",
                        "workflow_steps", "workflow_widgets", "navigation_results",
                        "background_agents",
                    ],
                    "client_coordinates": None,
                    "mentions": [],
                    "dsl_query": message,
                    "skip_search_enabled": True,
                    "is_nav_suggestions_disabled": False,
                    "source": "entropy",
                    "always_search_override": False,
                    "override_no_search": False,
                    "comet_info": {"rendering_place": "tab"},
                    "client_search_results_cache_key": f"nav-{REQUEST_UUID}",
                    "should_ask_for_mcp_tool_confirmation": True,
                    "supports_tool_approval_modal": True,
                    "browser_agent_allow_once_from_toggle": False,
                    "force_enable_browser_agent": False,
                    "supported_features": ["browser_agent_permission_banner_v1.1"],
                    "extended_context": False,
                    "local_workspace_directories": [],
                    "version": "2.18",
                    "rum_session_id": RUM_SESSION_ID,
                },
                "query_str": message,
            }

            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
                json.dump(payload, f)
                payload_file = f.name

            headers = [
                "-H", "accept: text/event-stream",
                "-H", "accept-language: ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "-H", "cache-control: no-cache",
                "-H", "content-type: application/json",
                "-H", "origin: https://www.perplexity.ai",
                "-H", "pragma: no-cache",
                "-H", "priority: u=1, i",
                "-H", "referer: https://www.perplexity.ai/?erp=new_tab",
                "-H", 'sec-ch-ua: "Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"',
                "-H", 'sec-ch-ua-arch: "arm"',
                "-H", 'sec-ch-ua-bitness: "64"',
                "-H", 'sec-ch-ua-full-version: "151.0.7922.47"',
                '-H', 'sec-ch-ua-full-version-list: "Not=A?Brand";v="99.0.0.0", "Google Chrome";v="151.0.7922.47", "Chromium";v="151.0.7922.47"',
                "-H", "sec-ch-ua-mobile: ?0",
                '-H', 'sec-ch-ua-model: ""',
                "-H", 'sec-ch-ua-platform: "macOS"',
                '-H', 'sec-ch-ua-platform-version: "26.2.0"',
                "-H", "sec-fetch-dest: empty",
                "-H", "sec-fetch-mode: cors",
                "-H", "sec-fetch-site: same-origin",
                "-H", "user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
                "-H", f"x-pplx-account: {PPLX_ACCOUNT_ID}",
                "-H", "x-perplexity-request-endpoint: https://www.perplexity.ai/rest/sse/perplexity_ask",
                "-H", "x-perplexity-request-reason: ask-query-state-provider",
                "-H", "x-perplexity-request-try-number: 1",
                "-H", f"x-request-id: {REQUEST_UUID}",
                "-b", PPLX_COOKIES,
            ]

            cmd = [
                "curl", "--http2", "-s", "-N", "-w", "\nHTTP_CODE:%{http_code}",
                "-X", "POST",
                "https://www.perplexity.ai/rest/sse/perplexity_ask",
                "--data-binary", f"@{payload_file}",
            ] + headers

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw = proc.stdout.read()
            proc.wait()

            text = raw.decode("utf-8", errors="ignore")

            # Cloudflare/рейт-лимит: 4xx отдаёт HTML-заглушку вместо SSE.
            # Раньше это «проглатывалось» и выглядело как пустой ответ LLM —
            # честно сигнализируем ошибку, чтобы сработал retry-механизм.
            http_line = [l for l in text.splitlines() if l.startswith("HTTP_CODE:")]
            if not text.strip().startswith("event:") and (http_line or "<html" in text.lower()[:200]):
                code = http_line[0].split(":", 1)[1].strip() if http_line else "?"
                logger.error(f"Perplexity HTTP {code} (rate limit / CF block)")
                return json.dumps(
                    {"type": "wait", "thought": f"Pplx error: HTTP {code}", "seconds": 15},
                    ensure_ascii=False,
                )

            # SSE: собираем data-блоки (JSON может быть разбит на несколько строк)
            data_blocks = []
            current_block = ""
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped.startswith("data:"):
                    current_block += stripped[5:]
                elif stripped == "event:" or stripped == "":
                    if current_block.strip():
                        data_blocks.append(current_block.strip())
                        current_block = ""

            full_text = ""
            for json_str in data_blocks:
                try:
                    event = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                is_final = event.get("final_sse_message") or event.get("final")
                status = event.get("status", "")

                for block in event.get("blocks", []):
                    if "workflow_block" in block:
                        wb = block["workflow_block"]
                        if is_final and status == "COMPLETED":
                            for step in wb.get("steps", []):
                                for item in step.get("items", []):
                                    if item.get("type") == "WORKFLOW_ITEM_TEXT":
                                        tp = item.get("payload", {}).get("text_payload", {})
                                        full_text = tp.get("text", "")
                                        if full_text:
                                            os.unlink(payload_file)
                                            return full_text
                        elif not is_final:
                            for step in wb.get("steps", []):
                                for item in step.get("items", []):
                                    if item.get("type") == "WORKFLOW_ITEM_TEXT":
                                        tp = item.get("payload", {}).get("text_payload", {})
                                        chunks = tp.get("chunks", [])
                                        if chunks:
                                            text_val = "".join(chunks)
                                            if text_val and len(text_val) > 5:
                                                full_text = text_val
                    # Ответ также приходит в markdown_block (answer при DONE,
                    # chunks при стриминге) — у запросов с attachments ответ
                    # стабильно идёт этим путём.
                    if "markdown_block" in block:
                        mb = block["markdown_block"]
                        ans = mb.get("answer", "")
                        if ans:
                            full_text = ans
                        else:
                            chunks = mb.get("chunks", [])
                            if chunks:
                                text_val = "".join(chunks)
                                if text_val and len(text_val) > 5:
                                    full_text = text_val

            os.unlink(payload_file)
            return full_text if full_text else '{"type": "wait", "thought": "Perplexity вернул пустой ответ", "seconds": 1}'

        except Exception as e:
            logger.error(f"Pplx error: {e}")
            return json.dumps({"type": "wait", "thought": f"Pplx error: {e}", "seconds": 1}, ensure_ascii=False)

    def _resolve_api_key(self) -> str:
        """Ключ из переменной окружения, названной в spec.api_key_env.
        Для локальных серверов (requires_key=False) отсутствие ключа — норма."""
        env = self.spec.api_key_env if self.spec else ""
        return os.environ.get(env, "") if env else ""

    async def _chat_openai_compat(self, message: str, image_b64: str = "",
                                  image_mime: str = "image/png") -> str:
        """Универсальный транспорт OpenAI-совместимых API.

        Один метод обслуживает OpenAI, Kimi, DeepSeek, Groq, Mistral,
        OpenRouter, Gemini (compat-слой), xAI, LM Studio, LocalAI и любого
        нового вендора с тем же контрактом — различия только в base_url,
        имени ключа и названии модели (всё берётся из ProviderSpec).
        """
        label = self.spec.label if self.spec else self.provider
        try:
            import httpx

            api_key = self._resolve_api_key()
            if not api_key and self.spec and self.spec.requires_key:
                env = self.spec.api_key_env or "API_KEY"
                logger.warning(f"{env} not set. Using fallback.")
                return json.dumps(
                    {"type": "wait", "thought": f"{label}: не задан {env}", "seconds": 1},
                    ensure_ascii=False,
                )

            if image_b64:
                content: object = [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image_mime};base64,{image_b64}"}},
                ]
            else:
                content = message

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    self.spec.chat_url,
                    headers=headers,
                    json={
                        "model": self.model or self.spec.default_model_id(),
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 2048,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

        except ImportError:
            logger.error("httpx not installed.")
            return '{"type": "wait", "thought": "httpx не установлен", "seconds": 1}'
        except Exception as e:
            logger.error(f"{label} error: {e}")
            return json.dumps({"type": "wait", "thought": f"{label} error: {e}", "seconds": 1},
                              ensure_ascii=False)

    async def _chat_ollama(self, message: str, image_b64: str = "") -> str:
        """Нативный транспорт Ollama (/api/chat).

        Ключ не нужен: сервер локальный. Vision передаётся полем images:[base64]
        (без data:-префикса) — его принимают qwen3-vl, llama3.2-vision, gemma3.
        """
        try:
            import httpx

            payload = {
                "model": self.model or self.spec.default_model_id(),
                "messages": [{"role": "user", "content": message}],
                "stream": False,
                "format": "json",
                # qwen3/deepseek-r1 — «думающие» модели: без этого генерируют
                # тысячи thinking-токенов вместо JSON и парсер спотыкается.
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                    "think": False,
                },
            }
            if image_b64:
                payload["messages"][0]["images"] = [image_b64]

            async with httpx.AsyncClient(timeout=120) as client:
                import time as _t
                t0 = _t.monotonic()
                resp = await client.post(self.spec.chat_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                # Тайминги Ollama: prompt_eval — разбор промпта, eval — генерация.
                # Без них непонятно, что тормозит шаг агента: огромный DOM-снимок
                # или длинный ответ модели.
                logger.info(
                    "Ollama %s: %.1fs total, prompt=%s tok, answer=%s tok",
                    self.model, _t.monotonic() - t0,
                    data.get("prompt_eval_count"), data.get("eval_count"))
                return data.get("message", {}).get("content", "")

        except ImportError:
            logger.error("httpx not installed.")
            return '{"type": "wait", "thought": "httpx не установлен", "seconds": 1}'
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return json.dumps({"type": "wait", "thought": f"Ollama error: {e}", "seconds": 1},
                              ensure_ascii=False)