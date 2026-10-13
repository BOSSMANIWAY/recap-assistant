"""
Реестр LLM-провайдеров.

Каждый вендор описывается одной записью ProviderSpec — сам клиент
(LLMClient) не содержит ветвлений по имени провайдера. Чтобы подключить
новый вендор, достаточно добавить запись в PROVIDERS (или вызвать
register_provider из конфигурации) — код цикла агента, планировщик и
веб-интерфейс менять не нужно: они читают список из этого модуля.

kind определяет транспорт:
  • "openai"  — OpenAI-совместимый /chat/completions (Bearer + JSON).
                Покрывает OpenAI, Kimi, DeepSeek, Groq, Mistral, OpenRouter,
                Gemini (через compat-слой), xAI, LocalAI, LM Studio и др.
  • "ollama"  — нативный /api/chat локального Ollama-сервера (без ключа).
  • "pplx"    — особенный: Perplexity через cookies-сессию (curl + SSE).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    """Одна модель провайдера для выпадающего списка в UI."""
    id: str
    name: str
    best: bool = False

    def to_dict(self) -> dict:
        d = {"id": self.id, "name": self.name}
        if self.best:
            d["best"] = True
        return d


@dataclass
class ProviderSpec:
    key: str                       # машинный id ("openai"); передаётся в LLMClient
    label: str                     # человекочитаемое имя для UI
    kind: str                      # "openai" | "ollama" | "pplx"
    base_url: str = ""             # для kind=openai — корень API без /chat/completions
    api_key_env: str = ""          # имя переменной окружения с ключом
    default_model: str = ""        # модель по умолчанию
    models: List[ModelInfo] = field(default_factory=list)
    vision: bool = False           # принимает ли image-attachment в chat
    requires_key: bool = True      # False для локальных серверов (Ollama, LM Studio)
    docs_url: str = ""             # где взять ключ
    note: str = ""                 # подсказка в UI (например «нужен файл cookies»)

    @property
    def base_url_env(self) -> str:
        """Имя переменной окружения, которой можно переопределить base_url."""
        return f"{self.key.upper()}_BASE_URL"

    @property
    def effective_base_url(self) -> str:
        """base_url реестра, переопределённый окружением (если задан).

        Нужно, чтобы эндпоинт менялся без правки кода: Kimi по умолчанию
        смотрит на api.moonshot.cn (Китай), а международный ключ живёт на
        api.moonshot.ai; самохостed-прокси и корпоративные шлюзы отдают тот
        же OpenAI-контракт на своём адресе. Задаётся одной переменной:
            export KIMI_BASE_URL=https://api.moonshot.ai/v1
        """
        return os.environ.get(self.base_url_env, "").strip() or self.base_url

    @property
    def chat_url(self) -> str:
        """Полный URL эндпоинта чата для данного транспорта."""
        if self.kind == "ollama":
            base = self.effective_base_url or "http://localhost:11434"
            return f"{base.rstrip('/')}/api/chat"
        return f"{self.effective_base_url.rstrip('/')}/chat/completions"

    def default_model_id(self) -> str:
        if self.default_model:
            return self.default_model
        for m in self.models:
            if m.best:
                return m.id
        return self.models[0].id if self.models else ""

    def to_public_dict(self) -> dict:
        """Сериализация для /api/providers (без каких-либо секретов)."""
        # has_key — сам ключ не отдаём, только факт его наличия в окружении.
        if not self.requires_key:
            has_key: Optional[bool] = True
        elif self.api_key_env:
            has_key = bool(os.environ.get(self.api_key_env, ""))
        else:
            has_key = None  # pplx: креды в cookies-файле, проверить дёшево нельзя
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "default_model": self.default_model_id(),
            "vision": self.vision,
            "requires_key": self.requires_key,
            "key_env": self.api_key_env,
            "has_key": has_key,
            "base_url_env": self.base_url_env if self.kind != "pplx" else "",
            # Публичный адрес эндпоинта (не секрет): UI показывает, куда реально
            # пойдёт запрос с учётом переопределения из окружения.
            "base_url": "" if self.kind == "pplx" else self.effective_base_url,
            "docs_url": self.docs_url,
            "note": self.note,
            "models": [m.to_dict() for m in self.models],
        }


# ─── Реестр ────────────────────────────────────────────────────────────────────
# Записей много, но они однотипны. Формат моделей: ModelInfo(id, name, best?)

def _m(*entries) -> List[ModelInfo]:
    out = []
    for e in entries:
        if isinstance(e, tuple):
            out.append(ModelInfo(e[0], e[1], len(e) > 2 and e[2]))
        else:
            out.append(ModelInfo(e, e))
    return out


PROVIDERS: Dict[str, ProviderSpec] = {
    "pplx": ProviderSpec(
        key="pplx",
        label="Perplexity AI",
        kind="pplx",
        default_model="claude47opus",
        vision=True,
        requires_key=True,
        docs_url="https://www.perplexity.ai",
        note="Cookies авторизованной сессии в .pplx_cookies.txt (см. README).",
        models=_m(
            ("claude47opus", "Claude 4.7 Opus", True),
            ("claude46sonnet", "Claude 4.6 Sonnet"),
            ("gpt55", "GPT-5.5"),
            ("gpt54", "GPT-5.4"),
            ("grok4", "Grok 4"),
            ("claudecode", "Claude Code"),
            ("codex4", "Codex 4"),
            ("gemini30flash", "Gemini 3.0 Flash"),
            ("o3pro", "o3 Pro"),
            ("turbo", "Turbo"),
        ),
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI (GPT)",
        kind="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-5.1",
        vision=True,
        docs_url="https://platform.openai.com/api-keys",
        models=_m(
            ("gpt-5.1", "GPT-5.1", True),
            ("gpt-5", "GPT-5"),
            ("gpt-4o", "GPT-4o"),
            ("o3", "o3"),
            ("o4-mini", "o4 Mini"),
        ),
    ),
    "kimi": ProviderSpec(
        key="kimi",
        label="Kimi (Moonshot)",
        kind="openai",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
        default_model="kimi-k2-0905-preview",
        vision=True,
        docs_url="https://platform.moonshot.cn/console/api-keys",
        models=_m(
            ("kimi-k2-0905-preview", "Kimi K2 (0905)", True),
            ("kimi-k2-turbo-preview", "Kimi K2 Turbo"),
            ("moonshot-v1-8k", "Moonshot v1 8K"),
            ("moonshot-v1-32k", "Moonshot v1 32K"),
            ("moonshot-v1-128k", "Moonshot v1 128K"),
        ),
    ),
    "ollama": ProviderSpec(
        key="ollama",
        label="Ollama (локально)",
        kind="ollama",
        base_url="http://localhost:11434",
        default_model="qwen3:32b",
        requires_key=False,
        vision=True,
        docs_url="https://ollama.com/library",
        note="Локальный сервер: ollama serve. Модель с vision (qwen3-vl, llama3.2-vision) даст агенту «зрение».",
        models=_m(
            ("qwen3:32b", "Qwen3 32B", True),
            ("qwen3:235b-a22b", "Qwen3 235B"),
            ("qwen3:8b", "Qwen3 8B"),
            ("llama3.3:70b", "Llama 3.3 70B"),
            ("deepseek-r1:70b", "DeepSeek R1 70B"),
        ),
    ),
    "deepseek": ProviderSpec(
        key="deepseek",
        label="DeepSeek",
        kind="openai",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        docs_url="https://platform.deepseek.com/api_keys",
        models=_m(
            ("deepseek-chat", "DeepSeek Chat (V3)", True),
            ("deepseek-reasoner", "DeepSeek Reasoner (R1)"),
        ),
    ),
    "groq": ProviderSpec(
        key="groq",
        label="Groq",
        kind="openai",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        docs_url="https://console.groq.com/keys",
        models=_m(
            ("llama-3.3-70b-versatile", "Llama 3.3 70B", True),
            ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout"),
            ("deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill"),
        ),
    ),
    "mistral": ProviderSpec(
        key="mistral",
        label="Mistral AI",
        kind="openai",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        default_model="mistral-large-latest",
        vision=True,
        docs_url="https://console.mistral.ai/api-keys",
        models=_m(
            ("mistral-large-latest", "Mistral Large", True),
            ("mistral-small-latest", "Mistral Small"),
            ("pixtral-12b-2409", "Pixtral 12B (vision)"),
        ),
    ),
    "openrouter": ProviderSpec(
        key="openrouter",
        label="OpenRouter (агрегатор)",
        kind="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="anthropic/claude-opus-4.1",
        vision=True,
        docs_url="https://openrouter.ai/keys",
        note="Один ключ — сотни моделей от разных вендоров.",
        models=_m(
            ("anthropic/claude-opus-4.1", "Claude Opus 4.1", True),
            ("openai/gpt-5", "GPT-5"),
            ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
            ("deepseek/deepseek-chat", "DeepSeek Chat"),
            ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
        ),
    ),
    "gemini": ProviderSpec(
        key="gemini",
        label="Google Gemini",
        kind="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.5-pro",
        vision=True,
        docs_url="https://aistudio.google.com/apikey",
        models=_m(
            ("gemini-2.5-pro", "Gemini 2.5 Pro", True),
            ("gemini-2.5-flash", "Gemini 2.5 Flash"),
            ("gemini-2.0-flash", "Gemini 2.0 Flash"),
        ),
    ),
    "xai": ProviderSpec(
        key="xai",
        label="xAI (Grok)",
        kind="openai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        default_model="grok-4",
        vision=True,
        docs_url="https://console.x.ai",
        models=_m(
            ("grok-4", "Grok 4", True),
            ("grok-3", "Grok 3"),
        ),
    ),
    "lmstudio": ProviderSpec(
        key="lmstudio",
        label="LM Studio (локально)",
        kind="openai",
        base_url="http://localhost:1234/v1",
        api_key_env="LMSTUDIO_API_KEY",
        default_model="local-model",
        requires_key=False,
        vision=True,
        docs_url="https://lmstudio.ai",
        note="Локальный сервер: включите «Server» в LM Studio, ключ не нужен.",
        models=_m(("local-model", "Загруженная в Studio модель", True)),
    ),
    "localai": ProviderSpec(
        key="localai",
        label="LocalAI (локально)",
        kind="openai",
        base_url="http://localhost:8080/v1",
        api_key_env="LOCALAI_API_KEY",
        default_model="qwen3:32b",
        requires_key=False,
        vision=True,
        docs_url="https://localai.io",
        note="Docker-образ localai/localai. Ключ не обязателен.",
        models=_m(("qwen3:32b", "qwen3:32b", True)),
    ),
}


# Провайдер по умолчанию. Единственная причина, по которой он здесь задан, —
# обратная совместимость с уже написанным кодом и тестами, ожидающими pplx.
DEFAULT_PROVIDER = "pplx"


def get_provider(key: str) -> Optional[ProviderSpec]:
    return PROVIDERS.get(key)


def default_model(key: str = DEFAULT_PROVIDER) -> str:
    """Рекомендованная модель провайдера из реестра.

    Использовать вместо литералов вроде "claude47opus": при смене вендора
    захардкоженное имя модели улетело бы запросом в несуществующий id.
    """
    spec = PROVIDERS.get(key)
    return spec.default_model_id() if spec else ""


def register_provider(spec: ProviderSpec) -> None:
    """Подключить вендора в рантайме (например из env/конфига) без правки кода."""
    PROVIDERS[spec.key] = spec


def list_providers() -> List[dict]:
    """Публичное описание всех провайдеров для UI/CLI (секретов нет)."""
    return [p.to_public_dict() for p in PROVIDERS.values()]
