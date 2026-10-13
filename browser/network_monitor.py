"""
Network Monitor - программное чтение сетевой активности вкладки через CDP.
Аналог вкладки "Сеть" в DevTools, но без открытия самого окна DevTools
(не конфликтует с Playwright-сессией агента).
"""

import asyncio
import json
import logging
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CDP_HOST = "127.0.0.1"
CDP_PORT = 9222


def _http_get(path: str) -> dict:
    with urllib.request.urlopen(f"http://{CDP_HOST}:{CDP_PORT}{path}", timeout=5) as r:
        return json.load(r)


def get_page_targets() -> List[dict]:
    """Все обычные вкладки (не devtools://, не service worker, не browser_ui)."""
    targets = _http_get("/json")
    return [t for t in targets
            if t.get("type") == "page" and not t["url"].startswith("devtools://")]


async def capture_network(
    target_id: Optional[str] = None,
    reload: bool = True,
    duration: float = 8.0,
    include_bodies: bool = False,
    body_limit: int = 2000,
) -> dict:
    """
    Собрать сетевую активность вкладки (аналог вкладки "Сеть" в DevTools).

    target_id: конкретная вкладка (None = первая обычная)
    reload: перезагрузить страницу перед сбором (как F5)
    duration: сколько секунд собирать события
    include_bodies: подтянуть тела ответов (для XHR/Fetch)
    """
    import websockets

    targets = get_page_targets()
    if not targets:
        raise RuntimeError("Нет открытых вкладок")

    if target_id:
        target = next((t for t in targets if t["id"] == target_id), targets[0])
    else:
        target = targets[0]

    browser_ws = _http_get("/json/version")["webSocketDebuggerUrl"]

    requests: Dict[str, dict] = {}
    responses: Dict[str, dict] = {}

    async with websockets.connect(browser_ws, max_size=10**7) as ws:
        msg_id = 0

        async def send(method, params=None, session_id=None):
            nonlocal msg_id
            msg_id += 1
            msg = {"id": msg_id, "method": method, "params": params or {}}
            if session_id:
                msg["sessionId"] = session_id
            await ws.send(json.dumps(msg))
            return msg_id

        async def wait_response(rid, timeout=5.0):
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                resp = json.loads(raw)
                if resp.get("id") == rid:
                    return resp

        # Прицепляемся к вкладке
        rid = await send("Target.attachToTarget",
                         {"targetId": target["id"], "flatten": True})
        resp = await wait_response(rid)
        if "result" not in resp:
            raise RuntimeError(f"Не удалось прицепиться к вкладке: {resp}")
        session_id = resp["result"]["sessionId"]

        # Включаем перехват сети
        rid = await send("Network.enable", session_id=session_id)
        await wait_response(rid)

        if reload:
            await send("Page.reload", session_id=session_id)

        # Собираем события
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=duration)
                msg = json.loads(raw)
                m, params = msg.get("method"), msg.get("params", {})

                if m == "Network.requestWillBeSent":
                    r = params["request"]
                    requests[params["requestId"]] = {
                        "method": r["method"],
                        "url": r["url"],
                        "type": params.get("type", "?"),
                    }
                elif m == "Network.responseReceived":
                    r = params["response"]
                    responses[params["requestId"]] = {
                        "status": r.get("status"),
                        "mime": r.get("mimeType", ""),
                        "size": r.get("encodedDataLength", 0),
                    }
        except asyncio.TimeoutError:
            pass

        # Тела ответов для API-запросов
        bodies = {}
        if include_bodies:
            for rid_, req in requests.items():
                if req["type"] not in ("XHR", "Fetch"):
                    continue
                try:
                    rid = await send("Network.getResponseBody",
                                     {"requestId": rid_}, session_id=session_id)
                    resp = await wait_response(rid, timeout=3.0)
                    if "result" in resp:
                        body = resp["result"].get("body", "")
                        bodies[req["url"]] = body[:body_limit]
                except Exception:
                    pass

    # Формируем итог
    entries = []
    for rid_, req in requests.items():
        r = responses.get(rid_, {})
        entries.append({
            "method": req["method"],
            "url": req["url"],
            "type": req["type"],
            "status": r.get("status"),
            "size": r.get("size", 0),
            "body": bodies.get(req["url"]),
        })

    js = [e for e in entries if e["url"].endswith(".js") or e["type"] == "Script"]
    api = [e for e in entries if e["type"] in ("XHR", "Fetch")]

    return {
        "page": {"title": target["title"], "url": target["url"]},
        "total": len(entries),
        "js_count": len(js),
        "api_count": len(api),
        "entries": entries,
    }


def format_network_report(data: dict) -> str:
    """Человекочитаемый отчёт — как вкладка "Сеть" в DevTools."""
    lines = []
    lines.append(f"Сетевая активность: {data['page']['title']}")
    lines.append(f"URL: {data['page']['url']}")
    lines.append("")
    lines.append(f"{'МЕТОД':<7}{'СТАТУС':<8}{'ТИП':<12}URL")
    lines.append("-" * 100)
    for e in data["entries"]:
        status = str(e["status"]) if e["status"] else "..."
        lines.append(f"{e['method']:<7}{status:<8}{e['type']:<12}{e['url'][:70]}")
    lines.append("")
    lines.append(f"Итого: {data['total']} запросов, "
                 f"JS-файлов: {data['js_count']}, "
                 f"API-запросов (XHR/Fetch): {data['api_count']}")
    # Тела API-ответов, если есть
    for e in data["entries"]:
        if e.get("body"):
            lines.append("")
            lines.append(f"--- Тело ответа {e['method']} {e['url'][:60]} ---")
            lines.append(e["body"])
    return "\n".join(lines)
