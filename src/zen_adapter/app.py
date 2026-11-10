"""FastAPI facade over OpenCode Zen free-tier models.

Emulates the OpenCode client identity required by upstream's free-tier gate:

* ``User-Agent: opencode/<ENV OPENCODE_VERSION>``
* ``x-opencode-session`` / ``x-opencode-session-id`` with a freshly minted
  "descending" id per upstream attempt (``~(ms*0x1000+counter)`` masked to 48
  bits, rendered as 12 hex chars, then 14 base62 chars).
* Free models are only served from ``https://opencode.ai/inference/v1``.
* Every upstream request declares a ``read`` tool with a non-empty property
  schema; caller tools are preserved and the placeholder is only appended.

The adapter does NOT mask quota exhaustion: upstream ``FreeUsageLimitError``
is surfaced verbatim with its status code so callers can back off correctly.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=os.environ.get("ZEN_LOG_LEVEL", "INFO"))
logger = logging.getLogger("zen_adapter")

UPSTREAM = os.environ.get("ZEN_UPSTREAM_URL", "https://opencode.ai/inference/v1")
UPSTREAM_TIMEOUT = float(os.environ.get("ZEN_UPSTREAM_TIMEOUT_S", "180"))
CLIENT_VERSION = os.environ.get("OPENCODE_VERSION", "1.18.35")
DEFAULT_MODEL = os.environ.get("ZEN_DEFAULT_MODEL", "beam-free")

BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_LAST_ID_MS = 0
_ID_COUNTER = 0


def descending_session_id() -> str:
    """OpenCode's descending identifier: 12-hex inverted timestamp + 14 base62."""
    global _LAST_ID_MS, _ID_COUNTER
    now = int(time.time() * 1000)
    if now != _LAST_ID_MS:
        _LAST_ID_MS = now
        _ID_COUNTER = 0
    _ID_COUNTER += 1
    value = ~(now * 0x1000 + _ID_COUNTER) & 0xFFFFFFFFFFFF
    suffix = "".join(secrets.choice(BASE62) for _ in range(14))
    return f"ses_{value:012x}{suffix}"


def _api_keys() -> list[str]:
    raw = os.environ.get("ZEN_API_KEY") or os.environ.get("OPENCODE_API_KEY", "")
    return [key.strip() for key in raw.split(",") if key.strip()]


def _adapter_api_key() -> str:
    return os.environ.get("ADAPTER_API_KEY", "")


def _check_auth(request: Request) -> None:
    expected = _adapter_api_key()
    if not expected:
        return
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
    if supplied != expected and request.headers.get("x-api-key") != expected:
        raise HTTPException(status_code=401, detail="invalid adapter API key")


def _upstream_headers(api_key: str) -> dict[str, str]:
    session = descending_session_id()
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"opencode/{CLIENT_VERSION}",
        "x-opencode-client": "cli",
        "x-opencode-session": session,
        "x-opencode-session-id": session,
    }


_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Unavailable in this request. Never call this tool.",
        "parameters": {
            "type": "object",
            "properties": {"filePath": {"type": "string"}},
            "required": ["filePath"],
            "additionalProperties": False,
        },
    },
}


def _declared_tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
        elif isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


def _prepare_body(body: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(body)
    prepared.setdefault("model", DEFAULT_MODEL)
    tools = prepared.get("tools")
    if "read" not in _declared_tool_names(tools):
        prepared["tools"] = list(tools or []) + [_READ_TOOL]
    return prepared


def _tune_headers(headers: dict[str, str], body: dict[str, Any]) -> dict[str, str]:
    """Drop Content-Length after body mutation so httpx recomputes it."""
    headers.pop("Content-Length", None)
    headers.pop("content-length", None)
    if body.get("stream"):
        headers["Accept"] = "text/event-stream"
    return headers


app = FastAPI(title="zen-adapter", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    keys = _api_keys()
    return {
        "status": "ok" if keys else "no-api-key",
        "upstream": UPSTREAM,
        "keys": len(keys),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/v1/models")
@app.get("/models")
async def models(request: Request) -> JSONResponse:
    _check_auth(request)
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        try:
            resp = await client.get(f"{UPSTREAM}/models")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
    payload = resp.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return JSONResponse(payload)
    free = [
        row
        for row in rows
        if isinstance(row, dict)
        and (
            str(row.get("id", "")).endswith("-free")
            or row.get("id") == "big-pickle"
        )
    ]
    return JSONResponse({"object": "list", "data": free})


async def _stream_relay(resp: httpx.Response, client: httpx.AsyncClient):
    try:
        async for chunk in resp.aiter_raw():
            yield chunk
    finally:
        await resp.aclose()
        await client.aclose()


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    _check_auth(request)
    keys = _api_keys()
    if not keys:
        raise HTTPException(status_code=500, detail="ZEN_API_KEY is not configured")

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    if not body.get("messages"):
        raise HTTPException(status_code=400, detail="messages[] is required")

    prepared = _prepare_body(body)
    want_stream = bool(prepared.get("stream"))
    url = f"{UPSTREAM}/chat/completions"

    last_detail = "no attempt"
    for attempt, key in enumerate(keys):
        headers = _tune_headers(_upstream_headers(key), prepared)
        client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
        try:
            req = client.build_request("POST", url, json=prepared, headers=headers)
            resp = await client.send(req, stream=want_stream)
        except httpx.HTTPError as exc:
            await client.aclose()
            last_detail = f"upstream transport error: {exc}"
            logger.warning("attempt %d transport error: %s", attempt, exc)
            continue

        if resp.status_code >= 400:
            detail = (await resp.aread()).decode("utf-8", errors="replace")
            await resp.aclose()
            await client.aclose()
            last_detail = f"upstream {resp.status_code}: {detail[:500]}"
            logger.warning("attempt %d upstream %s: %s", attempt, resp.status_code, detail[:200])
            if resp.status_code in (401, 429) and attempt < len(keys) - 1:
                continue
            headers_out = {}
            retry_after = resp.headers.get("retry-after")
            if retry_after:
                headers_out["Retry-After"] = retry_after
            return JSONResponse(
                status_code=resp.status_code,
                content={"error": {"type": "upstream_error", "detail": last_detail}},
                headers=headers_out,
            )

        if want_stream:
            media_type = resp.headers.get("content-type", "text/event-stream")
            return StreamingResponse(
                _stream_relay(resp, client),
                media_type=media_type,
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            data = await resp.aread()
        finally:
            await resp.aclose()
            await client.aclose()
        return JSONResponse(content=json.loads(data) if data else {})

    raise HTTPException(status_code=502, detail=last_detail)
