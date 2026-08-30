from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://alb-8nsht2rhuxdg0zey3s.ap-southeast-1.alb.aliyuncsslbintl.com"

# hy3/gpt-5 live on the UniAPI gateway; other providers (added for cross-model
# comparison) live behind their own base_url/api_key and, for cc.ixg.be, need a
# browser-like User-Agent or Cloudflare's WAF blocks urllib's default UA (403/1010).
_UNIAPI_ROUTE = {
    "base_url_env": "UNIAPI_BASE_URL",
    "default_base_url": DEFAULT_BASE_URL,
    "api_key_env": "UNIAPI_KEY",
    "user_agent": None,
}
_CC_ROUTE = {
    "base_url_env": "CC_BASE_URL",
    "default_base_url": "https://cc.ixg.be/v1",
    "api_key_env": "CC_API_KEY",
    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}
MODEL_ROUTES = {
    "hy3": _UNIAPI_ROUTE,
    "gpt-5": _UNIAPI_ROUTE,
    # Also available directly on the UniAPI gateway (model names are case-sensitive
    # there, lowercase only) - no Cloudflare proxy, no 524s, much faster than cc.ixg.be.
    "glm-5.3": _UNIAPI_ROUTE,
    "glm-5.3-flash": _UNIAPI_ROUTE,
    "gemini-2.5-flash": _UNIAPI_ROUTE,
    # cc.ixg.be only: not available (or not confirmed available) on UniAPI.
    "GLM-5.3": _CC_ROUTE,
    "GLM-5.3-Flash": _CC_ROUTE,
    "Qwen3.8-Flash-Next": _CC_ROUTE,
    "hy4-preview": _CC_ROUTE,
}


class Hy3ClientError(RuntimeError):
    pass


def _route_for(model: str) -> dict:
    return MODEL_ROUTES.get(model, _UNIAPI_ROUTE)


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def call_hy3(
    messages: list[dict],
    *,
    model: str = "hy3",
    temperature: float = 0.2,
    max_tokens: int = 8192,
) -> str:
    route = _route_for(model)
    base_url = os.environ.get(route["base_url_env"], route["default_base_url"])
    url = _chat_completions_url(base_url)
    api_key = os.environ.get(route["api_key_env"])
    if not api_key:
        raise Hy3ClientError(
            f"{route['api_key_env']} is not set. export {route['api_key_env']}=<key> "
            f"before calling model '{model}'."
        )
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if route["user_agent"]:
        headers["User-Agent"] = route["user_agent"]
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise Hy3ClientError(f"{model} timed out waiting for a response from {url} after 300s") from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 401:
            raise Hy3ClientError(
                f"401 from {url}: key missing/invalid Authorization header. {detail}"
            ) from exc
        if exc.code == 403:
            raise Hy3ClientError(
                f"403 from {url}: model '{model}' not enabled for this key, or model name is "
                f"wrong (OpenAI protocol uses bare names like 'hy3', not 'anthropic-hy3'). {detail}"
            ) from exc
        raise Hy3ClientError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise Hy3ClientError(f"failed to reach {url}: {exc}") from exc
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        finish_reason = payload.get("choices", [{}])[0].get("finish_reason") if payload.get("choices") else None
        if finish_reason == "length":
            raise Hy3ClientError(
                f"{model} exhausted max_tokens={max_tokens} on reasoning before emitting an answer "
                f"(finish_reason=length, usage={payload.get('usage')}). Retry with a larger max_tokens."
            ) from exc
        raise Hy3ClientError(f"unexpected response shape from {url}: {payload}") from exc
    if not content:
        raise Hy3ClientError(
            f"{model} returned empty content (finish_reason={choice.get('finish_reason')}); "
            f"likely exhausted max_tokens={max_tokens} on reasoning."
        )
    return content
