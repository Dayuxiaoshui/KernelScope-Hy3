from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://alb-8nsht2rhuxdg0zey3s.ap-southeast-1.alb.aliyuncsslbintl.com"


class Hy3ClientError(RuntimeError):
    pass


def _base_url() -> str:
    return os.environ.get("UNIAPI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> str:
    key = os.environ.get("UNIAPI_KEY")
    if not key:
        raise Hy3ClientError(
            "UNIAPI_KEY is not set. export UNIAPI_KEY=<your univ- key> before calling hy3."
        )
    return key


def call_hy3(
    messages: list[dict],
    *,
    model: str = "hy3",
    temperature: float = 0.2,
    max_tokens: int = 8192,
) -> str:
    url = f"{_base_url()}/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
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
