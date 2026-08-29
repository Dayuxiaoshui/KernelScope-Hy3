import json
from unittest.mock import patch

import pytest

from kernelscope.providers.hy3_client import Hy3ClientError, call_hy3
from kernelscope.providers.hy3_prompt import build_messages, extract_json
from kernelscope.tasks import TASKS


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_call_hy3_sends_bearer_auth_and_parses_content(monkeypatch):
    monkeypatch.setenv("UNIAPI_KEY", "univ-test-key")
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

    with patch("urllib.request.urlopen", fake_urlopen):
        content = call_hy3([{"role": "user", "content": "hello"}], model="hy3")

    assert content == "hi"
    assert captured["headers"]["Authorization"] == "Bearer univ-test-key"
    assert captured["body"]["model"] == "hy3"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["url"].endswith("/v1/chat/completions")


def test_call_hy3_requires_api_key(monkeypatch):
    monkeypatch.delenv("UNIAPI_KEY", raising=False)
    with pytest.raises(Hy3ClientError, match="UNIAPI_KEY"):
        call_hy3([{"role": "user", "content": "hello"}])


def test_call_hy3_maps_403_to_helpful_message(monkeypatch):
    import urllib.error

    monkeypatch.setenv("UNIAPI_KEY", "univ-test-key")

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    with patch("urllib.request.urlopen", fake_urlopen):
        with pytest.raises(Hy3ClientError, match="not enabled"):
            call_hy3([{"role": "user", "content": "hello"}])


def test_call_hy3_reports_truncation_when_reasoning_exhausts_budget(monkeypatch):
    monkeypatch.setenv("UNIAPI_KEY", "univ-test-key")

    def fake_urlopen(request, timeout=None):
        body = {
            "choices": [{"message": {"role": "assistant", "reasoning_content": "..."}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 4096, "reasoning_tokens": 4096},
        }
        return _FakeResponse(json.dumps(body).encode())

    with patch("urllib.request.urlopen", fake_urlopen):
        with pytest.raises(Hy3ClientError, match="exhausted max_tokens"):
            call_hy3([{"role": "user", "content": "hello"}], max_tokens=4096)


def test_extract_json_strips_markdown_fence():
    text = '```json\n{"a": 1}\n```'
    assert extract_json(text) == {"a": 1}


def test_extract_json_handles_raw_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_reports_snippet_on_failure():
    with pytest.raises(ValueError, match="not valid JSON"):
        extract_json("not json at all")


def test_build_messages_includes_task_and_signature():
    task = TASKS["rmsnorm"]
    messages = build_messages(task)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert task.task_id in messages[1]["content"]
    assert "def candidate(x: torch.Tensor, weight: torch.Tensor)" in messages[1]["content"]
