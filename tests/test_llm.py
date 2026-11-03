import json

import httpx
import pytest

from tokenmind.llm import Client, LLMError

OK = {"choices": [{"message": {"content": "Lyon"}}], "usage": {"prompt_tokens": 42}}


def make_client(tmp_path, statuses, sent=None):
    statuses = iter(statuses)

    def handler(request):
        if sent is not None:
            sent.append(request)
        status = next(statuses)
        return httpx.Response(status, json=OK if status == 200 else {"error": "x"})

    return Client(
        base_url="https://llm.test/v1",
        api_key="secret",
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        sleep=lambda seconds: None,
    )


def test_sends_a_deterministic_chat_request(tmp_path):
    sent = []
    reply = make_client(tmp_path, [200], sent).chat("org/model", "Where?")
    assert (reply.text, reply.prompt_tokens) == ("Lyon", 42)
    request = sent[0]
    assert request.url == "https://llm.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["temperature"] == 0 and body["model"] == "org/model"


def test_cached_responses_are_not_requested_again(tmp_path):
    sent = []
    client = make_client(tmp_path, [200], sent)
    client.chat("org/model", "Where?")
    assert client.chat("org/model", "Where?").text == "Lyon"
    assert len(sent) == 1
    assert list(tmp_path.glob("org__model/*.json"))


def test_retries_rate_limits_then_succeeds(tmp_path):
    assert make_client(tmp_path, [429, 503, 200]).chat("m", "q").text == "Lyon"


def test_client_errors_are_raised(tmp_path):
    with pytest.raises(LLMError, match="400"):
        make_client(tmp_path, [400]).chat("m", "q")


def test_gives_up_after_the_retry_budget(tmp_path):
    with pytest.raises(LLMError, match="gave up"):
        make_client(tmp_path, [429] * 5).chat("m", "q")
