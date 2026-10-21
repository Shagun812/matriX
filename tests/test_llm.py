import json

import pytest

from matrix.llm import (
    CachedLLM,
    EmptyResponseError,
    LLMClient,
    LLMError,
    extract_json,
    parse_distribution,
)
from matrix.schemas import HYPOTHESES


class FakeClient(LLMClient):
    def __init__(self, responses):
        self.model = "fake-model"
        self.params = {"temperature": 0}
        self.responses = list(responses)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def valid_distribution():
    return json.dumps({
        "distribution": {h: 0.25 for h in HYPOTHESES}
    })


def validate_distribution(data):
    return parse_distribution(data["distribution"])


def test_extract_json_accepts_fenced_json():
    result = extract_json('```json\n{"ok": true}\n```')
    assert result == {"ok": True}


def test_extract_json_rejects_missing_object():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("There is no JSON here.")


def test_parse_distribution_normalizes_small_sum_error():
    distribution = {h: 0.251 for h in HYPOTHESES}
    result = parse_distribution(distribution)

    assert result.sum() == pytest.approx(1.0)


def test_parse_distribution_rejects_wrong_keys():
    with pytest.raises(ValueError, match="keys"):
        parse_distribution({"wrong": 1.0})


def test_parse_distribution_rejects_negative_probability():
    distribution = {h: 0.25 for h in HYPOTHESES}
    distribution[HYPOTHESES[0]] = -0.1

    with pytest.raises(ValueError, match="negative"):
        parse_distribution(distribution)


def test_cache_prevents_duplicate_api_call(tmp_path):
    client = FakeClient([valid_distribution()])
    llm = CachedLLM(client, tmp_path, "test-v1")

    first = llm.complete("system", "user", sample_idx=0)
    second = llm.complete("system", "user", sample_idx=0)

    assert first == second
    assert client.calls == 1
    assert llm.stats["calls"] == 1
    assert llm.stats["cache_hits"] == 1


def test_cache_key_changes_with_prompt_and_sample(tmp_path):
    client = FakeClient([
        '{"a": 1}',
        '{"a": 2}',
        '{"a": 3}',
    ])
    llm = CachedLLM(client, tmp_path, "test-v1")

    a = llm.complete("system", "user", sample_idx=0)
    b = llm.complete("system", "different user", sample_idx=0)
    c = llm.complete("system", "user", sample_idx=1)

    assert [a, b, c] == ['{"a": 1}', '{"a": 2}', '{"a": 3}']
    assert client.calls == 3


def test_api_errors_are_not_cached(tmp_path):
    client = FakeClient([
        RuntimeError("temporary failure"),
        valid_distribution(),
    ])
    llm = CachedLLM(client, tmp_path, "test-v1")

    with pytest.raises(LLMError):
        llm.complete("system", "user")

    result = llm.complete("system", "user")

    assert json.loads(result)["distribution"]
    assert client.calls == 2
    assert llm.stats["api_errors"] == 1


def test_empty_response_is_counted_separately_from_api_errors(tmp_path):
    client = FakeClient([EmptyResponseError("empty response")])
    llm = CachedLLM(client, tmp_path, "test-v1")

    with pytest.raises(EmptyResponseError):
        llm.complete("system", "user")

    assert llm.stats["empty_responses"] == 1
    assert llm.stats["api_errors"] == 0


def test_get_json_retries_invalid_output_once(tmp_path):
    client = FakeClient([
        "not json",
        valid_distribution(),
    ])
    llm = CachedLLM(client, tmp_path, "test-v1")

    result, meta = llm.get_json(
        "system", "user", validate_distribution, retries=1
    )

    assert result is not None
    assert result.sum() == pytest.approx(1.0)
    assert meta["attempts"] == 2
    assert len(meta["errors"]) == 1
    assert llm.stats["invalid_outputs"] == 1


def test_get_json_returns_none_after_retry_failure(tmp_path):
    client = FakeClient(["bad output", "still bad"])
    llm = CachedLLM(client, tmp_path, "test-v1")

    result, meta = llm.get_json(
        "system", "user", validate_distribution, retries=1
    )

    assert result is None
    assert meta["attempts"] == 2
    assert client.calls == 2


def test_get_json_handles_api_error_without_raising(tmp_path):
    client = FakeClient([RuntimeError("offline")])
    llm = CachedLLM(client, tmp_path, "test-v1")

    result, meta = llm.get_json(
        "system", "user", validate_distribution, retries=0
    )

    assert result is None
    assert meta["attempts"] == 1
    assert "api:" in meta["errors"][0]


def test_get_json_handles_wrapped_gemini_api_error(tmp_path):
    client = FakeClient([LLMError("Gemini unavailable")])
    llm = CachedLLM(client, tmp_path, "test-v1")

    result, meta = llm.get_json(
        "system", "user", validate_distribution, retries=0
    )

    assert result is None
    assert meta["attempts"] == 1
    assert meta["errors"] == ["api: Gemini unavailable"]
    assert llm.stats["api_errors"] == 1


def test_cache_key_changes_when_model_parameters_change(tmp_path):
    client = FakeClient(['{"value": 1}', '{"value": 2}'])
    llm = CachedLLM(client, tmp_path, "test-v1")

    first = llm.complete("system", "user")
    client.params = {"temperature": 0.5}
    second = llm.complete("system", "user")

    assert first != second
    assert client.calls == 2
