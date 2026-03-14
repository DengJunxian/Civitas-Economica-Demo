from core.llm_client import _try_fix_json, safe_json_loads


def test_try_fix_json_strips_think_and_closes():
    raw = "<think>hidden</think>{\"a\": 1, \"b\": 2"
    fixed = _try_fix_json(raw)
    assert fixed.endswith("}")
    data = safe_json_loads(raw)
    assert data["a"] == 1
    assert data["b"] == 2


def test_safe_json_loads_removes_control_chars():
    raw = "{\"x\": 1, \"y\": 2}\x00"
    data = safe_json_loads(raw)
    assert data["x"] == 1
    assert data["y"] == 2
