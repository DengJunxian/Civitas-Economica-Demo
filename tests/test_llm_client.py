import json

from core.llm_client import _try_fix_json


def test_try_fix_json_removes_think_and_code_fence():
    raw = """<think>内部推理</think>
```json
{\n  \"action\": \"BUY\",\n  \"quantity\": 100\n}
```
"""
    fixed = _try_fix_json(raw)
    data = json.loads(fixed)
    assert data["action"] == "BUY"
    assert data["quantity"] == 100


def test_try_fix_json_auto_closes_truncated_object_and_array():
    raw = '{"action":"SELL","legs":[{"qty":10}'
    fixed = _try_fix_json(raw)
    data = json.loads(fixed)
    assert data["action"] == "SELL"
    assert data["legs"][0]["qty"] == 10


def test_try_fix_json_removes_control_chars():
    raw = '{"action":"HOLD",\x00"quantity":0}'
    fixed = _try_fix_json(raw)
    data = json.loads(fixed)
    assert data["action"] == "HOLD"
    assert data["quantity"] == 0
