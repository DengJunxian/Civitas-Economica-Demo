# Reference Demo Sync Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the reference demo's agent-to-market storytelling, simulated day-by-day auto-run behavior, and lighter policy-lab presentation polish into the current `Civitas_new` project without downgrading the existing `PolicySession + MarketEnvironment + SimulationRunner` architecture.

**Architecture:** Keep the current project's session-oriented simulation core as the source of truth. Re-express the reference project's "background auto-play" and "three-stage policy demo" as UI/session helpers around `ui/policy_lab.py`, `core/policy_session.py`, and `engine/simulation_loop.py`, instead of copying the reference project's monolithic `app.py` thread loop. Assume "自动每日运行" means simulated trading days auto-advance inside the policy experiment session, not an OS-level real-calendar cron job.

**Tech Stack:** Python 3.14, Streamlit, pandas, Plotly, Mesa-style agent simulation, custom IPC runner, pytest

---

### Task 1: Add Simulated Daily Auto-Run To Policy Sessions

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\ui\policy_lab.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\core\policy_session.py`
- Test: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_lab_session.py`

**Step 1: Write the failing test**

Add focused tests that describe the intended behavior:

```python
def test_policy_session_autoplay_advances_until_complete() -> None:
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="下调印花税并释放流动性。",
        policy_type="市场稳定",
        total_days=3,
        intensity=1.0,
        effective_day=1,
        half_life_days=10,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"
    session["autoplay"] = {"enabled": True, "step_days": 1, "last_wallclock_ts": 0.0}

    advanced = _policy_session_maybe_autoplay(session, now_ts=999.0, min_interval_seconds=0.0)

    assert advanced is True
    assert session["current_day"] == 1


def test_policy_session_autoplay_stops_when_finished() -> None:
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="下调印花税并释放流动性。",
        policy_type="市场稳定",
        total_days=1,
        intensity=1.0,
        effective_day=1,
        half_life_days=10,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"
    session["autoplay"] = {"enabled": True, "step_days": 1, "last_wallclock_ts": 0.0}

    _policy_session_maybe_autoplay(session, now_ts=999.0, min_interval_seconds=0.0)

    assert session["status"] == "completed"
    assert session["autoplay"]["enabled"] is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_policy_lab_session.py -k autoplay -v`

Expected: FAIL with missing helper or assertion mismatches because auto-run state is not implemented yet.

**Step 3: Write minimal implementation**

Implement session-level autoplay metadata and one helper that advances the session by simulated trading days:

```python
def _policy_session_autoplay_state(session: Dict[str, Any]) -> Dict[str, Any]:
    return session.setdefault(
        "autoplay",
        {"enabled": False, "step_days": 1, "last_wallclock_ts": 0.0},
    )


def _policy_session_maybe_autoplay(
    session: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    min_interval_seconds: float = 0.8,
) -> bool:
    autoplay = _policy_session_autoplay_state(session)
    if not autoplay.get("enabled"):
        return False
    if str(session.get("status", "")).lower() not in {"running", "paused"}:
        autoplay["enabled"] = False
        return False
    now_ts = float(now_ts if now_ts is not None else time.time())
    if now_ts - float(autoplay.get("last_wallclock_ts", 0.0)) < float(min_interval_seconds):
        return False
    autoplay["last_wallclock_ts"] = now_ts
    _policy_session_advance(session, int(autoplay.get("step_days", 1) or 1))
    if str(session.get("status", "")).lower() == "completed":
        autoplay["enabled"] = False
    return True
```

Wire it into the Streamlit control area so the user can toggle:
- `自动逐日运行`
- `每次推进天数`
- optional interval note

Also make the page rerun while autoplay is active.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_policy_lab_session.py -k autoplay -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ui/policy_lab.py core/policy_session.py tests/test_policy_lab_session.py
git commit -m "feat: add policy session autoplay"
```

### Task 2: Surface The Agent → Matching → Index Transmission Chain

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\engine\simulation_loop.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\core\policy_session.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\ui\policy_lab.py`
- Test: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_market_realism_pipeline.py`
- Test: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_lab_session.py`

**Step 1: Write the failing test**

Add one environment/session-level assertion that the latest step report exposes the same chain the reference demo emphasizes:

```python
def test_policy_session_snapshot_contains_transmission_chain() -> None:
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="提高流动性支持，稳定市场预期。",
        policy_type="市场稳定",
        total_days=5,
        intensity=1.0,
        effective_day=1,
        half_life_days=15,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"
    _policy_session_advance(session, 1)

    latest = session.get("latest_step_report", {})
    chain = latest.get("transmission_chain", {})

    assert "policy_signal" in chain
    assert "agent_sentiment" in chain
    assert "order_flow" in chain
    assert "matching_result" in chain
    assert "index_move" in chain
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_policy_lab_session.py -k transmission_chain -v`

Expected: FAIL because the current step report does not yet normalize these fields for the policy-lab UI.

**Step 3: Write minimal implementation**

Normalize and expose a compact chain payload in `engine/simulation_loop.py` and preserve it in the session snapshot:

```python
latest_report["transmission_chain"] = {
    "policy_signal": {
        "text": applied_policy_text,
        "strength": float(policy_signal_strength),
    },
    "agent_sentiment": {
        "social_mean": float(self._last_social_mean),
        "committee_enabled": bool(self.enable_policy_committee),
    },
    "order_flow": {
        "buy_volume": float(buy_volume),
        "sell_volume": float(sell_volume),
    },
    "matching_result": {
        "trade_count": int(match_payload.get("trade_count", 0)),
        "last_price": float(new_price),
    },
    "index_move": {
        "old_price": float(old_price),
        "new_price": float(new_price),
        "return_pct": float(price_change_pct),
    },
}
```

Mirror that into `session["latest_step_report"]` and a UI-friendly summary dict so policy-lab cards do not need to infer everything from raw fields.

**Step 4: Run test to verify it passes**

Run:
- `pytest tests/test_policy_lab_session.py -k transmission_chain -v`
- `pytest tests/test_policy_market_realism_pipeline.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add engine/simulation_loop.py core/policy_session.py ui/policy_lab.py tests/test_policy_lab_session.py tests/test_policy_market_realism_pipeline.py
git commit -m "feat: expose policy transmission chain in policy lab"
```

### Task 3: Refresh Policy Lab Presentation Using The Reference Demo's Stronger Storytelling

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\ui\policy_lab.py`
- Optional Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\app.py`
- Optional Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\theme\tokens.json`
- Test: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_lab_session.py`

**Step 1: Write the failing test**

Add helper-level tests for deterministic view-model builders instead of brittle Streamlit DOM tests:

```python
def test_policy_lab_demo_cards_cover_reference_three_stage_story() -> None:
    cards = _build_policy_demo_cards(
        policy_text="下调印花税并释放流动性。",
        latest_step_report={
            "transmission_chain": {
                "policy_signal": {"strength": 1.2},
                "agent_sentiment": {"social_mean": -0.1},
                "order_flow": {"buy_volume": 1200.0, "sell_volume": 900.0},
                "matching_result": {"trade_count": 18},
                "index_move": {"return_pct": 0.8},
            }
        },
    )

    assert [card["phase"] for card in cards] == ["政策注入", "情绪扩散", "撮合落地"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_policy_lab_session.py -k demo_cards -v`

Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Create view-model helpers and update layout in `ui/policy_lab.py`:
- add a top summary hero for current session status
- render three compact phase cards inspired by `ui/demo_wind_tunnel.py`
- show a small market tape / key metrics strip above the chart
- keep the current advanced controls, but move them below the primary story area

Example helper:

```python
def _build_policy_demo_cards(policy_text: str, latest_step_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    chain = dict(latest_step_report.get("transmission_chain", {}) or {})
    return [
        {"phase": "政策注入", "summary": policy_text[:60] or "等待政策输入"},
        {"phase": "情绪扩散", "summary": f"情绪均值 {chain.get('agent_sentiment', {}).get('social_mean', 0.0):.2f}"},
        {"phase": "撮合落地", "summary": f"成交 {chain.get('matching_result', {}).get('trade_count', 0)} 笔"},
    ]
```

Do not replace the current policy-lab feature set. This task is presentation polish only.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_policy_lab_session.py -k demo_cards -v`

Expected: PASS

**Step 5: Commit**

```bash
git add ui/policy_lab.py app.py theme/tokens.json tests/test_policy_lab_session.py
git commit -m "feat: refresh policy lab storytelling layout"
```

### Task 4: Regression Testing And Documentation

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\README.md`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\docs\user_manual.md`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_lab_session.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_simulation_runner.py`

**Step 1: Write the failing test**

Add a final regression that the session can autoplay and still preserve runner/session invariants:

```python
def test_policy_session_autoplay_keeps_timeline_and_summary_in_sync() -> None:
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="发布稳市政策。",
        policy_type="市场稳定",
        total_days=4,
        intensity=1.0,
        effective_day=1,
        half_life_days=20,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"
    session["autoplay"] = {"enabled": True, "step_days": 1, "last_wallclock_ts": 0.0}

    while _policy_session_maybe_autoplay(session, now_ts=999.0, min_interval_seconds=0.0):
        pass

    frame = pd.DataFrame(session["frame_rows"])
    assert session["current_day"] == 4
    assert len(frame) == 4
    assert session["summary"]["latest_close"] == frame.iloc[-1]["close"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_policy_lab_session.py -k sync -v`

Expected: FAIL until autoplay, session storage, and summary refresh are fully consistent.

**Step 3: Write minimal implementation**

Finish cleanup work:
- make session stop/reset disable autoplay cleanly
- ensure session summary refresh runs after autoplay and manual advance
- document the new control flow in README and user manual

Documentation notes to add:
- what "自动逐日运行" means
- that it is simulated business-day playback, not a real-world scheduled job
- how it coexists with `继续 1 天 / 继续 5 天 / 运行到结束`

**Step 4: Run test to verify it passes**

Run:
- `pytest tests/test_policy_lab_session.py -v`
- `pytest tests/test_simulation_runner.py -v`
- `pytest tests/test_policy_market_realism_pipeline.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/user_manual.md tests/test_policy_lab_session.py tests/test_simulation_runner.py
git commit -m "docs: explain policy lab autoplay behavior"
```
