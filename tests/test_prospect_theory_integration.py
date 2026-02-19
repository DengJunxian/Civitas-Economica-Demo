# file: tests/test_prospect_theory_integration.py
"""
前景理论集成验证测试

验证以下模块的正确性:
1. utility.py - 前景理论计算
2. cognitive_agent.py - 心理效用值计算
3. brain.py - 机构/散户 Prompt 区分
4. network.py - 社交网络功能
5. trader_agent.py - 社交方法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0

def test_ok(name, msg=""):
    global passed
    passed += 1
    print(f"  [OK] {name} {msg}")

def test_fail(name, err):
    global failed
    failed += 1
    print(f"  [FAIL] {name}: {err}")

print("=" * 60)
print("前景理论 & 社交网络 集成验证测试")
print("=" * 60)

# ============ 1. utility.py ============
print("\n--- 1. utility.py (前景理论) ---")
try:
    from agents.cognition.utility import (
        ProspectTheory, ProspectValue, InvestorType,
        ConfidenceTracker, AnchorTracker,
        create_panic_retail, create_normal_investor, create_disciplined_quant
    )
    test_ok("Import utility.py")
except Exception as e:
    test_fail("Import utility.py", e)

try:
    pt = ProspectTheory(investor_type=InvestorType.PANIC_RETAIL)
    assert pt.lambda_coeff >= 3.0, f"PANIC_RETAIL lambda should be >= 3.0, got {pt.lambda_coeff}"
    test_ok("ProspectTheory init (PANIC_RETAIL)", f"lambda={pt.lambda_coeff}")
except Exception as e:
    test_fail("ProspectTheory init", e)

try:
    pt = ProspectTheory(investor_type=InvestorType.NORMAL)
    val_loss = pt.calculate_value(-0.08)
    val_gain = pt.calculate_value(0.08)
    assert val_loss < 0, f"Loss should be negative, got {val_loss}"
    assert val_gain > 0, f"Gain should be positive, got {val_gain}"
    assert abs(val_loss) > abs(val_gain), "Loss should hurt more than gain (loss aversion)"
    test_ok("calculate_value", f"loss={val_loss:.4f}, gain={val_gain:.4f}")
except Exception as e:
    test_fail("calculate_value", e)

try:
    pt = ProspectTheory(investor_type=InvestorType.NORMAL)
    full = pt.calculate_full(-0.08)
    assert isinstance(full, ProspectValue), f"Expected ProspectValue, got {type(full)}"
    assert full.subjective_value < 0, "Subjective value should be negative for loss"
    test_ok("calculate_full", f"value={full.subjective_value:.4f}, bias={full.decision_bias}")
except Exception as e:
    test_fail("calculate_full", e)

try:
    pt = ProspectTheory(investor_type=InvestorType.PANIC_RETAIL)
    action, reason = pt.should_override_decision("BUY", -0.15)
    assert action != "BUY" or reason is not None, "Should override or provide reason for panic retail buying at -15%"
    test_ok("should_override_decision", f"action={action}, reason={reason}")
except Exception as e:
    test_fail("should_override_decision", e)

try:
    tracker = ConfidenceTracker()
    tracker.update(-0.05)
    tracker.update(-0.03)
    desc = tracker.get_description()
    assert isinstance(desc, str), "Description should be string"
    test_ok("ConfidenceTracker", f"desc={desc[:30]}...")
except Exception as e:
    test_fail("ConfidenceTracker", e)

# ============ 2. cognitive_agent.py ============
print("\n--- 2. cognitive_agent.py (认知 Agent) ---")
try:
    from agents.cognition.cognitive_agent import CognitiveAgent
    test_ok("Import cognitive_agent.py")
except Exception as e:
    test_fail("Import cognitive_agent.py", e)

try:
    agent = CognitiveAgent(
        "test_agent", InvestorType.NORMAL,
        use_local_reasoner=True,
        reference_point=10.0,
        risk_aversion_lambda=2.5
    )
    assert agent.reference_point == 10.0, f"reference_point should be 10.0, got {agent.reference_point}"
    test_ok("CognitiveAgent init with new params", f"ref={agent.reference_point}")
except Exception as e:
    test_fail("CognitiveAgent init", e)

try:
    pv = agent.calculate_psychological_value(9.5)
    assert pv < 0, f"Price below ref should give negative value, got {pv}"
    pv2 = agent.calculate_psychological_value(10.5)
    assert pv2 > 0, f"Price above ref should give positive value, got {pv2}"
    test_ok("calculate_psychological_value", f"loss={pv:.4f}, gain={pv2:.4f}")
except Exception as e:
    test_fail("calculate_psychological_value", e)

try:
    pv3 = agent.calculate_psychological_value(10.0)
    assert pv3 == 0, f"Price at ref should give 0, got {pv3}"
    test_ok("psychological_value at reference point", f"value={pv3}")
except Exception as e:
    test_fail("psychological_value at reference point", e)

# ============ 3. brain.py (System Prompt) ============
print("\n--- 3. brain.py (System Prompt 区分) ---")
try:
    from agents.brain import DeepSeekBrain
    test_ok("Import brain.py")
except Exception as e:
    test_fail("Import brain.py", e)

try:
    brain_retail = DeepSeekBrain("retail_001", {
        "agent_type": "retail",
        "loss_aversion": 3.0,
        "risk_preference": "激进"
    })
    prompt_r = brain_retail._build_system_prompt()
    assert "个人投资者" in prompt_r or "散户" in prompt_r, "Retail prompt missing keywords"
    assert "前景理论" in prompt_r, "Retail prompt missing Prospect Theory"
    assert "心理效用值" in prompt_r, "Retail prompt missing psychological value"
    test_ok("Retail System Prompt", f"length={len(prompt_r)}")
except Exception as e:
    test_fail("Retail System Prompt", e)

try:
    brain_inst = DeepSeekBrain("inst_001", {
        "agent_type": "institution",
        "risk_preference": "稳健"
    })
    prompt_i = brain_inst._build_system_prompt()
    assert "机构投资者" in prompt_i, "Institution prompt missing keywords"
    assert "<reasoning>" in prompt_i, "Institution prompt missing CoT structure"
    assert "Chain-of-Thought" in prompt_i, "Institution prompt missing CoT"
    test_ok("Institution System Prompt", f"length={len(prompt_i)}")
except Exception as e:
    test_fail("Institution System Prompt", e)

# ============ 4. network.py ============
print("\n--- 4. network.py (社交网络功能) ---")
try:
    from core.society.network import SocialGraph, SentimentState
    test_ok("Import network.py")
except Exception as e:
    test_fail("Import network.py", e)

try:
    graph = SocialGraph(n_agents=20, k=4, p=0.3, seed=42)
    # 设置一些节点为 bullish
    graph.agents[1].sentiment_state = SentimentState.BULLISH
    graph.agents[2].sentiment_state = SentimentState.BULLISH
    br = graph.get_bullish_ratio(0)
    assert isinstance(br, float), f"Expected float, got {type(br)}"
    test_ok("get_bullish_ratio", f"ratio={br:.2f}")
except Exception as e:
    test_fail("get_bullish_ratio", e)

try:
    summary = graph.generate_social_summary(0)
    assert isinstance(summary, str), f"Expected str, got {type(summary)}"
    assert len(summary) > 0, "Summary should not be empty"
    test_ok("generate_social_summary", f"summary={summary[:40]}...")
except Exception as e:
    test_fail("generate_social_summary", e)

# ============ 5. trader_agent.py ============
print("\n--- 5. trader_agent.py (社交方法) ---")
try:
    from agents.trader_agent import TraderAgent
    test_ok("Import trader_agent.py")
except Exception as e:
    test_fail("Import trader_agent.py", e)

try:
    ta = TraderAgent("social_test_agent")
    opinion = ta.share_opinion()
    assert "agent_id" in opinion, "Missing agent_id in opinion"
    assert "sentiment" in opinion, "Missing sentiment in opinion"
    test_ok("share_opinion", f"opinion={opinion}")
except Exception as e:
    test_fail("share_opinion", e)

try:
    opinions = [
        {"sentiment": "bearish"}, {"sentiment": "bearish"},
        {"sentiment": "bearish"}, {"sentiment": "bullish"},
        {"sentiment": "neutral"}
    ]
    result = ta.receive_opinion(opinions)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 0, "Result should not be empty"
    test_ok("receive_opinion", f"result={result[:50]}...")
except Exception as e:
    test_fail("receive_opinion", e)

try:
    summary = ta.get_social_summary()
    assert isinstance(summary, str), f"Expected str, got {type(summary)}"
    test_ok("get_social_summary", f"summary={summary[:40]}...")
except Exception as e:
    test_fail("get_social_summary", e)

# ============ 总结 ============
print("\n" + "=" * 60)
print(f"测试结果: {passed} 通过, {failed} 失败")
print("=" * 60)

if failed > 0:
    sys.exit(1)
