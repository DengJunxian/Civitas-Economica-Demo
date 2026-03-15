"""Government and policy compiler for structured policy shocks."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from core.macro.state import MacroState


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class PolicyShock:
    """Structured policy shock compiled from free-form policy text."""

    policy_id: str
    policy_text: str
    policy_rate_delta: float = 0.0
    fiscal_stimulus_delta: float = 0.0
    liquidity_injection: float = 0.0
    credit_spread_delta: float = 0.0
    stamp_tax_delta: float = 0.0
    sentiment_delta: float = 0.0
    rumor_shock: float = 0.0

    def to_dict(self) -> Dict[str, float | str]:
        return asdict(self)


class PolicyCompiler:
    """Compile policy text into structured and simulation-ready shocks."""

    _RE_NUMERIC = re.compile(r"(-?\d+(?:\.\d+)?)")

    def compile(self, policy_text: str, *, tick: int = 0) -> PolicyShock:
        text = str(policy_text or "").strip()
        lower = text.lower()
        shock = PolicyShock(policy_id=f"policy-{tick}-{uuid.uuid4().hex[:8]}", policy_text=text)

        if not text:
            return shock

        numeric_hint = self._extract_numeric_hint(text)

        if "印花税" in text and any(token in text for token in ("下调", "降低", "减免", "下调")):
            scale = 0.001 + 0.0002 * numeric_hint
            shock.stamp_tax_delta -= scale
            shock.liquidity_injection += 0.08 + 0.03 * numeric_hint
            shock.sentiment_delta += 0.08
            shock.credit_spread_delta -= 0.0015

        if any(token in text for token in ("流动性注入", "降准", "降息", "MLF", "再贷款")):
            scale = 0.08 + 0.04 * numeric_hint
            shock.liquidity_injection += scale
            shock.policy_rate_delta -= 0.001 * (1.0 + numeric_hint)
            shock.credit_spread_delta -= 0.002
            shock.sentiment_delta += 0.06

        if any(token in text for token in ("财政刺激", "基建", "补贴", "减税")):
            shock.fiscal_stimulus_delta += 0.05 + 0.02 * numeric_hint
            shock.sentiment_delta += 0.04

        if any(token in text for token in ("负面谣言", "谣言", "爆雷", "panic", "挤兑")):
            shock.rumor_shock -= 0.25 - 0.05 * min(numeric_hint, 1.0)
            shock.sentiment_delta -= 0.12
            shock.liquidity_injection -= 0.06
            shock.credit_spread_delta += 0.003

        if any(token in lower for token in ("rate hike", "hike", "tightening")) or "加息" in text:
            shock.policy_rate_delta += 0.0015 + 0.001 * numeric_hint
            shock.credit_spread_delta += 0.0025
            shock.sentiment_delta -= 0.05

        if all(
            getattr(shock, field) == 0.0
            for field in (
                "policy_rate_delta",
                "fiscal_stimulus_delta",
                "liquidity_injection",
                "credit_spread_delta",
                "stamp_tax_delta",
                "sentiment_delta",
                "rumor_shock",
            )
        ):
            # Neutral fallback: slight sentiment move to avoid dead-flat dynamics.
            shock.sentiment_delta = 0.01 if "支持" in text else -0.01 if "风险" in text else 0.0

        return shock

    def _extract_numeric_hint(self, text: str) -> float:
        match = self._RE_NUMERIC.search(text)
        if not match:
            return 0.0
        value = abs(float(match.group(1)))
        if value > 1000:
            return 1.0
        if value > 100:
            return 0.6
        if value > 10:
            return 0.3
        if value > 1:
            return 0.1
        return 0.0


class GovernmentAgent:
    """Government layer that compiles policy text and applies macro shocks."""

    def __init__(self, compiler: Optional[PolicyCompiler] = None):
        self.compiler = compiler or PolicyCompiler()

    def compile_policy_text(self, policy_text: str, *, tick: int = 0) -> PolicyShock:
        """Compile free-form policy text into a structured policy shock."""
        return self.compiler.compile(policy_text, tick=tick)

    def apply_policy_shock(self, macro_state: MacroState, shock: PolicyShock) -> MacroState:
        """Apply policy shock to macro state."""
        macro_state.apply_delta(
            policy_rate=shock.policy_rate_delta,
            fiscal_stimulus=shock.fiscal_stimulus_delta,
            liquidity_index=shock.liquidity_injection,
            credit_spread=shock.credit_spread_delta,
            sentiment_index=shock.sentiment_delta,
        )
        # Tax cuts work through liquidity/sentiment channels in this simplified model.
        macro_state.apply_delta(
            liquidity_index=-0.8 * shock.stamp_tax_delta,
            sentiment_index=-2.0 * shock.stamp_tax_delta,
        )
        if shock.rumor_shock != 0.0:
            macro_state.apply_delta(
                sentiment_index=shock.rumor_shock,
                liquidity_index=0.25 * shock.rumor_shock,
                unemployment=-0.06 * shock.rumor_shock,
            )
        macro_state.policy_rate = _clip(macro_state.policy_rate, 0.0, 0.20)
        macro_state.clamp_inplace()
        return macro_state


if __name__ == "__main__":
    gov = GovernmentAgent()
    macro = MacroState()
    for idx, text in enumerate(("印花税下调", "流动性注入 5000 亿", "突发负面谣言"), start=1):
        shock = gov.compile_policy_text(text, tick=idx)
        gov.apply_policy_shock(macro, shock)
        print(shock.to_dict())
        print(macro.to_dict())
