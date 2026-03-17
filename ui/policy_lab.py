"""Policy lab page focused on government-facing policy experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from core.macro.government import GovernmentAgent, PolicyShock
from ui import dashboard as dashboard_ui


POLICY_TYPE_OPTIONS = {
    "税收调整": "tax",
    "流动性投放": "liquidity",
    "财政刺激": "fiscal",
    "监管收紧": "tightening",
    "市场稳定措施": "stabilization",
    "自定义政策": "custom",
}

POLICY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "降低印花税并释放流动性": {
        "policy_type": "税收调整",
        "title": "降低交易摩擦",
        "text": "阶段性下调证券交易印花税，并同步释放流动性支持，稳定市场预期。",
    },
    "财政扩张支持制造业": {
        "policy_type": "财政刺激",
        "title": "制造业投资刺激",
        "text": "加大财政刺激和专项补贴力度，支持制造业扩产与设备更新。",
    },
    "监管表态稳定市场": {
        "policy_type": "市场稳定措施",
        "title": "稳定市场信心",
        "text": "监管部门推出稳定市场组合拳，鼓励长期资金入市并加强做市支持。",
    },
    "风险谣言冲击测试": {
        "policy_type": "自定义政策",
        "title": "谣言冲击压测",
        "text": "突发负面谣言扩散，市场担忧企业信用风险并引发恐慌抛售。",
    },
}


@dataclass
class PolicyNarrativeCard:
    title: str
    summary: str
    bullets: List[str]
    tone: str = "neutral"


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _compile_scaled_shock(policy_text: str, intensity: float) -> PolicyShock:
    gov = GovernmentAgent()
    shock = gov.compile_policy_text(policy_text, tick=1)
    _apply_local_policy_keywords(policy_text, shock)
    scale = float(intensity)
    shock.policy_rate_delta *= scale
    shock.fiscal_stimulus_delta *= scale
    shock.liquidity_injection *= scale
    shock.credit_spread_delta *= scale
    shock.stamp_tax_delta *= scale
    shock.sentiment_delta *= scale
    shock.rumor_shock *= scale
    return shock


def _apply_local_policy_keywords(policy_text: str, shock: PolicyShock) -> None:
    text = str(policy_text or "")
    lower = text.lower()

    def _contains(*tokens: str) -> bool:
        return any(token in text or token in lower for token in tokens)

    if _contains("印花税", "减税", "税率下调", "lower tax", "cut tax"):
        shock.stamp_tax_delta -= 0.0005
        shock.liquidity_injection += 0.06
        shock.sentiment_delta += 0.05

    if _contains("流动性", "注入", "降准", "降息", "liquidity", "rrr cut", "rate cut"):
        shock.liquidity_injection += 0.08
        shock.policy_rate_delta -= 0.001
        shock.sentiment_delta += 0.05

    if _contains("财政", "补贴", "刺激", "基建", "fiscal", "subsidy", "stimulus"):
        shock.fiscal_stimulus_delta += 0.05
        shock.sentiment_delta += 0.04

    if _contains("稳定市场", "托底", "平准", "长期资金入市", "stabilize", "backstop"):
        shock.liquidity_injection += 0.04
        shock.sentiment_delta += 0.06
        shock.credit_spread_delta -= 0.001

    if _contains("谣言", "风险", "恐慌", "挤兑", "panic", "rumor", "selloff"):
        shock.rumor_shock -= 0.16
        shock.sentiment_delta -= 0.08
        shock.credit_spread_delta += 0.0015

    if _contains("加息", "收紧", "tighten", "tightening"):
        shock.policy_rate_delta += 0.0012
        shock.sentiment_delta -= 0.06
        shock.credit_spread_delta += 0.0015


def _shock_score(shock: PolicyShock) -> float:
    return (
        shock.liquidity_injection * 1.3
        + shock.fiscal_stimulus_delta * 1.5
        - shock.policy_rate_delta * 60.0
        - shock.credit_spread_delta * 18.0
        - shock.stamp_tax_delta * 420.0
        + shock.sentiment_delta * 1.2
        + shock.rumor_shock * 1.6
    )


def _generate_policy_metrics(
    *,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
) -> pd.DataFrame:
    shock = _compile_scaled_shock(policy_text, intensity)
    score = _shock_score(shock)
    rng = np.random.default_rng(_seed_from_text(f"{policy_text}|{intensity}|{duration_days}|{rumor_noise}"))

    periods = max(10, int(duration_days))
    dates = pd.bdate_range(pd.Timestamp.today().normalize(), periods=periods)
    base_price = 3000.0
    price = base_price
    rows: List[Dict[str, float | int | str]] = []

    baseline_drift = 0.0004
    support_drift = np.clip(score * 0.0032, -0.008, 0.008)
    base_vol = 0.0035 + 0.005 * min(1.0, abs(shock.sentiment_delta) + abs(shock.rumor_shock))
    panic_floor = 0.18 + 0.18 * max(0.0, -shock.sentiment_delta)

    for idx, dt in enumerate(dates, start=1):
        decay = np.exp(-(idx - 1) / max(periods * 0.55, 1.0))
        rumor_pulse = 0.0
        if rumor_noise:
            rumor_pulse = shock.rumor_shock * 0.010 * np.exp(-(idx - 1) / max(periods * 0.25, 1.0))
        drift = baseline_drift + support_drift * decay + rumor_pulse
        noise = rng.normal(0.0, base_vol) * (0.85 + 0.25 * np.sin(idx / 3.0))
        ret = drift + noise

        prev_price = price
        price = max(1600.0, prev_price * (1.0 + ret))
        intraday_noise = abs(noise) * 0.8 + 0.0012
        high = max(price, prev_price) * (1.0 + intraday_noise)
        low = min(price, prev_price) * (1.0 - intraday_noise)

        panic = np.clip(
            panic_floor
            + max(0.0, -support_drift) * 9.0
            + max(0.0, rumor_pulse) * 8.0
            + max(0.0, -ret) * 11.0
            - max(0.0, support_drift) * 4.0
            + rng.normal(0.0, 0.015),
            0.08,
            0.86,
        )
        csad = np.clip(0.05 + 0.11 * panic + abs(ret) * 5.5, 0.04, 0.21)
        volume = 1_150_000 * (
            1.0
            + 0.5 * abs(score)
            + 0.22 * idx / periods
            + 0.35 * panic
            + max(0.0, shock.liquidity_injection) * 0.7
        )

        rows.append(
            {
                "step": idx,
                "time": dt.strftime("%Y-%m-%d"),
                "open": round(prev_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": round(volume, 2),
                "csad": round(float(csad), 4),
                "panic_level": round(float(panic), 4),
            }
        )

    return pd.DataFrame(rows)


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _build_policy_cards(
    policy_title: str,
    policy_type: str,
    shock: PolicyShock,
    metrics: pd.DataFrame,
) -> List[PolicyNarrativeCard]:
    final_row = metrics.iloc[-1]
    start_close = float(metrics.iloc[0]["close"])
    end_close = float(final_row["close"])
    change = end_close / max(start_close, 1e-9) - 1.0
    avg_panic = float(metrics["panic_level"].mean())
    avg_csad = float(metrics["csad"].mean())
    score = _shock_score(shock)

    news_tone = "利好" if score >= 0 else "偏空"
    quant_view = "上行偏强" if change > 0.02 else "弱势承压" if change < -0.02 else "区间震荡"
    risk_view = "风险可控" if avg_panic < 0.32 else "需重点盯防" if avg_panic < 0.52 else "高压预警"
    manager_action = "分批加仓" if change > 0.03 else "保持观望" if abs(change) < 0.015 else "先防守再观察"

    return [
        PolicyNarrativeCard(
            title="新闻分析师",
            summary=f"{policy_title}在消息层面被市场解读为“{news_tone}”政策。",
            bullets=[
                f"政策类型：{policy_type}",
                f"情绪冲击：{shock.sentiment_delta:+.2f}，谣言扰动：{shock.rumor_shock:+.2f}",
                "传播速度会先影响成交情绪，再影响指数方向。",
            ],
            tone="good" if score >= 0 else "warn",
        ),
        PolicyNarrativeCard(
            title="量化分析师",
            summary=f"仿真结果显示市场大概率处于“{quant_view}”状态。",
            bullets=[
                f"区间累计变化：{_format_percent(change)}",
                f"平均羊群度：{avg_csad:.3f}",
                f"流动性注入强度：{shock.liquidity_injection:.2f}",
            ],
            tone="good" if change >= 0 else "warn",
        ),
        PolicyNarrativeCard(
            title="风险分析师",
            summary=f"当前市场处于“{risk_view}”区间。",
            bullets=[
                f"平均风险热度：{avg_panic:.2f}",
                f"信用利差冲击：{shock.credit_spread_delta:+.3f}",
                "若舆情再次放大，波动会先于价格扩张。",
            ],
            tone="bad" if avg_panic > 0.52 else "warn" if avg_panic > 0.32 else "good",
        ),
        PolicyNarrativeCard(
            title="经理最终决策",
            summary=f"建议采用“{manager_action}”而不是一次性重仓。",
            bullets=[
                "先观察政策落地后的流动性改善是否持续。",
                "若风险热度重新升高，优先保护流动性缓冲。",
                "对外汇报时可强调：政策先改变情绪，再改变价格路径。",
            ],
            tone="good" if change >= 0 else "warn",
        ),
    ]


def _render_policy_cards(cards: List[PolicyNarrativeCard]) -> None:
    cols = st.columns(len(cards))
    tone_map = {
        "good": "#1d4d39",
        "warn": "#5b4213",
        "bad": "#5b1d24",
        "neutral": "#16243b",
    }
    for idx, card in enumerate(cards):
        bullets = "".join(f"<li>{item}</li>" for item in card.bullets)
        with cols[idx]:
            st.markdown(
                f"""
                <div class="story-card" style="border-top:3px solid {tone_map.get(card.tone, '#16243b')};">
                  <div class="story-card-title">{card.title}</div>
                  <div class="story-card-summary">{card.summary}</div>
                  <ul class="story-card-list">{bullets}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_flow_strip() -> None:
    steps = ["政策输入", "宏观传导", "分析师讨论", "经理决策", "市场反馈"]
    cols = st.columns(len(steps))
    for idx, label in enumerate(steps):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="flow-card">
                  <div class="flow-step-index">0{idx + 1}</div>
                  <div class="flow-step-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_policy_header() -> None:
    st.markdown(
        """
        <div class="hero-panel">
          <div class="hero-kicker">Policy Sandbox</div>
          <h1>政府政策试验台</h1>
          <p>输入一项政策，用多智能体视角观察市场、情绪与风险如何联动变化。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_policy_lab() -> None:
    _render_policy_header()

    if "policy_lab_result" not in st.session_state:
        st.session_state.policy_lab_result = None

    selected_template = st.selectbox("载入示例政策", options=["不使用示例", *POLICY_TEMPLATES.keys()])
    template = POLICY_TEMPLATES.get(selected_template) if selected_template != "不使用示例" else None

    with st.form("policy_lab_form"):
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            policy_title = st.text_input("政策名称", value=template["title"] if template else "市场稳定政策测试")
            policy_text = st.text_area(
                "政策说明",
                value=template["text"] if template else "请描述要测试的政策，例如：下调印花税并释放流动性支持。",
                height=120,
            )
        with col2:
            policy_type = st.selectbox(
                "政策类型",
                options=list(POLICY_TYPE_OPTIONS.keys()),
                index=list(POLICY_TYPE_OPTIONS.keys()).index(template["policy_type"]) if template else 0,
            )
            intensity = st.slider("冲击强度", min_value=0.2, max_value=1.6, value=1.0, step=0.1)
            duration_days = st.slider("仿真时长（交易日）", min_value=12, max_value=60, value=30, step=6)
            rumor_noise = st.toggle("加入舆情扰动", value=("谣言" in policy_text or "风险" in policy_text))
        submitted = st.form_submit_button("开始仿真", use_container_width=True)

    if submitted:
        shock = _compile_scaled_shock(policy_text, intensity)
        metrics = _generate_policy_metrics(
            policy_text=policy_text,
            intensity=intensity,
            duration_days=duration_days,
            rumor_noise=rumor_noise,
        )
        cards = _build_policy_cards(policy_title, policy_type, shock, metrics)
        st.session_state.policy_lab_result = {
            "policy_title": policy_title,
            "policy_text": policy_text,
            "policy_type": policy_type,
            "intensity": intensity,
            "duration_days": duration_days,
            "rumor_noise": rumor_noise,
            "shock": shock.to_dict(),
            "metrics": metrics,
            "cards": cards,
        }

    result = st.session_state.policy_lab_result
    if not result:
        st.info("先输入一项政策，再点击“开始仿真”。系统会优先展示仿真大盘走势，而不是技术细节。")
        return

    metrics = result["metrics"]
    current_step = int(metrics.iloc[-1]["step"])
    shock = result["shock"]

    summary_col, side_col = st.columns([1.8, 1.0])
    with summary_col:
        st.markdown("### 仿真大盘走势")
        kpi = dashboard_ui.build_kpi_snapshot(metrics, current_step, regulation_hint="政策观察期")
        dashboard_ui.render_kpi_cards(kpi)
        dashboard_ui.render_market_overview(metrics, current_step, key_prefix="policy_lab")
    with side_col:
        st.markdown("### 政策摘要")
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">政策名称</div>
              <div class="summary-value">{result['policy_title']}</div>
              <div class="summary-label">政策类型</div>
              <div class="summary-note">{result['policy_type']}</div>
              <div class="summary-label">政策说明</div>
              <div class="summary-note">{result['policy_text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">结构化冲击</div>
              <div class="summary-note">流动性注入：{shock['liquidity_injection']:+.2f}</div>
              <div class="summary-note">财政刺激：{shock['fiscal_stimulus_delta']:+.2f}</div>
              <div class="summary-note">情绪变化：{shock['sentiment_delta']:+.2f}</div>
              <div class="summary-note">谣言冲击：{shock['rumor_shock']:+.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### AI 如何解读这项政策")
    _render_policy_cards(result["cards"])

    st.markdown("### 多智能体协作流程")
    _render_flow_strip()

    detail_left, detail_right = st.columns(2)
    with detail_left:
        dashboard_ui.render_policy_transmission_chain(
            {
                "policy": result["policy_text"],
                "macro_variables": {
                    "inflation": 0.021,
                    "unemployment": 0.048 - shock["fiscal_stimulus_delta"] * 0.05,
                    "wage_growth": 0.028 + shock["fiscal_stimulus_delta"] * 0.04,
                    "credit_spread": 0.015 + shock["credit_spread_delta"],
                    "liquidity_index": 1.0 + shock["liquidity_injection"],
                    "policy_rate": 0.022 + shock["policy_rate_delta"],
                    "fiscal_stimulus": shock["fiscal_stimulus_delta"],
                    "sentiment_index": 0.55 + shock["sentiment_delta"] + shock["rumor_shock"],
                },
                "social_sentiment": {"mean": 0.35 + shock["sentiment_delta"] + shock["rumor_shock"]},
                "industry_agent": {
                    "avg_household_risk": 0.48 + shock["sentiment_delta"] * 0.2,
                    "avg_firm_hiring": 0.12 + shock["fiscal_stimulus_delta"] * 0.6,
                },
                "market_microstructure": {
                    "buy_volume": float(metrics["volume"].iloc[-1]) * (1.1 - float(metrics["panic_level"].iloc[-1])),
                    "sell_volume": float(metrics["volume"].iloc[-1]) * (0.45 + float(metrics["panic_level"].iloc[-1])),
                    "trade_count": int(float(metrics["volume"].iloc[-1]) / 1200),
                    "matching_mode": "policy_lab",
                },
            },
            key_prefix="policy_lab",
        )
    with detail_right:
        dashboard_ui.render_risk_event_timeline(metrics, key_prefix="policy_lab")
