# file: ui/dashboard.py
"""
监管仪表板 - Streamlit 前端

政策风洞控制面板，允许监管者：
1. 调整交易参数 (印花税、T+0/T+1、杠杆)
2. 实时观察市场反应
3. 查看恐慌指数热力图

运行: streamlit run ui/dashboard.py

作者: Civitas Economica Team
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time
import os
import json

if os.environ.get("CIVITAS_DASHBOARD_EMBED") == "1":
    st.set_page_config = lambda *args, **kwargs: None


# ==========================================
# 页面配置
# ==========================================

st.set_page_config(
    page_title="Civitas 监管仪表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 自定义样式
# ==========================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .risk-high { color: #ff4d4d; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #00cc66; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 政策参数侧边栏
# ==========================================

def render_policy_sidebar() -> Dict:
    """
    渲染政策参数侧边栏
    
    Returns:
        政策参数字典
    """
    st.sidebar.header("📋 政策参数")
    st.sidebar.markdown("---")
    
    # 印花税率
    st.sidebar.subheader("💰 交易税费")
    stamp_duty = st.sidebar.slider(
        "印花税率",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.01,
        format="%.2f%%",
        help="仅卖出时收取"
    )
    
    commission = st.sidebar.slider(
        "佣金率",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01,
        format="%.2f%%",
        help="双向收取，最低 5 元"
    )
    
    st.sidebar.markdown("---")
    
    # 结算模式
    st.sidebar.subheader("📅 结算模式")
    settlement_mode = st.sidebar.radio(
        "选择模式",
        options=["T+1 (A股标准)", "T+0 (压力测试)"],
        index=0,
        help="T+0 模式用于测试极端情况"
    )
    
    st.sidebar.markdown("---")
    
    # 杠杆限制
    st.sidebar.subheader("📈 杠杆控制")
    leverage = st.sidebar.number_input(
        "最大杠杆倍数",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="1.0 = 无杠杆"
    )
    
    maintenance_margin = st.sidebar.slider(
        "维持保证金率",
        min_value=0.10,
        max_value=0.50,
        value=0.25,
        step=0.05,
        format="%.0f%%",
        help="低于此水平触发强制平仓"
    )
    
    st.sidebar.markdown("---")
    
    # 高频交易限制
    st.sidebar.subheader("🚀 高频交易")
    otr_threshold = st.sidebar.number_input(
        "OTR 阈值",
        min_value=5.0,
        max_value=100.0,
        value=10.0,
        step=5.0,
        help="订单成交比超过此值施加惩罚"
    )
    
    hft_penalty = st.sidebar.slider(
        "惩罚费率",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        format="%.2f%%"
    )
    
    st.sidebar.markdown("---")
    
    # 熔断设置
    st.sidebar.subheader("🛑 熔断机制")
    circuit_breaker_enabled = st.sidebar.checkbox("启用熔断", value=True)
    
    if circuit_breaker_enabled:
        circuit_breaker_threshold = st.sidebar.slider(
            "熔断阈值",
            min_value=3.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            format="%.1f%%"
        )
    else:
        circuit_breaker_threshold = None
    
    return {
        "stamp_duty": stamp_duty / 100,
        "commission": commission / 100,
        "t_plus_1": settlement_mode == "T+1 (A股标准)",
        "leverage": leverage,
        "maintenance_margin": maintenance_margin,
        "otr_threshold": otr_threshold,
        "hft_penalty": hft_penalty / 100,
        "circuit_breaker_enabled": circuit_breaker_enabled,
        "circuit_breaker_threshold": circuit_breaker_threshold
    }


# ==========================================
# K 线图
# ==========================================

def render_candlestick_chart(
    prices: pd.DataFrame,
    title: str = "实时 K 线图"
) -> None:
    """
    渲染 K 线图
    
    Args:
        prices: 包含 open, high, low, close, volume 的 DataFrame
        title: 图表标题
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # K 线
    fig.add_trace(
        go.Candlestick(
            x=prices.index,
            open=prices['open'],
            high=prices['high'],
            low=prices['low'],
            close=prices['close'],
            increasing_line_color='#ff4d4d',  # A股红涨
            decreasing_line_color='#00cc66',  # A股绿跌
            name="价格"
        ),
        row=1, col=1
    )
    
    # 成交量
    colors = ['#ff4d4d' if c >= o else '#00cc66' 
              for o, c in zip(prices['open'], prices['close'])]
    
    fig.add_trace(
        go.Bar(
            x=prices.index,
            y=prices['volume'],
            marker_color=colors,
            name="成交量"
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=500,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 恐慌指数热力图
# ==========================================

def render_panic_heatmap(
    panic_grid: np.ndarray,
    title: str = "恐慌指数热力图"
) -> None:
    """
    渲染恐慌指数热力图
    
    基于社会网络中 Agent 的情绪状态。
    
    Args:
        panic_grid: 2D 恐慌值数组 (0-1)
        title: 图表标题
    """
    fig = go.Figure(data=go.Heatmap(
        z=panic_grid,
        colorscale=[
            [0.0, '#00cc66'],   # 绿色 - 平静
            [0.3, '#ffff00'],   # 黄色 - 中性
            [0.5, '#ffaa00'],   # 橙色 - 担忧
            [0.7, '#ff6600'],   # 深橙 - 恐惧
            [1.0, '#ff0000']    # 红色 - 恐慌
        ],
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="恐慌指数",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["平静", "乐观", "中性", "担忧", "恐慌"]
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="网格 X",
        yaxis_title="网格 Y",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 风险指标卡片
# ==========================================

def render_risk_metrics(metrics: Dict) -> None:
    """渲染风险指标卡片"""
    cols = st.columns(4)
    
    with cols[0]:
        margin_level = metrics.get("avg_margin_level", 0.5)
        color = "risk-high" if margin_level < 0.3 else ("risk-medium" if margin_level < 0.5 else "risk-low")
        st.metric("平均保证金率", f"{margin_level:.1%}")
        
    with cols[1]:
        liquidations = metrics.get("pending_liquidations", 0)
        st.metric("待平仓订单", f"{liquidations}")
        
    with cols[2]:
        hft_violations = metrics.get("hft_violations", 0)
        st.metric("高频违规", f"{hft_violations}")
        
    with cols[3]:
        panic_index = metrics.get("panic_index", 0.3)
        color = "risk-high" if panic_index > 0.7 else ("risk-medium" if panic_index > 0.4 else "risk-low")
        st.metric("恐慌指数", f"{panic_index:.2f}")


# ==========================================
# 模拟控制
# ==========================================

def render_simulation_controls() -> Dict:
    """渲染模拟控制面板"""
    st.subheader("🎮 模拟控制")
    
    cols = st.columns([1, 1, 1, 2])
    
    with cols[0]:
        if st.button("▶️ 运行", type="primary", use_container_width=True):
            return {"action": "run"}
    
    with cols[1]:
        if st.button("⏸️ 暂停", use_container_width=True):
            return {"action": "pause"}
    
    with cols[2]:
        if st.button("🔄 重置", use_container_width=True):
            return {"action": "reset"}
    
    with cols[3]:
        speed = st.slider("速度", 1, 10, 5, key="sim_speed")
    
    return {"action": None, "speed": speed}


# ==========================================
# Agent 分布图
# ==========================================

def render_agent_distribution(agents_data: Dict) -> None:
    """渲染 Agent 类型分布"""
    fig = go.Figure(data=[go.Pie(
        labels=list(agents_data.keys()),
        values=list(agents_data.values()),
        hole=0.4,
        marker_colors=['#667eea', '#764ba2', '#f093fb', '#ffa600']
    )])
    
    fig.update_layout(
        title="Agent 类型分布",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 主页面
# ==========================================

def _iter_agent_cores(ctrl):
    if not ctrl or not hasattr(ctrl, "model") or not hasattr(ctrl.model, "agents"):
        return []
    try:
        agents = list(ctrl.model.agents)
    except Exception:
        return []
    rows = []
    for agent in agents:
        core = getattr(agent, "core", None)
        if core is None:
            continue
        rows.append((agent, core))
    return rows


def _read_beliefs(agent_id: str) -> Dict:
    path = os.path.join("data", "beliefs", f"beliefs_{agent_id}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def render_manager_analyst_panel(ctrl) -> None:
    st.subheader("经理-分析师状态")
    rows = _iter_agent_cores(ctrl)
    if not rows:
        st.info("暂无活动智能体。")
        return

    step = getattr(getattr(ctrl, "model", None), "steps", 0)
    ram_active = 0
    fearful = 0
    table = []
    for agent, core in rows:
        emotional = getattr(core, "emotional_state", "Unknown")
        if emotional == "Fearful":
            fearful += 1
        ram_until = int(getattr(core, "_ram_until_step", 0) or 0)
        ram_on = step <= ram_until
        if ram_on:
            ram_active += 1
        table.append({
            "agent_id": getattr(core, "agent_id", getattr(agent, "id", "unknown")),
            "emotion": emotional,
            "ram_active": ram_on,
            "ram_trigger": getattr(core, "_ram_last_trigger", ""),
            "last_cvar": getattr(core, "_last_cvar", None),
        })

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("智能体数", len(rows))
    with c2:
        st.metric("RAM 触发", ram_active)
    with c3:
        st.metric("恐惧情绪", fearful)
    st.dataframe(pd.DataFrame(table), use_container_width=True)


def render_intent_mapping_panel(ctrl) -> None:
    st.subheader("交易意图到订单映射（FCN）")
    rows = _iter_agent_cores(ctrl)
    if not rows:
        st.info("暂无活动智能体。")
        return

    table = []
    for agent, core in rows:
        action = None
        qty = None
        price = None
        pending = getattr(agent, "pending_action", None)
        if isinstance(pending, dict):
            action = pending.get("action")
            qty = pending.get("quantity")
            price = pending.get("price")
        table.append({
            "agent_id": getattr(core, "agent_id", getattr(agent, "id", "unknown")),
            "last_action": action or "HOLD",
            "quantity": qty or 0,
            "limit_price": price or 0.0,
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True)


def render_social_network_panel(ctrl) -> None:
    st.subheader("社交网络（BDI 与边权）")
    if not ctrl or not hasattr(ctrl, "model"):
        st.info("仿真尚未启动。")
        return

    graph = getattr(ctrl.model, "social_graph", None)
    diffusion = getattr(ctrl.model, "diffusion", None)
    if graph is None or diffusion is None:
        st.info("社交网络不可用。")
        return

    weights = [float(data.get("weight", 1.0)) for _, _, data in graph.graph.edges(data=True)]
    avg_weight = float(np.mean(weights)) if weights else 0.0

    pressure = []
    similarity = []
    sample_nodes = list(graph.agents.keys())[:50]
    for node_id in sample_nodes:
        diag = diffusion.compute_infection_signal(node_id)
        pressure.append(diag.get("pressure", 0.0))
        similarity.append(diag.get("avg_similarity", 0.0))

    stats = graph.get_network_stats()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg Edge Weight", f"{avg_weight:.2f}")
    with c2:
        st.metric("Avg Semantic Sim", f"{(np.mean(similarity) if similarity else 0.0):.2f}")
    with c3:
        st.metric("Avg Pressure", f"{(np.mean(pressure) if pressure else 0.0):.2f}")

    st.json({
        "nodes": stats.get("total_nodes"),
        "avg_degree": stats.get("avg_degree"),
        "clustering": stats.get("clustering_coefficient"),
        "sentiment": stats.get("sentiment_distribution"),
    })


def render_evolution_panel(ctrl, cadence: str = "day") -> None:
    st.subheader("演化周期")
    cadence_label = "按日" if cadence == "day" else "按周"
    st.write(f"触发节奏: {cadence_label}")
    if not ctrl or not hasattr(ctrl, "model"):
        st.info("仿真尚未启动。")
        return
    st.info("演化在后端仿真循环中执行，请用侧边栏节奏设置与后端保持一致。")


def render_wind_tunnel_panel(ctrl) -> None:
    st.subheader("风洞预测")
    rows = _iter_agent_cores(ctrl)
    if not rows:
        st.info("暂无活动智能体。")
        return
    agent_ids = [getattr(core, "agent_id", getattr(agent, "id", "unknown")) for agent, core in rows]
    selected = st.selectbox("智能体", agent_ids, key="wind_tunnel_agent")
    core = None
    for agent, c in rows:
        if getattr(c, "agent_id", getattr(agent, "id", "")) == selected:
            core = c
            break
    if core is None:
        st.info("未找到该智能体。")
        return
    records = getattr(core, "_wind_tunnel_records", [])
    if not records:
        st.info("暂无风洞记录。")
        return
    df = pd.DataFrame(records)
    st.line_chart(df.set_index("step")[["confidence"]])
    st.dataframe(df.tail(10), use_container_width=True)


def render_belief_panel(ctrl) -> None:
    st.subheader("投资信仰")
    rows = _iter_agent_cores(ctrl)
    if not rows:
        st.info("暂无活动智能体。")
        return
    agent_ids = [getattr(core, "agent_id", getattr(agent, "id", "unknown")) for agent, core in rows]
    selected = st.selectbox("智能体", agent_ids, key="belief_agent")
    payload = _read_beliefs(selected)
    if not payload:
        st.info("该智能体暂无持久化信仰。")
        return
    st.json(payload)


def main():
    """主页面"""
    st.markdown('<h1 class="main-header">📊 Civitas 监管仪表板</h1>', unsafe_allow_html=True)
    st.markdown("**政策风洞** - 调整监管参数，观察市场反应")
    
    # 侧边栏：政策参数
    policy_params = render_policy_sidebar()
    
    # 保存到 session state
    if 'policy_params' not in st.session_state:
        st.session_state.policy_params = policy_params
    else:
        st.session_state.policy_params = policy_params
    
    # 主区域布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 模拟控制
        control_result = render_simulation_controls()
        
        st.markdown("---")
        
        # 生成模拟数据
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        base_price = 3000
        
        prices_data = {
            'open': base_price + np.cumsum(np.random.randn(100) * 20),
            'close': base_price + np.cumsum(np.random.randn(100) * 20),
            'volume': np.random.randint(1000000, 5000000, 100)
        }
        prices_data['high'] = np.maximum(prices_data['open'], prices_data['close']) + np.random.rand(100) * 30
        prices_data['low'] = np.minimum(prices_data['open'], prices_data['close']) - np.random.rand(100) * 30
        
        prices_df = pd.DataFrame(prices_data, index=dates)
        
        # K 线图
        render_candlestick_chart(prices_df, "上证指数模拟")
    
    with col2:
        st.subheader("📈 当前参数")
        
        st.info(f"""
        **交易成本**
        - 印花税: {policy_params['stamp_duty']:.2%}
        - 佣金: {policy_params['commission']:.2%}
        
        **结算模式**
        - {'T+1' if policy_params['t_plus_1'] else 'T+0'}
        
        **杠杆控制**
        - 最大杠杆: {policy_params['leverage']:.1f}x
        - 维持保证金: {policy_params['maintenance_margin']:.0%}
        
        **高频限制**
        - OTR 阈值: {policy_params['otr_threshold']:.0f}
        """)
        
        # Agent 分布
        render_agent_distribution({
            "散户": 8000,
            "机构": 500,
            "量化": 200,
            "做市商": 50
        })
    
    st.markdown("---")
    
    # 风险指标
    st.subheader("⚠️ 风险监控")
    render_risk_metrics({
        "avg_margin_level": 0.45,
        "pending_liquidations": 12,
        "hft_violations": 5,
        "panic_index": 0.38
    })
    
    # 恐慌热力图
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # 生成模拟恐慌网格
        panic_grid = np.random.rand(20, 20) * 0.6  # 基础恐慌
        # 在某些区域增加恐慌
        panic_grid[5:10, 5:10] += 0.3  # 恐慌聚集
        panic_grid = np.clip(panic_grid, 0, 1)
        
        render_panic_heatmap(panic_grid, "社会网络恐慌热力图")
    
    with col2:
        # 杠杆分布直方图
        leverage_data = np.random.exponential(1.5, 1000)
        leverage_data = np.clip(leverage_data, 1, policy_params['leverage'])
        
        fig = go.Figure(data=[go.Histogram(
            x=leverage_data,
            nbinsx=20,
            marker_color='#667eea'
        )])
        fig.update_layout(
            title="杠杆使用分布",
            xaxis_title="杠杆倍数",
            yaxis_title="Agent 数量",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 页脚
    st.markdown("---")
    st.caption("Civitas Economica © 2024 | 政策风洞模拟系统")


if __name__ == "__main__":
    main()
