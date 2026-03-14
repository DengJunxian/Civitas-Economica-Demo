# file: app.py
"""
Civitas A股政策仿真平台 - 增强版

新增功能：
1. 加载进度条（带百分比）
2. Agent 思维链可视化（fMRI）
3. 量化群体监控
4. 历史回测模式
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import nest_asyncio
import os

# --- 1. 页面配置 (必须是第一个 Streamlit 命令) ---
# --- 1. 页面配置 ---
# st.set_page_config 已移至文件顶部

# 解决 Streamlit 与 Asyncio 的循环冲突
nest_asyncio.apply()

from datetime import datetime
from typing import Optional

# 引入核心模块
from config import GLOBAL_CONFIG
from core.scheduler import SimulationController
from agents.brain import DeepSeekBrain, ThoughtRecord
from agents.quant_group import QuantGroupManager, QuantStrategyGroup
from core.backtester import HistoricalBacktester, BacktestConfig, BacktestReportGenerator

# 新增模块
from agents.debate_brain import DebateBrain, DebateRecord, DebateRole
from agents.reflection import ReflectionEngine, ReflectiveAgent
from core.behavioral_finance import (
    prospect_utility, calculate_csad, herding_intensity,
    create_behavioral_profile, BehavioralProfile
)
from core.regulatory_sandbox import (
    RegulatoryModule, NationalStabilityFund, CircuitBreaker,
    ProgrammaticTradingRegulator
)
from core.policy_committee import PolicyCommittee

# 引入UI组件
os.environ["CIVITAS_DASHBOARD_EMBED"] = "1"
from ui import dashboard as dashboard_ui
from ui.demo_wind_tunnel import render_demo_tab

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="数治观澜 —— 金融政策风洞推演沙箱",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 深色主题 CSS 增强
st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; color: #e0e0e0; }
    div.stButton > button {
        background-color: #1a1a2e; color: #e0e0e0; border: 1px solid #4a4a6a;
    }
    div.stButton > button:hover {
        background-color: #2a2a4e; color: #ffffff;
    }
    /* 认知透镜样式 */
    .reasoning-box {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px;
        height: 300px;
        overflow-y: scroll;
        font-family: 'Microsoft YaHei', 'PingFang SC', monospace;
        font-size: 13px;
        color: #c9d1d9;
        line-height: 1.6;
    }
    /* 政策分析框样式 */
    .policy-analysis-box {
        background-color: #161b22;
        border: 1px solid #f0883e;
        border-radius: 6px;
        padding: 12px;
        max-height: 400px;
        overflow-y: scroll;
        font-family: 'Microsoft YaHei', 'PingFang SC', monospace;
        font-size: 13px;
        color: #f0883e;
        line-height: 1.6;
    }
    /* 统计指标样式 */
    .metric-card {
        background-color: #161b22;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    /* Agent 卡片样式 */
    .agent-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    .agent-card:hover {
        border-color: #58a6ff;
        background-color: #1f2937;
    }
    /* 情绪指标条 */
    .emotion-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(to right, #34C759, #FFD60A, #FF3B30);
    }
    .emotion-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    /* 进度条容器 */
    .progress-container {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 状态管理 ---

if 'controller' not in st.session_state:
    st.session_state.controller = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'market_history' not in st.session_state:
    st.session_state.market_history = []
if 'csad_history' not in st.session_state:
    st.session_state.csad_history = []
if 'policy_analysis' not in st.session_state:
    st.session_state.policy_analysis = None
if 'historical_loaded' not in st.session_state:
    st.session_state.historical_loaded = False
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = None
if 'quant_manager' not in st.session_state:
    st.session_state.quant_manager = None
if 'backtest_mode' not in st.session_state:
    st.session_state.backtest_mode = False
if 'backtester' not in st.session_state:
    st.session_state.backtester = None
if 'init_progress' not in st.session_state:
    st.session_state.init_progress = 0
if 'init_status' not in st.session_state:
    st.session_state.init_status = ""
# 新增: 仿真模式状态
if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = "SMART"
if 'day_cycle_paused' not in st.session_state:
    st.session_state.day_cycle_paused = False
if 'evolution_cadence' not in st.session_state:
    st.session_state.evolution_cadence = "day"
if 'regulatory_module' not in st.session_state:
    from core.regulatory_sandbox import RegulatoryModule
    st.session_state.regulatory_module = RegulatoryModule()

# --- 3. 辅助函数 ---

def get_emotion_icon(score: float) -> str:
    """根据情绪分数获取图标"""
    if score > 0.3:
        return "🟢"  # 贪婪
    elif score < -0.3:
        return "🔴"  # 恐惧
    else:
        return "⚪"  # 中性

def get_emotion_color(score: float) -> str:
    """根据情绪分数获取颜色"""
    if score > 0.3:
        return "#34C759"  # 绿色
    elif score < -0.3:
        return "#FF3B30"  # 红色
    else:
        return "#FFD60A"  # 黄色

def render_progress_bar(current: int, total: int, status: str = ""):
    """渲染进度条"""
    progress = current / max(total, 1)
    percentage = int(progress * 100)
    
    st.markdown(f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span>{status}</span>
            <span style="color: #58a6ff; font-weight: bold;">{percentage}%</span>
        </div>
        <div style="background: #2a2a4e; border-radius: 4px; height: 12px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #58a6ff, #34C759); 
                        width: {percentage}%; height: 100%; 
                        transition: width 0.3s ease;"></div>
        </div>
        <div style="color: #888; font-size: 12px; margin-top: 5px;">
            {current} / {total}
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. 侧边栏配置 ---

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <div style="font-size: 22px; font-weight: bold; color: #4DA6FF;">🏛️ 数治观澜</div>
        <div style="font-size: 12px; color: #888; margin-top: 5px;">基于大模型多智能体的金融政策风洞推演沙箱</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"版本: {GLOBAL_CONFIG.VERSION} | DeepSeek R1 + 多模型")
    
    # --- API 密钥输入 ---
    api_key = st.text_input("DeepSeek API 密钥 *", type="password", 
                            value="",
                            help="必填，用于驱动智能体思考")
    zhipu_key = st.text_input("智谱 API 密钥 (快速模式)", type="password",
                              value="",
                              help="快速模式专用，使用GLM-4-FlashX模型")
    
    st.divider()
    
    # --- 仿真模式选择 ---
    st.subheader("⚡ 仿真模式")
    sim_mode = st.radio(
        "选择模式",
        ["🧠 智能模式", "⚡ 快速模式", "🔬 深度模式"],
        help="智能:自适应(≤15s/天) | 快速:仅对话模型(≤5s) | 深度:完整推理(≤30s)",
        key="sim_mode_radio"
    )
    mode_map = {"🧠 智能模式": "SMART", "⚡ 快速模式": "FAST", "🔬 深度模式": "DEEP"}
    st.session_state.simulation_mode = mode_map.get(sim_mode, "SMART")
    
    # 如果控制器已初始化，实时切换模式
    if st.session_state.controller:
        st.session_state.controller.set_mode(st.session_state.simulation_mode)
    
    st.divider()
    
    # --- 运行模式切换 ---
    mode_tab = st.radio(
        "运行模式",
        ["🎮 实时仿真", "📊 历史回测"],
        horizontal=True,
        help="实时仿真：Agent在模拟市场中交易\n历史回测：使用真实历史数据校准"
    )
    st.session_state.backtest_mode = (mode_tab == "📊 历史回测")
    
    st.divider()
    
    # --- 政策输入模块 ---
    st.subheader("📜 政策注入")
    
    policy_text = st.text_area(
        "输入政策内容",
        placeholder="例如：降低证券交易印花税至0.05%以刺激市场流动性...",
        height=120,
        help="输入政策文本后，系统会通过DeepSeek推理链分析其对市场和投资者行为的影响"
    )
    
    analyze_btn = st.button("🔍 分析政策影响", use_container_width=True)
    
    if analyze_btn and policy_text:
        if st.session_state.controller:
            # 发送指令到队列
            st.session_state.cmd_queue.put({
                "type": "policy",
                "content": policy_text
            })
            st.info("📨 政策分析请求已提交，系统正在后台异步处理...")
            # 临时 placeholder
            st.session_state.policy_analysis = {
                "text": policy_text,
                "result": {"market_impact": "分析中..."},
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        else:
            st.warning("请先点击 '启动' 初始化仿真系统 (无需开始运行即可注入政策)")
    
    # 显示政策分析结果
    if st.session_state.policy_analysis:
        with st.expander("📊 政策分析结果", expanded=True):
            result = st.session_state.policy_analysis["result"]
            st.markdown(f"**分析时间:** {st.session_state.policy_analysis['timestamp']}")
            st.markdown(f"**印花税率:** {result.get('tax_rate', 0.0005):.4%}")
            st.markdown(f"**流动性注入概率:** {result.get('liquidity_injection', 0):.1%}")
            st.markdown(f"**市场恐慌因子:** {result.get('fear_factor', 0):.2f}")
            st.markdown(f"**初始新闻:** {result.get('initial_news', '无')}")
    
    st.divider()
    
    # --- 量化群体配置 ---
    with st.expander("🤖 量化群体设置", expanded=False):
        quant_strategy = st.selectbox(
            "策略模板",
            ["momentum", "mean_reversion", "risk_parity", "news_driven"],
            format_func=lambda x: {
                'momentum': '动量追踪',
                'mean_reversion': '均值回归',
                'risk_parity': '风险平价',
                'news_driven': '消息驱动'
            }.get(x, x)
        )
        
        quant_agents = st.slider("群体规模", min_value=5, max_value=20, value=10)
        
        if st.button("创建量化群体", use_container_width=True):
            if api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY:
                if st.session_state.quant_manager is None:
                    st.session_state.quant_manager = QuantGroupManager(
                        api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
                    )
                
                # 创建进度条占位
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                def update_progress(current, total, msg):
                    progress_placeholder.progress(current / total)
                    status_placeholder.text(f"正在创建 {msg}...")
                
                group = st.session_state.quant_manager.create_from_template(
                    f"quant_{len(st.session_state.quant_manager.groups)}",
                    quant_strategy,
                    quant_agents,
                    update_progress
                )
                
                progress_placeholder.empty()
                status_placeholder.empty()
                
                if group:
                    st.success(f"✅ 创建了 {quant_agents} 个 {group.strategy_name} Agent")
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.warning("请先输入 API 密钥")
    
    st.divider()
    
    # --- 状态指示器 ---
    st.subheader("功能面板设置")
    st.session_state.evolution_cadence = st.selectbox(
        "演化触发节奏",
        ["day", "week"],
        index=0 if st.session_state.evolution_cadence == "day" else 1
    )

    st.divider()

    status_color = "#34C759" if st.session_state.is_running else ("#FFD60A" if st.session_state.get('day_cycle_paused') else "#8E8E93")
    status_text = "🟢 仿真运行中" if st.session_state.is_running else ("🟡 仿真已暂停" if st.session_state.get('day_cycle_paused') else "⚪ 待启动")
    
    st.markdown(f"""
    <div style="background-color: #1c1c1e; padding: 10px; border-radius: 8px; border-left: 5px solid {status_color}; margin-bottom: 10px;">
        <span style="font-weight: bold; font-size: 16px; color: {status_color};">{status_text}</span>
        <span style="float: right; color: #888; font-size: 12px;">{st.session_state.simulation_mode} MODE</span>
    </div>
    """, unsafe_allow_html=True)

    # --- 控制按钮 ---
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ 启动", use_container_width=True)
    with col2:
        stop_btn = st.button("⏸ 暂停", use_container_width=True)
    
    # 启动逻辑
    if start_btn and not st.session_state.is_running:
        # 检查API密钥：DeepSeek或智谱至少需要一个
        has_deepseek = bool(api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY)
        has_zhipu = bool(zhipu_key or GLOBAL_CONFIG.ZHIPU_API_KEY)
        
        if not has_deepseek and not has_zhipu:
            st.error("请至少输入一个API密钥（DeepSeek或智谱）!")
        else:
            # 如果只有智谱API，强制使用快速模式
            if not has_deepseek and has_zhipu:
                st.session_state.simulation_mode = "FAST"
                st.warning("⚠️ 未检测到DeepSeek API，已自动切换到快速模式（使用智谱GLM）")
            
            st.session_state.is_running = True
            
            if st.session_state.controller is None:
                # 显示初始化进度
                init_container = st.container()
                
                with init_container:
                    st.markdown("### 🚀 正在初始化仿真系统...")
                    
                    # 阶段1: 连接API
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if has_deepseek:
                        status_text.text("🔌 连接 DeepSeek API...")
                    else:
                        status_text.text("🔌 连接 智谱GLM API...")
                    progress_bar.progress(10)
                    time.sleep(0.3)
                    
                    # 阶段2: 初始化控制器
                    status_text.text("⚙️ 初始化仿真控制器...")
                    progress_bar.progress(30)
                    
                    st.session_state.controller = SimulationController(
                        deepseek_key=api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY or "",
                        zhipu_key=zhipu_key or GLOBAL_CONFIG.ZHIPU_API_KEY or None,
                        mode=st.session_state.simulation_mode,
                        quant_manager=st.session_state.quant_manager,
                        regulatory_module=st.session_state.regulatory_module
                    )
                    
                    progress_bar.progress(50)
                    
                    # 阶段3: 加载历史数据
                    status_text.text("📈 加载历史K线数据...")
                    progress_bar.progress(60)
                    
                    if not st.session_state.historical_loaded:
                        total_candles = len(st.session_state.controller.market.candles)
                        for i, candle in enumerate(st.session_state.controller.market.candles):
                            st.session_state.market_history.append({
                                "time": candle.timestamp,
                                "open": candle.open,
                                "high": candle.high,
                                "low": candle.low,
                                "close": candle.close,
                                "is_historical": not candle.is_simulated
                            })
                            if i % 50 == 0:
                                progress_bar.progress(60 + int(30 * i / max(total_candles, 1)))
                        
                        st.session_state.historical_loaded = True
                    
                    # 阶段4: 初始化 Agent 和默认量化群体
                    status_text.text("🧠 初始化智能体...")
                    progress_bar.progress(90)
                    
                    # 创建默认量化群体（4种策略各3个，共12个）
                    if st.session_state.quant_manager is None:
                        from agents.quant_group import QuantGroupManager
                        effective_key = zhipu_key or GLOBAL_CONFIG.ZHIPU_API_KEY or api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
                        st.session_state.quant_manager = QuantGroupManager(effective_key)
                        
                        # 四种策略模板
                        strategies = ["momentum", "mean_reversion", "risk_parity", "news_driven"]
                        for strategy in strategies:
                            st.session_state.quant_manager.create_from_template(
                                f"default_{strategy}",
                                strategy,
                                1,  # 每种策略1个Agent
                                lambda c, t, m: None  # 静默创建
                            )
                    
                    progress_bar.progress(95)
                    time.sleep(0.2)
                    
                    # 完成
                    status_text.text("✅ 系统就绪！")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                
            st.rerun()
            
    if stop_btn:
        st.session_state.is_running = False
        st.rerun()

    st.markdown("---")
    if st.button("📑 生成报告", use_container_width=True):
        if st.session_state.controller:
            if st.session_state.is_running:
                st.warning("⚠️ 仿真正在运行中。请先点击上方的「停止/暂停」按钮暂停仿真，然后再生成报告。")
            else:
                with st.spinner("正在生成仿真结果评估报告..."):
                    try:
                        import concurrent.futures

                        # --- 收集仿真上下文数据 ---
                        history = st.session_state.market_history
                        sim_history = [h for h in history if not h.get('is_historical', True)]

                        if not sim_history:
                            st.warning("暂无仿真数据，请先运行仿真。")
                        else:
                            ctrl_ref = st.session_state.controller
                            first_sim = sim_history[0]
                            last_sim = sim_history[-1]
                            sim_days = len(sim_history)

                            start_price = first_sim['close']
                            end_price = last_sim['close']
                            total_return = (end_price - start_price) / start_price * 100

                            # 波动率
                            if len(sim_history) > 1:
                                import numpy as np
                                closes = [h['close'] for h in sim_history]
                                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                                volatility = float(np.std(returns) * 100)
                                max_price = max(closes)
                                min_price = min(closes)
                            else:
                                volatility = 0.0
                                max_price = end_price
                                min_price = end_price

                            # 政策信息
                            policy_info = st.session_state.policy_analysis
                            policy_text = policy_info.get('text', '无') if policy_info else '无'

                            # 政策参数
                            try:
                                policy_status = ctrl_ref.model.get_policy_status()
                                cb_info = policy_status.get('circuit_breaker', {})
                                tax_info = policy_status.get('transaction_tax', {})
                            except Exception:
                                cb_info = {}
                                tax_info = {}

                            # CSAD 均值
                            csad_data = st.session_state.csad_history
                            avg_csad = sum(csad_data) / len(csad_data) if csad_data else 0

                            # 恐慌指数
                            panic = ctrl_ref.market.panic_level if hasattr(ctrl_ref.market, 'panic_level') else 0

                            # --- 构建评估报告 Prompt ---
                            summary_prompt = f"""你是一位资深的金融政策分析师。请根据以下仿真实验数据，生成一份「政策效果评估报告」。

【仿真概况】
- 仿真天数: {sim_days} 天
- 起始日期: {first_sim['time']}
- 结束日期: {last_sim['time']}
- 起始价格: {start_price:.2f}
- 结束价格: {end_price:.2f}
- 累计涨跌幅: {total_return:+.2f}%
- 日波动率: {volatility:.3f}%
- 价格区间: {min_price:.2f} ~ {max_price:.2f}
- 当前恐慌指数: {panic:.2f}
- 平均 CSAD(羊群效应): {avg_csad:.4f}

【注入的政策】
{policy_text}

【当前监管参数】
- 熔断机制: {'启用' if cb_info.get('active') else '未启用'}, 阈值: {cb_info.get('threshold', 0):.0%}
- 交易印花税: {'启用' if tax_info.get('active') else '未启用'}, 税率: {tax_info.get('rate', 0.001) * 1000:.1f}‰

【要求】
请从以下角度进行分析，字数控制在500字左右：
1. **政策效果评估**: 分析注入政策对市场价格、波动率、流动性和投资者行为的影响
2. **市场稳定性分析**: 基于恐慌指数和CSAD评估市场稳定程度
3. **风险提示**: 指出仿真中暴露出的潜在风险
4. **政策建议**: 针对当前市场状态给出政策调整建议"""

                            # --- 使用独立线程执行异步 API 调用 ---
                            router = ctrl_ref.model_router
                            # 报告模型优先级：先尝试 deepseek-chat，再降级到智谱 GLM
                            priority = ["deepseek-chat"]
                            if router.has_zhipu:
                                priority.append("glm-4-flashx")
                            
                            def _sync_get_report(prompt, models):
                                """在独立线程的独立事件循环中执行异步 API 调用"""
                                loop = asyncio.new_event_loop()
                                try:
                                    return loop.run_until_complete(
                                        router.call_with_fallback(
                                            [{"role": "user", "content": prompt}],
                                            priority_models=models,
                                            timeout_budget=90.0,
                                            fallback_response="报告生成服务暂时不可用，请稍后重试。"
                                        )
                                    )
                                finally:
                                    loop.close()

                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(_sync_get_report, summary_prompt, priority)
                                content, _, model = future.result(timeout=120)

                            st.success(f"✅ 报告已生成 (使用模型: {model}) High-Speed Mode")
                            
                            # Store report content in session state to persist
                            st.session_state.last_report = content
                            
                            st.markdown(f"""
                            <div style="background: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d;">
                                <h4 style="color: #58a6ff;">📊 仿真结果评估报告</h4>
                                <div style="font-size: 13px; color: #888; margin-bottom: 12px;">
                                    仿真周期: {first_sim['time']} ~ {last_sim['time']} ({sim_days}天)
                                    | 累计涨跌: <span style="color: {'#FF3B30' if total_return >= 0 else '#34C759'}">{total_return:+.2f}%</span>
                                </div>
                                <div style="font-size: 14px; line-height: 1.8; color: #c9d1d9; white-space: pre-wrap;">
{content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    except concurrent.futures.TimeoutError:
                        st.error("⏰ 报告生成超时（120秒），请稍后重试。")
                    except Exception as e:
                        st.error(f"报告生成失败: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")
        


# --- 5. 主界面逻辑 ---

# 创建标签页
if st.session_state.backtest_mode:
    tab1, tab2 = st.tabs(["📊 回测结果", "🧠 Agent fMRI"])
    tab_demo = tab_feature = tab_debate = tab_reg = tab_behavior = tab_quant = None
else:
    tab_guide, tab_demo, tab1, tab2, tab_feature, tab_debate, tab_reg, tab_behavior, tab_quant = st.tabs([
        "🏠 系统导览",
        "🌪️ 沙箱风洞",
        "📈 市场走势", 
        "🧠 Agent fMRI", 
        "🧩 功能面板",
        "⚔️ 辩论室",
        "🏛️ 监管沙盒",
        "📊 行为金融",
        "🤖 量化群体"
    ])


ctrl = st.session_state.controller

# --- 主内容区 ---
if st.session_state.backtest_mode:
    # 回测模式
    with tab1:
        st.subheader("📊 历史回测校准")
        
        if st.session_state.backtester is None:
            st.session_state.backtester = HistoricalBacktester(
                BacktestConfig(period_days=1095)  # 3年数据
            )
        
        col_bt1, col_bt2 = st.columns([2, 1])
        
        with col_bt1:
            if st.button("🚀 开始回测", use_container_width=True):
                if ctrl:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def bt_progress(current, total, msg):
                        progress_bar.progress(current / max(total, 1))
                        status_text.text(msg)
                    
                    result = st.session_state.backtester.run_backtest(
                        ctrl.model.population,
                        ctrl.market,
                        bt_progress
                    )
                    
                    status_text.text("✅ 回测完成！")
                    
                    # 显示结果
                    st.markdown(BacktestReportGenerator.generate_html_report(result), 
                               unsafe_allow_html=True)
                else:
                    st.warning("请先启动仿真系统")
        
        with col_bt2:
            st.markdown("""
            **回测说明**
            
            系统将使用近3年的上证指数历史数据，
            让Agent群体在真实行情中进行交易决策。
            
            校准指标包括：
            - 📈 价格走势相关性
            - 📊 换手率相关性
            - 📉 波动率相关性
            """)

else:
    # 实时仿真模式
    with tab_guide:
        st.markdown("## 🏠 系统导览与核心结论")
        
        # 核心结论区
        if 'last_report' in st.session_state:
            with st.container():
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px; border-radius: 12px; border: 1px solid #3b82f6; margin-bottom: 20px;">
                    <h3 style="color: #60a5fa; margin-top: 0;">🎯 最新推演结论 (Key Takeaway)</h3>
                    <div style="color: #e2e8f0; font-size: 15px; line-height: 1.6;">
                """, unsafe_allow_html=True)
                st.write(st.session_state.last_report[:500] + "..." if len(st.session_state.last_report) > 500 else st.session_state.last_report)
                st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.info("💡 尚未生成政策评估报告。请在左侧注入政策并运行仿真，完成后点击「生成报告」以获取核心结论。")

        # 功能模块导览
        st.markdown("### 🛠️ 仿真功能说明")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("""
            <div class="agent-card">
                <h4 style="color: #4DA6FF;">🌪️ 沙箱风洞</h4>
                <p style="font-size: 0.9em; color: #888;">全自动的极端行情演练场。通过“政策输入-认知博弈-情绪传染-订单涌现”的完整链路，展示政策对市场的穿透式影响。</p>
            </div>
            <div class="agent-card">
                <h4 style="color: #4DA6FF;">📈 市场走势</h4>
                <p style="font-size: 0.9em; color: #888;">实时观测指数波动、K线走势及全市场成交数据。这是仿真系统最直观的宏观结果呈现。</p>
            </div>
            <div class="agent-card">
                <h4 style="color: #4DA6FF;">🧠 Agent fMRI</h4>
                <p style="font-size: 0.9em; color: #888;">深入智能体“大脑”。实时展示散户与机构在面临市场变化时的思维链（CoT）及情绪因子波动。</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_g2:
            st.markdown("""
            <div class="agent-card">
                <h4 style="color: #4DA6FF;">⚔️ 辩论室</h4>
                <p style="font-size: 0.9em; color: #888;">防幻觉博弈池。展示多空双方引擎如何通过辩论对抗，最终形成具有逻辑支撑的交易决策。</p>
            </div>
            <div class="agent-card">
                <h4 style="color: #4DA6FF;">🏛️ 监管沙盒</h4>
                <p style="font-size: 0.9em; color: #888;">推演监管干预效果。可实时调整印花税、熔断阈值等参数，观测政府干预对市场情绪的回填作用。</p>
            </div>
            <div class="agent-card">
                <h4 style="color: #4DA6FF;">🤖 量化群体</h4>
                <p style="font-size: 0.9em; color: #888;">基于算法的特殊群体。监控动量、均值回归等策略在极端行情下的“合力绞杀”或流动性提供行为。</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_demo:
        render_demo_tab()
        
    with tab_feature:
        dashboard_ui.render_manager_analyst_panel(ctrl)
        st.markdown("---")
        dashboard_ui.render_intent_mapping_panel(ctrl)
        st.markdown("---")
        dashboard_ui.render_social_network_panel(ctrl)
        st.markdown("---")
        dashboard_ui.render_evolution_panel(ctrl, st.session_state.evolution_cadence)
        st.markdown("---")
        dashboard_ui.render_wind_tunnel_panel(ctrl)
        st.markdown("---")
        dashboard_ui.render_belief_panel(ctrl)
        
    with tab1:
        # K线图全宽显示
        if ctrl:
            # --- 100天循环检测 ---
            if ctrl.day_count > 0 and ctrl.day_count % GLOBAL_CONFIG.SIMULATION_DAY_CYCLE == 0 and st.session_state.is_running:
                st.session_state.is_running = False
                st.session_state.day_cycle_paused = True
            
            # 100天循环暂停提示
            if st.session_state.day_cycle_paused:
                st.warning(f"📅 已完成 {ctrl.day_count} 天仿真！")
                col_continue, col_stop = st.columns(2)
                with col_continue:
                    if st.button("▶ 继续仿真下一个100天", use_container_width=True):
                        st.session_state.day_cycle_paused = False
                        st.session_state.is_running = True
                        st.rerun()
                with col_stop:
                    if st.button("⏹️ 结束仿真", use_container_width=True):
                        st.session_state.day_cycle_paused = False
                        st.session_state.is_running = False
            
            # --- 可视化渲染 ---
            
            # 统计面板（K线图上方）
            # 统计面板（K线图上方）
            st.markdown("### 📊 统计面板")
            
            # [Fix] 优先从 market_history 获取最新仿真数据 (包含已完成的 step)
            if st.session_state.market_history:
                latest_data = st.session_state.market_history[-1]
                current_price = latest_data['close']
                # 获取前一天的收盘价作为对比基准
                if len(st.session_state.market_history) >= 2:
                    prev_close = st.session_state.market_history[-2]['close']
                elif latest_data.get('is_historical'):
                    #如果是第一天或者历史数据，尝试用 open 
                    prev_close = latest_data['open']
                else:
                    # Fallback
                    prev_close = ctrl.market.engine.prev_close
            else:
                current_price = ctrl.market.engine.last_price
                prev_close = ctrl.market.engine.prev_close
            
            # 计算涨跌
            change_val = current_price - prev_close
            change_pct = (change_val / prev_close * 100) if prev_close else 0
            
            if change_val >= 0:
                price_color = "#FF3B30"
                arrow = "↑"
                sign = "+"
            else:
                price_color = "#34C759"
                arrow = "↓"
                sign = ""
            
            # 第一行：指数显示
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #888; font-size: 14px;">上证指数</div>
                <div style="color: {price_color}; font-size: 32px; font-weight: bold;">
                    {current_price:.2f} {arrow} <span style="font-size: 18px;">{change_val:+.2f} ({sign}{change_pct:.2f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 第二行：两列指标 (移除耗时统计和成交量)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("恐慌指数", f"{ctrl.market.panic_level:.2f}")
            with col_s2:
                st.metric("仿真天数", f"{ctrl.day_count}")
            
            st.markdown("---")
            
            # 1. K线图 (东方财富风格)
            st.subheader("📈 市场走势")
            
            st.markdown("""
            <div style="background: rgba(77, 166, 255, 0.1); border-left: 4px solid #4DA6FF; padding: 10px; margin-bottom: 20px;">
                <p style="margin: 0; font-size: 0.9em; color: #e0e0e0;">
                    <b>🔍 核心观察点：</b> 关注价格波动与成交量的共振。在政策注入后，观察是否存在非线性的剧烈跳空或流动性断层。
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 周期选择器
            period_col, _ = st.columns([1, 4])
            with period_col:
                kline_period = st.radio("周期", ["日K", "周K", "月K"], horizontal=True, label_visibility="collapsed")

            if st.session_state.market_history:
                raw_df = pd.DataFrame(st.session_state.market_history)
                raw_df['time'] = pd.to_datetime(raw_df['time'])
                
                # 安全检查：确保 volume 列存在 (防止 KeyError)
                if 'volume' not in raw_df.columns:
                    raw_df['volume'] = 0
                if not raw_df.empty and 'time' in raw_df.columns:
                    # 确保 time 列为 datetime 类型
                    raw_df['time'] = pd.to_datetime(raw_df['time'])
                    
                    # 数据重采样逻辑
                    if kline_period == "周K":
                        df = raw_df.resample('W-FRI', on='time').agg({
                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                        }).dropna().reset_index()
                    elif kline_period == "月K":
                        df = raw_df.resample('ME', on='time').agg({
                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                        }).dropna().reset_index()
                    else: # 日K
                        df = raw_df.copy()
                else:
                    df = pd.DataFrame() # Handle empty or missing 'time' column case
                
                # 计算均线
                if len(df) > 0:
                    df['MA5'] = df['close'].rolling(window=5).mean()
                    df['MA10'] = df['close'].rolling(window=10).mean()
                    df['MA20'] = df['close'].rolling(window=20).mean()
                    
                    # 确定涨跌颜色 (用于成交量)
                    df['color'] = np.where(df['close'] >= df['open'], '#FF3B30', '#34C759')
                    
                    from plotly.subplots import make_subplots
                    
                    # 创建子图: K线图占70%，成交量占30%
                    fig = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        subplot_titles=('上证指数', '成交量'),
                        row_heights=[0.7, 0.3]
                    )
                    
                    # 1. K线
                    fig.add_trace(go.Candlestick(
                        x=df['time'],
                        open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'],
                        increasing_line_color='#FF3B30',
                        increasing_fillcolor='#FF3B30',
                        decreasing_line_color='#34C759',
                        decreasing_fillcolor='#34C759',
                        name='K线'
                    ), row=1, col=1)
                    
                    # 2. 均线 (仅在非月K显示短期均线，避免混乱，或者全部显示)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['MA5'], line=dict(color='#E5B80B', width=1), name='MA5'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['MA10'], line=dict(color='#FF69B4', width=1), name='MA10'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['time'], y=df['MA20'], line=dict(color='#87CEFA', width=1), name='MA20'), row=1, col=1)
                    
                    # 3. 成交量
                    fig.add_trace(go.Bar(
                        x=df['time'], y=df['volume'],
                        marker_color=df['color'],
                        name='成交量'
                    ), row=2, col=1)
                    
                    # 布局优化
                    
                    # 默认显示范围 (最近 N 根K线)
                    default_zoom = 120 if kline_period == "日K" else (60 if kline_period == "周K" else 24)
                    if len(df) > default_zoom:
                         start_date = df['time'].iloc[-default_zoom]
                         end_date = df['time'].iloc[-1]
                         fig.update_xaxes(range=[start_date, end_date])
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(10,10,10,0.8)',
                        font=dict(color='#e0e0e0', family='Microsoft YaHei'),
                        height=600,
                        xaxis_rangeslider_visible=False,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    
                    # 坐标轴格式
                    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
                    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
                    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', title='点位', row=1, col=1)
                    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', title='成交量', row=2, col=1, showticklabels=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # CSAD 羊群效应仪表盘（全宽显示）
            st.subheader("🐑 羊群效应指标 (CSAD)")
            if st.session_state.csad_history:
                csad_df = pd.DataFrame({
                    'step': range(len(st.session_state.csad_history)), 
                    'csad': st.session_state.csad_history
                })
                fig_csad = go.Figure()
                fig_csad.add_trace(go.Scatter(
                    x=csad_df['step'], y=csad_df['csad'], 
                    mode='lines', line=dict(color='#FFD60A', width=2), name='CSAD'
                ))
                fig_csad.add_hline(y=0.15, line_dash="dash", line_color="#FF3B30", 
                                  annotation_text="羊群效应警戒线", annotation_font_color="#FF3B30")
                fig_csad.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.8)',
                    font=dict(color='#c0c0c0', family='Microsoft YaHei'), height=200,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(title='仿真步数'), yaxis=dict(title='CSAD值')
                )
                st.plotly_chart(fig_csad, use_container_width=True)
            else:
                st.info("等待仿真数据...")

# --- 异步仿真与线程管理 ---
import threading
import queue
import traceback

# 全局线程安全队列 (用于跨线程传递 metric 和 command)
if 'metrics_queue' not in st.session_state:
    st.session_state.metrics_queue = queue.Queue()
if 'cmd_queue' not in st.session_state:
    st.session_state.cmd_queue = queue.Queue()

def simulation_worker(controller, metrics_queue, cmd_queue, stop_event, pause_event):
    """
    后台仿真线程工作函数
    
    Args:
        controller: 仿真控制器实例
        metrics_queue: 用于发送指标给 UI 的队列
        cmd_queue: 接收 UI 命令的队列
        stop_event: 线程停止信号 (Set = Stop Thread)
        pause_event: 仿真暂停信号 (Set = Running, Clear = Paused)
    """
    # 为该线程创建独立的事件循环
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception as e:
        print(f"[Thread] Event Loop creation failed: {e}")
        return

    print("[Thread] 仿真线程启动 (State: Idle/Running)")
    
    while not stop_event.is_set():
        # 1. 优先处理控制指令 (无论暂停与否都处理)
        try:
            while not cmd_queue.empty():
                cmd = cmd_queue.get_nowait()
                
                if cmd['type'] == 'policy':
                    print(f"[Thread] Processing Policy: {cmd['content'][:20]}...")
                    # 异步执行政策分析
                    try:
                        policy_result = loop.run_until_complete(controller.apply_policy_async(cmd['content']))
                        # 推送完成状态
                        metrics_queue.put({
                            "type": "policy_done", 
                            "result": policy_result, 
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                    except Exception as e:
                        print(f"[Thread Policy Error] {e}")
                        metrics_queue.put({"error": f"政策注入失败: {e}"})
                        
                elif cmd['type'] == 'stop':
                    print("[Thread] Received Stop Command")
                    stop_event.set()
                    break
        except Exception as e:
            print(f"[Thread] Error processing commands: {e}")
            metrics_queue.put({"error": f"指令处理错误: {e}"})

        # 2. 如果停止信号由于指令处理被设置，退出大循环
        if stop_event.is_set():
            break

        # 3. 执行仿真步 (仅当未暂停时)
        if pause_event.is_set():
            try:
                # 执行一步仿真
                metrics = loop.run_until_complete(controller.run_tick())
                
                # 将结果放入队列 (非阻塞)
                metrics_queue.put(metrics)
                
                # 速度控制
                mode = controller.mode.value
                if mode == "FAST":
                    time.sleep(0.1)
                elif mode == "SMART":
                    time.sleep(0.5)
                else:
                    time.sleep(1.0)
                    
            except Exception as e:
                # 捕获异常但不退出线程，而是自动暂停
                err_msg = f"仿真步执行异常: {str(e)}\n{traceback.format_exc()}"
                print(f"[Thread Error] {err_msg}")
                metrics_queue.put({"error": err_msg})
                
                # 自动暂停以保护现场
                pause_event.clear()
        else:
            # 暂停状态下的空转等待，避免 CPU 100%
            time.sleep(0.1)
            
    loop.close()
    print("[Thread] 仿真线程停止")

# --- 主界面逻辑中的控制部分 ---

if 'sim_thread' not in st.session_state:
    st.session_state.sim_thread = None
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'pause_event' not in st.session_state:
    st.session_state.pause_event = threading.Event()

# 启动逻辑 (修改 start_btn 的处理)
# (这部分代码在 app.py 较上方，这里只能替换 loop 部分)

# --- 接收并更新 UI ---

# 只要 Controller 存在，就应该尝试保持后台线程运行，以便处理 Policy
if st.session_state.controller:
    # 1. 检查线程是否存活，如果没有则启动
    if st.session_state.sim_thread is None or not st.session_state.sim_thread.is_alive():
        # 重置信号
        st.session_state.stop_event.clear()
        
        # 根据 is_running 状态设置 pause_event
        # 如果 is_running 为 True，则 pause_event set (运行)
        # 如果 is_running 为 False，则 pause_event clear (暂停)
        if st.session_state.get('is_running', False):
            st.session_state.pause_event.set()
        else:
            st.session_state.pause_event.clear()

        st.session_state.sim_thread = threading.Thread(
            target=simulation_worker,
            args=(
                st.session_state.controller, 
                st.session_state.metrics_queue,
                st.session_state.cmd_queue,
                st.session_state.stop_event,
                st.session_state.pause_event
            ),
            daemon=True
        )
        st.session_state.sim_thread.start()
        print("🚀 仿真后台线程已自动启动 (Ready)")

    # 2. 同步 UI 状态到控制信号
    if st.session_state.is_running:
        if not st.session_state.pause_event.is_set():
            st.session_state.pause_event.set()
    else:
        if st.session_state.pause_event.is_set():
            st.session_state.pause_event.clear()

    # 3. 消费队列中的数据 (非阻塞，取出所有积压的数据，只渲染最新的)
    latest_metrics = None
    while not st.session_state.metrics_queue.empty():
        latest_metrics = st.session_state.metrics_queue.get()
    
    if latest_metrics:
        if "error" in latest_metrics:
            st.error(f"仿真异常: {latest_metrics['error']}")
            #如果是严重错误，暂停仿真
            if "仿真步执行异常" in latest_metrics['error']:
                st.session_state.is_running = False
                st.session_state.pause_event.clear()
                
        elif latest_metrics.get("type") == "policy_done":
            # 处理政策分析完成
            result = latest_metrics.get("result")
            timestamp = latest_metrics.get("timestamp")
            st.session_state.policy_analysis = {
                "text": "政策已注入", 
                "result": result,
                "timestamp": timestamp
            }
            st.toast("✅ 政策分析已完成并生效！")
        else:
            # 更新 Session State 数据 (常规仿真步)
            candle = latest_metrics['candle']
            
            # 检查是否重复添加 (通过 timestamp)
            last_ts = st.session_state.market_history[-1]['time'] if st.session_state.market_history else ""
            if candle.timestamp != last_ts:
                st.session_state.market_history.append({
                    "time": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "is_historical": False
                })
                # Check if CSAD is present
                if 'csad' in latest_metrics:
                    st.session_state.csad_history.append(latest_metrics['csad'])
                
            # 检查是否有降级事件
            if hasattr(ctrl.model_router, 'fallback_events') and ctrl.model_router.fallback_events:
                 for event in ctrl.model_router.fallback_events:
                     st.toast(f"⚠️ {event.get('message', '模型已降级')}", icon="⚠️")
                 ctrl.model_router.fallback_events.clear()

    # 4. 自动刷新 UI (仅当正在仿真时，或者有 Policy 结果时可能需要刷新)
    if st.session_state.is_running or (latest_metrics and latest_metrics.get("type") == "policy_done"):
        time.sleep(0.5) 
        st.rerun()
    
else:
    # 如果不处于 Controller 初始化状态，确保线程清理 (一般不会走到这里，除非手动 Stop)
    if st.session_state.sim_thread and st.session_state.sim_thread.is_alive():
        if not st.session_state.controller:
            st.session_state.stop_event.set()
            st.session_state.sim_thread.join(timeout=1.0)
            st.session_state.sim_thread = None


            
    # 欢迎页 - 当不在运行状态且没有初始化完成时显示
    if not st.session_state.controller:
        with tab1:
            st.markdown("""
            ## 🏛️ 欢迎使用 Civitas A股政策仿真平台
            
            本系统使用 **DeepSeek R1** 大模型驱动智能体进行投资决策，模拟政策对A股市场的影响。
            
            ### 快速开始
            
            1. 在侧边栏输入您的 **DeepSeek API 密钥**
            2. 点击 **▶ 启动** 按钮初始化仿真
            3. 在"政策注入"区域输入政策文本
            4. 观察智能体如何响应政策变化
            
            ### 系统状态
            
            | 组件 | 状态 |
            |------|------|
            | 仿真引擎 | ✅ 就绪 |
            | 神经网络 | ⏳ 待连接 |
            | 市场数据 | ✅ 已加载 |
            
            ---
            
            > 💡 **提示:** 系统将自动加载近3年的上证指数历史数据作为仿真起点。
            """)


# --- Agent fMRI 标签页 ---
with tab2:
    st.subheader("🧠 Agent 心理核磁共振 (fMRI)")
    st.markdown("""
    <div style="background: rgba(168, 85, 247, 0.1); border-left: 4px solid #a855f7; padding: 10px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 0.9em; color: #e0e0e0;">
            <b>🔍 核心观察点：</b> 观察智能体在面临亏损时的“认知失调”现象。当情绪分数剧烈下降时，智能体通常会从理性的 System 2 转向恐慌驱动的 System 1。
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("点击任意 Agent 查看其完整思维链")

    if ctrl:
        # Agent 列表
        col_list, col_detail = st.columns([1, 2])

        with col_list:
            st.markdown("### 智能体列表")

            for agent in ctrl.model.population.smart_agents[:20]:  # 显示前20个
                agent_id = agent.id

                # 获取该 Agent 的情绪
                emotion = 0.0
                if agent_id in DeepSeekBrain.thought_history:
                    history = DeepSeekBrain.thought_history[agent_id]
                    if history:
                        emotion = history[-1].emotion_score

                # 创建可点击的按钮
                if st.button(
                    f"{get_emotion_icon(emotion)} {agent_id}",
                    key=f"agent_{agent_id}",
                    use_container_width=True
                ):
                    st.session_state.selected_agent = agent_id

        with col_detail:
            st.markdown("### 思维链详情")

            selected = st.session_state.selected_agent

            if selected and selected in DeepSeekBrain.thought_history:
                history = DeepSeekBrain.thought_history[selected]

                if history:
                    latest = history[-1]

                    # 情绪仪表盘
                    st.markdown(f"""
                    <div style="background: #161b22; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <div style="font-size: 18px; font-weight: bold;">
                            {get_emotion_icon(latest.emotion_score)} {selected}
                        </div>
                        <div style="margin-top: 10px;">
                            <span style="color: #888;">情绪分数:</span>
                            <span style="color: {get_emotion_color(latest.emotion_score)}; font-size: 20px; font-weight: bold;">
                            {latest.emotion_score:+.2f}
                            </span>
                        </div>
                        <div style="margin-top: 5px; color: #b0b0bb;">
                            状态: {latest.market_context.get('emotional_state', 'Unknown')}
                            <span style="font-size: 12px; color: #666; margin-left: 10px;">
                                (社交信号: {latest.market_context.get('social_signal', 'N/A')})
                            </span>
                        </div>
                        <div style="margin-top: 5px; color: #888;">
                            最后更新: {datetime.fromtimestamp(latest.timestamp).strftime('%H:%M:%S')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 决策显示
                    st.markdown("**最终决策:**")
                    st.json(latest.decision)

                    # 完整思维链
                    st.markdown("**完整思维链 (CoT):**")
                    st.markdown(f"""
                    <div class="reasoning-box" style="height: 400px;">
                        {latest.reasoning_content.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)

                    # 历史记录
                    if len(history) > 1:
                        st.markdown("**历史思维记录:**")
                        for i, record in enumerate(reversed(history[:-1])):
                            with st.expander(f"记录 {len(history) - i - 1}: {get_emotion_icon(record.emotion_score)} {record.decision.get('action', 'HOLD')}"):
                                st.text(record.reasoning_content[:500] + "..." if len(record.reasoning_content) > 500 else record.reasoning_content)
                else:
                    st.info("该 Agent 尚未产生思维记录")
            else:
                st.info("👈 请从左侧选择一个 Agent 查看详情")
    else:
        st.warning("请先启动仿真系统")

# --- 以下标签页仅在非回测模式下显示 ---
if not st.session_state.backtest_mode:

    # --- 辩论室标签页 ---
    with tab_debate:
        st.subheader("⚔️ Agent 内心辩论室")
        st.markdown("""
        <div style="background: rgba(45, 212, 191, 0.1); border-left: 4px solid #2dd4bf; padding: 10px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.9em; color: #e0e0e0;">
                <b>🔍 核心观察点：</b> 辩论室揭示了自洽决策的形成过程。关注空头（Bear）与多头（Bull）如何通过事实交换达成最终共识，这对防止单一维度的“幻觉”决策至关重要。
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("*观察 Agent 的 Bull vs Bear 内心对抗过程*")

        if ctrl:
            # 获取所有具有辩论能力的 Agent，即使他们还没有辩论记录
            debate_agents = []
            for agent in ctrl.model.population.smart_agents:
                # 兼容不同导入方式的类型检查
                if (hasattr(agent, 'brain') and "DebateBrain" in str(type(agent.brain))) or agent.id.startswith("Debate_"):
                    debate_agents.append(agent.id)
            
            # 补齐只存在于 history 里的 agent
            for aid in DebateBrain.debate_history.keys():
                if aid not in debate_agents:
                    debate_agents.append(aid)

            if debate_agents:
                col_d1, col_d2 = st.columns([1, 3])

                with col_d1:
                    st.markdown("### 🎭 Agent 列表")
                    selected_debate_agent = st.selectbox(
                        "选择 Agent",
                        debate_agents,
                        key="debate_agent_select"
                    )

                with col_d2:
                    if selected_debate_agent:
                        debates = DebateBrain.debate_history.get(selected_debate_agent, [])

                        if debates:
                            latest_debate = debates[-1]

                            # 辩论头信息
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                                        padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                                <h4 style="margin: 0; color: #00d4ff;">📋 最新辩论记录</h4>
                                <p style="color: #888; margin: 5px 0;">
                                    时间: {datetime.fromtimestamp(latest_debate.timestamp).strftime('%H:%M:%S')} | 
                                    风控: {'✅ 通过' if latest_debate.risk_approval else '❌ 否决'}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            # 辩论内容
                            for msg in latest_debate.debate_rounds:
                                if msg.role == DebateRole.BULL:
                                    bg_color = "#0d2818"
                                    border_color = "#00ff88"
                                    icon = "🐂"
                                    role_name = "牛牛 (看多派)"
                                elif msg.role == DebateRole.BEAR:
                                    bg_color = "#2d0d0d"
                                    border_color = "#ff4444"
                                    icon = "🐻"
                                    role_name = "空空 (看空派)"
                                else:
                                    bg_color = "#1a1a2e"
                                    border_color = "#4a4a6a"
                                    icon = "🛡️"
                                    role_name = "风控经理"

                                st.markdown(f"""
                                <div style="background: {bg_color}; border-left: 3px solid {border_color}; 
                                            padding: 12px; margin: 8px 0; border-radius: 5px;">
                                    <div style="font-weight: bold; color: {border_color}; margin-bottom: 5px;">
                                        {icon} {role_name}
                                    </div>
                                    <div style="color: #e0e0e0; line-height: 1.6;">
                                        {msg.content}
                                    </div>
                                    <div style="color: #666; font-size: 12px; margin-top: 5px;">
                                        情绪分数: {msg.emotion_score:+.2f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            # 最终决策
                            st.markdown("### 📝 最终决策")
                            st.json(latest_debate.final_decision)

                            # 历史辩论
                            if len(debates) > 1:
                                with st.expander(f"📜 历史辩论记录 ({len(debates) - 1} 条)"):
                                    for i, d in enumerate(reversed(debates[:-1])):
                                        st.markdown(f"**辩论 #{len(debates) - i - 1}** - {datetime.fromtimestamp(d.timestamp).strftime('%H:%M:%S')}")
                                        st.text(f"决策: {d.final_decision.get('action', 'N/A')} | 风控: {'通过' if d.risk_approval else '否决'}")
                                        st.markdown("---")
                        else:
                            st.info("该 Agent 尚无辩论记录")
            else:
                st.info("💡 使用 DebateBrain 的 Agent 运行后，辩论记录将显示在此处")
        else:
            st.warning("请先启动仿真系统")

    # --- 监管沙盒标签页 ---
    with tab_reg:
        st.subheader("🏛️ 监管沙盒")
        st.markdown("""
        <div style="background: rgba(244, 63, 94, 0.1); border-left: 4px solid #f43f5e; padding: 10px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.9em; color: #e0e0e0;">
                <b>🔍 核心观察点：</b> 监管干预的滞后性与有效性。观察国家队入场释放的预期信号是否能有效遏制 CSAD（羊群效应）的持续走高。
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("*模拟中国 A 股特色监管机制*")

        # 初始化监管模块（如果尚未初始化）
        if 'regulatory_module' not in st.session_state:
            st.session_state.regulatory_module = RegulatoryModule()

        reg = st.session_state.regulatory_module

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("### 🛡️ 国家稳定基金")
            fund_status = reg.stability_fund.get_status_report()

            st.metric("可用资金", fund_status["可用资金"])
            st.metric("已投入资金", fund_status["已投入资金"])
            st.metric("干预次数", fund_status["干预次数"])

            # 干预历史
            if reg.stability_fund.intervention_history:
                st.markdown("**近期干预:**")
                for intervention in reg.stability_fund.intervention_history[-3:]:
                    st.markdown(f"""
                    <div style="background: #1a2e1a; padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <div style="color: #00ff88;">💰 {intervention['capital']/1e8:.0f} 亿元</div>
                        <div style="color: #888; font-size: 12px;">
                            恐慌指数: {intervention['panic_level']:.2f} | 
                            价格: ¥{intervention['price']:.2f}
                        </div>
                        <div style="color: #ccc; margin-top: 5px; font-size: 13px;">
                            "{intervention['statement']}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("国家队尚未出手")

        with col_r2:
            st.markdown("### ⚡ 熔断机制")
            breaker = reg.circuit_breaker

            if breaker.is_halted:
                st.error(f"🔴 熔断中 (等级 {breaker.halt_level})")
            else:
                st.success("🟢 交易正常")

            st.metric("历史熔断次数", len(breaker.halt_history))

            # 熔断历史
            if breaker.halt_history:
                st.markdown("**熔断记录:**")
                for h in breaker.halt_history[-3:]:
                    level_text = "一级" if h['level'] == 1 else "二级"
                    st.markdown(f"- Tick {h['tick']}: {level_text}熔断 ({h['price_change']:+.2%})")

        st.markdown("---")
        st.markdown("### 📊 程序化交易监控")

        # 违规统计
        violations = reg.trading_regulator.violations
        col_v1, col_v2, col_v3 = st.columns(3)

        with col_v1:
            st.metric("监控 Agent 数", len(reg.trading_regulator.agent_stats))
        with col_v2:
            st.metric("今日违规次数", len(violations))
        with col_v3:
            suspended_count = sum(1 for s in reg.trading_regulator.agent_stats.values() 
                                  if s.current_restriction.value == 'suspended')
            st.metric("已停止交易", suspended_count)

        # 最近违规记录
        if violations:
            with st.expander("⚠️ 最近违规记录"):
                for v in violations[-5:]:
                    st.markdown(f"""
                    - **Agent {v['agent_id']}**: {v['type']} - {v['detail']}
                    """)

        # ====== PolicyManager 策略风洞控制台 ======
        st.markdown("---")
        st.markdown("### 🎛️ 策略风洞控制台")
        st.caption("实时调整监管策略参数，观察对市场微观结构的影响")

        if ctrl:
            # 获取当前策略状态
            policy_status = ctrl.model.get_policy_status()

            col_p1, col_p2 = st.columns(2)

            with col_p1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                            padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #ff6b6b;">⚡ 动态熔断机制</h4>
                    <p style="color: #888; font-size: 12px; margin: 5px 0;">
                        当价格偏离基准超过阈值时，自动暂停交易
                    </p>
                </div>
                """, unsafe_allow_html=True)

                cb_active = st.toggle(
                    "启用熔断",
                    value=policy_status["circuit_breaker"]["active"],
                    key="policy_cb_active"
                )

                cb_threshold = st.slider(
                    "熔断阈值 (%)",
                    min_value=1, max_value=20,
                    value=int(policy_status["circuit_breaker"]["threshold"] * 100),
                    step=1,
                    key="policy_cb_threshold",
                    help="价格偏离前收盘价的百分比阈值"
                )

                # Apply changes
                ctrl.model.set_policy("circuit_breaker", "active", cb_active)
                ctrl.model.set_policy("circuit_breaker", "threshold_pct", cb_threshold / 100.0)

                # Status indicator
                if policy_status["circuit_breaker"]["is_halted"]:
                    st.error("🔴 市场已熔断 — 订单将被拒绝")
                else:
                    st.success(f"🟢 市场正常 — 阈值 ±{cb_threshold}%")

            with col_p2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1a2e1a 0%, #16213e 100%); 
                            padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #4DA6FF;">💰 交易税 (印花税)</h4>
                    <p style="color: #888; font-size: 12px; margin: 5px 0;">
                        每笔成交按成交额收取印花税，影响交易成本与流动性
                    </p>
                </div>
                """, unsafe_allow_html=True)

                tax_active = st.toggle(
                    "启用交易税",
                    value=policy_status["transaction_tax"]["active"],
                    key="policy_tax_active"
                )

                tax_rate = st.slider(
                    "税率 (‰)",
                    min_value=0.0, max_value=10.0,
                    value=float(policy_status["transaction_tax"]["rate"] * 1000),
                    step=0.1,
                    key="policy_tax_rate",
                    help="每笔成交额的千分比税率 (当前A股印花税为 1‰)"
                )

                # Apply changes
                ctrl.model.set_policy("tax", "active", tax_active)
                ctrl.model.set_policy("tax", "rate", tax_rate / 1000.0)

                # Display
                st.metric("当前税率", f"{tax_rate:.1f}‰")
                if tax_rate > 1.0:
                    st.warning(f"⚠️ 税率高于基准 (1‰)，可能抑制流动性")
                elif tax_rate < 1.0 and tax_active:
                    st.info(f"💡 税率低于基准 (1‰)，可能刺激交易")
        else:
            st.info("💡 请先启动仿真系统，策略控制台将在仿真运行时可用")

    # --- 行为金融标签页 ---
    with tab_behavior:
        st.subheader("📊 行为金融量化面板")
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 10px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.9em; color: #e0e0e0;">
                <b>🔍 核心观察点：</b> 羊群效应强度。CSAD 值的显著偏离通常预示着市场进入了“无脑跟随”阶段，这往往是崩盘或报复性反弹的前兆。
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("*用数学量化人性偏差*")

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            st.markdown("### 📈 前景理论计算器")

            gain = st.slider("盈亏百分比", -50, 50, 0, 1, key="prospect_gain")
            loss_aversion = st.slider("损失厌恶系数 (λ)", 1.0, 4.0, 2.25, 0.1, key="prospect_lambda")

            utility = prospect_utility(gain / 100, loss_aversion=loss_aversion)

            # 可视化
            utility_color = "#00ff88" if utility >= 0 else "#ff4444"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #1a1a2e; border-radius: 10px;">
                <div style="color: #888;">心理效用值</div>
                <div style="font-size: 48px; color: {utility_color}; font-weight: bold;">
                    {utility:+.1f}
                </div>
                <div style="color: #666; font-size: 12px; margin-top: 10px;">
                    盈利 {gain}% 带来的心理感受 (λ={loss_aversion})
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 说明
            st.markdown("""
            > **前景理论** (Kahneman & Tversky):
            > - 人们对损失的痛苦是收益快乐的 2.25 倍
            > - 这解释了为什么投资者常常"死扛亏损"
            """)

        with col_b2:
            st.markdown("### 🐑 羊群效应检测")

            # 获取实时 CSAD 数据
            if 'csad_history' in st.session_state and st.session_state.csad_history:
                csad_data = st.session_state.csad_history[-20:]

                # 绘制 CSAD 趋势图
                fig_csad = go.Figure()
                fig_csad.add_trace(go.Scatter(
                    y=csad_data,
                    mode='lines+markers',
                    name='CSAD',
                    line=dict(color='#00d4ff', width=2)
                ))
                fig_csad.update_layout(
                    title="横截面绝对偏差 (CSAD)",
                    template="plotly_dark",
                    height=250,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_csad, use_container_width=True)

                # 羊群强度
                latest_csad = csad_data[-1] if csad_data else 0.02
                market_return = st.session_state.market_history[-1].get('change_pct', 0) if st.session_state.market_history else 0

                herd_intensity = herding_intensity(latest_csad, market_return)

                herd_color = "#ff4444" if herd_intensity > 0.5 else "#ffaa00" if herd_intensity > 0.2 else "#00ff88"
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background: #1a1a2e; border-radius: 10px;">
                    <div style="color: #888;">羊群效应强度</div>
                    <div style="font-size: 36px; color: {herd_color}; font-weight: bold;">
                        {herd_intensity:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("启动仿真后将显示 CSAD 走势图")

            st.markdown("""
            > **羊群效应检测**:
            > - CSAD 下降 + 市场大涨/大跌 = 羊群行为
            > - γ₂ < 0 表示存在显著羊群效应
            """)

    # --- 量化群体标签页 ---
    with tab_quant:
        st.subheader("🤖 量化群体监控")
        st.markdown("""
        <div style="background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; padding: 10px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.9em; color: #e0e0e0;">
                <b>🔍 核心观察点：</b> 量化共识及其导致的流动性特征。高频止损策略的同步性会加剧市场的“核按钮”效应，导致价格非连续性暴跌。
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.quant_manager and st.session_state.quant_manager.groups:
            # 系统风险检测
            risk = st.session_state.quant_manager.detect_systemic_risk()

            if risk['warning']:
                st.warning(risk['warning'])

            # 显示各群体状态
            for group_id, group in st.session_state.quant_manager.groups.items():
                with st.expander(f"📊 {group.strategy_name} ({len(group.agents)} Agents)", expanded=True):
                    col_g1, col_g2, col_g3 = st.columns(3)

                    with col_g1:
                        st.metric("一致性", f"{group.action_consensus:.2%}")
                    with col_g2:
                        st.metric("抛售压力", f"{group.sell_pressure:.2%}")
                    with col_g3:
                        action_label = {
                            'PANIC_SELL': '🔴 集体抛售',
                            'SELL': '🟠 倾向卖出',
                            'BUY': '🟢 倾向买入',
                            'MIXED': '⚪ 分歧'
                        }.get(group.collective_action, '⚪ 待激活')
                        st.metric("群体行为", action_label)

                    # 情绪分布
                    emotion_dist = group.get_emotion_distribution()
                    st.markdown(f"""
                    **情绪分布:** 
                    🟢 贪婪 {emotion_dist['greedy']} / 
                    ⚪ 中性 {emotion_dist['neutral']} / 
                    🔴 恐惧 {emotion_dist['fearful']}
                    """)
        else:
            st.info("尚未创建量化群体。请在侧边栏\"量化群体设置\"中创建。")
