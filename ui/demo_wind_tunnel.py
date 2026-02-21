import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx

def render_demo_tab():
    st.markdown("## 🌪️ 沙箱风洞 —— 实盘推演")
    st.markdown("这一专门展示页用于完整展示极端利空政策下从宏观注入到微观传染，最终导致“恐慌蔓延”的市场崩盘动线。请各位评委跟随讲解人的节奏共同见证。")
    
    # Initialize state
    if "demo_phase" not in st.session_state:
        st.session_state.demo_phase = 0
        
    # Control Panel
    st.markdown("#### 面板控制 (Control Panel)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("⏹️ 重置演示", use_container_width=True):
            st.session_state.demo_phase = 0
    with col2:
        if st.button("1️⃣ 政策与辩论", use_container_width=True):
            st.session_state.demo_phase = 1
            st.rerun()
    with col3:
        if st.button("2️⃣ 网络传染", use_container_width=True):
            st.session_state.demo_phase = 2
            st.rerun()
    with col4:
        if st.button("3️⃣ 撮合与崩盘", use_container_width=True):
            st.session_state.demo_phase = 3
            st.rerun()
    with col5:
        if st.button("▶️ 完整展示态", use_container_width=True):
            st.session_state.demo_phase = 4
            st.rerun()
            
    st.markdown("---")
    
    # Render view based on phase
    if st.session_state.demo_phase == 1:
        render_phase1()
    elif st.session_state.demo_phase == 2:
        render_phase1()
        st.markdown("---")
        render_phase2()
    elif st.session_state.demo_phase == 3:
        render_phase1()
        st.markdown("---")
        render_phase2()
        st.markdown("---")
        render_phase3()
    elif st.session_state.demo_phase == 4:
        render_phase1()
        st.markdown("---")
        render_phase2()
        st.markdown("---")
        render_phase3()
    else:
        st.info("👈 请点击上方按钮进入演示阶段。")

def render_phase1():
    st.markdown("### 阶段一：宏观注入与机构拆解 (00:00 - 00:40)")
    st.markdown("> **解说核心**: 系统注入极严厉政策，Policy Committee瞬间激活。量化节点进行多轮SOP辩论，精准纠正宏观专家的幻觉逻辑。")
    
    col_input, col_log = st.columns([1, 2])
    
    with col_input:
        st.markdown("**主控台界面 - 极值假设**")
        st.text_area("突发利空政策输入：", value="即日起全面禁止机构高频量化交易接口，并单边上调机构印花税。", height=100, disabled=True)
        st.button("⚡ 定点投放至网络", disabled=True)
        st.success("状态: 平稳拟合态 -> 已注入", icon="✅")
        
    with col_log:
        st.markdown("**终端控制台 - 实时流式日志 (政策委员会)**")
        # Simulated streaming log
        html = """
        <div style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; height: 300px; overflow-y: scroll; font-family: 'Consolas', monospace; font-size: 13px; color: #c9d1d9;">
            <div style="color: #4DA6FF;">[Sys] Policy stream received. Awakening Policy Committee (Macro, Quant, Risk)...</div>
            <div style="color: #888;">[Agent: Macro_Expert] 正在切分文本... 指令转化为[限制高频, 上调印花税]。初步评估: 政策意图在于限制过度投机。短期内市场换手率将下降，但对核心资产流动性估算的影响处于中性可控区间。</div>
            <div style="color: #FF3B30; font-weight: bold; margin-top: 10px; margin-bottom: 5px;">[Agent: Quant_Analyst] (INTERRUPT) ⚠️ 修正幻觉逻辑！</div>
            <div style="color: #FFD60A; padding-left: 10px; border-left: 2px solid #FFD60A;">[Agent: Quant_Analyst] 宏观节点的流动性估算存在根本性错误。全面禁用高频接口 + 税率单边上调，将瞬间推高交易摩擦成本，直接击穿做市商(Market Maker)的容忍底线。这不会导致换手率缓降，而是会导致LOB双边深度在毫秒级别内部全部撤单！流动性是瞬间干涸！</div>
            <div style="color: #888; margin-top: 10px;">[Agent: Risk_Control] 交叉验证完毕。支持量化节点观点。系统性风险预警级别提升至 [CRITICAL]。</div>
            <br>
            <div style="background: rgba(255, 59, 48, 0.1); border: 1px solid #FF3B30; padding: 8px; color: #FF3B30; font-weight: bold;">
                🎯 委员会共识达成: 输出《一致性看空矩阵》(CONSENSUS: STRONG_BEARISH)
            </div>
            <div style="color: #4DA6FF; margin-top: 5px;">[Sys] 政策解译完成，生成致命利空信号，开始定点投放至超大节点集群。</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

def render_phase2():
    st.markdown("### 阶段二：网络传染与微观异动 (00:40 - 01:20)")
    st.markdown("> **解说核心**: 中心机构超级节点响应信号转红，恐慌文本顺着图谱涟漪般扩散，散户内部System 1防线被击穿。")
    
    # Using Plotly to simulate a network graph
    np.random.seed(42)
    G = nx.barabasi_albert_graph(250, 2)
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.4, color='rgba(255, 59, 48, 0.4)'), # Red rippling lines
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    colors = []
    sizes = []
    texts = []
    
    # Identify hubs (institutions)
    degrees = dict(G.degree())
    hubs = sorted(degrees, key=degrees.get, reverse=True)[:6]
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in hubs:
            colors.append('#FF3B30') # Infected (Red)
            sizes.append(28)
            texts.append("<b>超级节点 (机构巨鲸)</b><br>System 2 阈值触发<br>状态: INFECTED (红色恐慌传染区)<br>行动: 广播市价卖单文本")
        else:
            # Simulate widespread spreading radially
            if np.random.rand() > 0.4:
                colors.append('#ff6b6b') # Lighter red for infected retail
                sizes.append(10)
                texts.append("<b>底层散户 Agent</b><br>System 2: 读取大V联名看空言论<br>修改自身风险参数!<br>System 1: 从众反应，恐慌抛压积聚")
            else:
                colors.append('#34C759') # Some still green
                sizes.append(8)
                texts.append("底层散户 Agent<br>System 1: 启发模式<br>状态: 观望中立")
                
    nodes_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=texts,
        marker=dict(
            showscale=False,
            color=colors,
            size=sizes,
            line_width=1,
            line_color='rgba(255,255,255,0.8)'
        )
    )

    fig = go.Figure(data=[edges_trace, nodes_trace],
             layout=go.Layout(
                title=dict(text='Social Graph Contagion (SIR Force-Directed Model)', font=dict(size=16, color="#c9d1d9")),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
             )
    fig.update_layout(height=450)
    
    st.plotly_chart(fig, use_container_width=True)

def render_phase3():
    st.markdown("### 阶段三：订单撮合与宏观崩盘 (01:20 - 02:00)")
    st.markdown("> **解说核心**: 左侧LOB深度图被巨量绿色卖单吞噬，宏观K线大阴线垂直俯冲触发熔断。Agent的fMRI显示“交出带血的筹码”。")
    
    col_lob, col_kline = st.columns([1.2, 2])
    
    with col_lob:
        st.markdown("**实时限价订单簿深度 (LOB)**")
        # Custom LOB Display showing ask engulfing bid 
        html_lob = """
        <div style="background: #111; padding: 10px; border-radius: 8px; font-family: monospace; font-size: 13px; height: 350px;">
            <div style="text-align: center; color: #34C759; font-weight: bold; background: rgba(52, 199, 89, 0.1); padding: 5px; margin-bottom: 5px; border-bottom: 1px solid #34C759;">
                巨量 Ask (卖单) 如雪崩般涌入
            </div>
            <!-- In A-share, Ask is typically Green and Bid is Red. "巨量绿色卖单吞噬" matches this. -->
            <table style="width: 100%; color: #34C759; text-align: right;">
                <tr style="background: rgba(52, 199, 89, 0.3);"><td>卖五</td><td>2750.50</td><td>158,200</td></tr>
                <tr style="background: rgba(52, 199, 89, 0.4);"><td>卖四</td><td>2750.00</td><td>235,000</td></tr>
                <tr style="background: rgba(52, 199, 89, 0.5);"><td>卖三</td><td>2749.50</td><td>489,100</td></tr>
                <tr style="background: rgba(52, 199, 89, 0.6);"><td>卖二</td><td>2748.00</td><td>820,000</td></tr>
                <tr style="background: rgba(52, 199, 89, 0.8); font-weight:bold;"><td>卖一</td><td>2745.00</td><td>1,500,000</td></tr>
            </table>
            <div style="height: 2px; background: #666; margin: 10px 0;"></div>
            <table style="width: 100%; color: #FF3B30; text-align: right; opacity: 0.5;">
                <tr><td>买一</td><td>2700.00</td><td>12,000</td></tr>
                <tr><td>买二</td><td>2695.00</td><td>8,500</td></tr>
                <tr><td>买三</td><td>2690.00</td><td>5,000</td></tr>
                <tr><td>买四</td><td>2680.00</td><td>2,000</td></tr>
                <tr><td>买五</td><td>2650.00</td><td>100</td></tr>
            </table>
            <div style="text-align: center; color: #FF3B30; font-weight: bold; background: rgba(255, 59, 48, 0.05); padding: 5px; margin-top: 5px; border-top: 1px solid #FF3B30;">
                BID价买方流动性瞬间完全干涸!
            </div>
        </div>
        """
        st.markdown(html_lob, unsafe_allow_html=True)
        
    with col_kline:
        st.markdown("**宏观走势：崩盘与熔断**")
        dates = pd.date_range("2026-02-21 09:30", periods=20, freq="1T")
        o = np.full(20, 3000)
        h = np.full(20, 3005)
        l = np.full(20, 2980)
        c = np.full(20, 2990)
        
        # Simulate flash crash (A-share: Drop implies Close < Open -> Green candlestick (Decrease is Green))
        # Wait, the plot colors: Green if C<O, Red if C>=O.
        o[-6:] = [2980, 2920, 2850, 2800, 2750, 2710]
        h[-6:] = [2980, 2920, 2850, 2800, 2750, 2710]
        c[-6:] = [2920, 2850, 2800, 2750, 2710, 2700]
        l[-6:] = [2910, 2840, 2790, 2740, 2700, 2700]

        fig_k = go.Figure(data=[go.Candlestick(x=dates, open=o, high=h, low=l, close=c, increasing_line_color='#FF3B30', decreasing_line_color='#34C759', increasing_fillcolor='#FF3B30', decreasing_fillcolor='#34C759')])
        
        fig_k.add_hline(y=2700, line_dash="dash", line_color="#FFD60A", line_width=2, annotation_text="跌停板 / 一级熔断触发 (-10%)", annotation_position="bottom right", annotation_font_color="#FFD60A", annotation_font_size=14)
        
        fig_k.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.8)', height=350, margin=dict(t=10, b=10, l=10, r=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_k, use_container_width=True)
        st.error("【系统警告】波动率异常聚集！复现史诗级闪电崩盘！推演结束。")

    st.markdown("---")
    st.markdown("### 🧠 行为脑核磁 (fMRI) 溯源面板 (个体微缩视角)")
    col_fmri1, col_fmri2 = st.columns([1, 4])
    with col_fmri1:
         st.markdown(f"""
        <div style="background: rgba(255, 59, 48, 0.1); border: 1px solid #FF3B30; padding: 15px; border-radius: 8px; text-align: center;">
            <div style="font-size: 30px;">🔴</div>
            <div style="font-weight: bold; color: #e0e0e0; font-size: 18px; margin-top:5px;">Retail Agent #8922</div>
            <div style="color: #FF3B30; margin-top: 10px; font-weight: bold;">[极度恐慌状态]</div>
            <div style="color: #666; font-size: 13px; margin-top: 5px;">情绪效用跌入冰点</div>
        </div>
        """, unsafe_allow_html=True)
    with col_fmri2:
         html_fmri = """
        <div class="reasoning-box" style="height: 150px; border-color: #FF3B30; background: #161b22; font-family: 'Consolas', monospace; font-size: 13px;">
            <span style="color: #888;">[14:15:32] (Social Graph Polling) 读取社交图谱时间线：环境安全检查...</span><br>
            <span style="color: #ffaa00;">[14:15:33] (System 2 Alert) ⚠️ 发现「朋友圈 60% 都在跑」！（超强悲观信号接收）</span><br>
            <span style="color: #FF3B30;">[14:15:33] (Emotion Engine) 情绪防线彻底破防！触发从众效应 (Conformity Threshold Exceeded)。情绪因子计算：-0.98</span><br>
            <span style="color: #c9d1d9;">[14:15:34] (Cognitive Override) 取消了原定市盈率(PE)分析：当前第一优先级任务转变：止损保命。</span><br>
            <span style="color: #FF3B30; font-weight: bold; background: rgba(255,59,48,0.2); display: inline-block; padding: 2px;">[14:15:35] (Execution) 强制抛出带血的筹码。生成市价卖出（砸盘止损单）指令，全仓撤离！📉</span>
        </div>
        """
         st.markdown(html_fmri, unsafe_allow_html=True)
