import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime

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
    
    ctrl = st.session_state.get('controller')
    if not ctrl and st.session_state.demo_phase > 0:
        st.warning("⚠️ 仿真系统尚未启动，正在展示离线占位数据。请在左侧启动仿真以获取实时数据。")
    
    # Render view based on phase
    if st.session_state.demo_phase == 1:
        render_phase1(ctrl)
    elif st.session_state.demo_phase == 2:
        render_phase1(ctrl)
        st.markdown("---")
        render_phase2(ctrl)
    elif st.session_state.demo_phase == 3:
        render_phase1(ctrl)
        st.markdown("---")
        render_phase2(ctrl)
        st.markdown("---")
        render_phase3(ctrl)
    elif st.session_state.demo_phase == 4:
        render_phase1(ctrl)
        st.markdown("---")
        render_phase2(ctrl)
        st.markdown("---")
        render_phase3(ctrl)
    else:
        st.info("👈 请点击上方按钮进入演示阶段。")

def render_phase1(ctrl):
    st.markdown("### 阶段一：宏观注入与机构拆解")
    st.markdown("> **解说核心**: 系统注入极严厉政策，Policy Committee瞬间激活。量化节点进行多轮SOP辩论，精准纠正宏观专家的幻觉逻辑。")
    
    col_input, col_log = st.columns([1, 2])
    
    policy_info = st.session_state.get('policy_analysis')
    policy_text = policy_info['text'] if policy_info else "等待注入突发利空政策..."
    
    with col_input:
        st.markdown("**主控台界面 - 极值假设**")
        st.text_area("实时政策指令池：", value=policy_text, height=100, disabled=True)
        if policy_info:
            st.success("状态: 平稳拟合态 -> 已注入", icon="✅")
        else:
            st.info("状态: 平稳拟合态", icon="ℹ️")
        
    with col_log:
        st.markdown("**终端控制台 - 实时流式日志 (政策委员会 & 辩论室)**")
        
        from agents.debate_brain import DebateBrain, DebateRole
        
        debate_agents = []
        if ctrl and hasattr(ctrl, 'model') and ctrl.model and hasattr(ctrl.model, 'population') and ctrl.model.population:
            for agent in ctrl.model.population.smart_agents:
                if "DebateBrain" in str(type(agent.brain)) or agent.id.startswith("Debate_"):
                    debate_agents.append(agent.id)
                    
        if hasattr(DebateBrain, 'debate_history'):
            for aid in DebateBrain.debate_history.keys():
                if aid not in debate_agents:
                    debate_agents.append(aid)
        
        if debate_agents:
            # 找到最新的辩论记录
            latest_debate = None
            for agent in debate_agents:
                if hasattr(DebateBrain, 'debate_history') and agent in DebateBrain.debate_history:
                    debates = DebateBrain.debate_history[agent]
                    if debates and (not latest_debate or debates[-1].timestamp > latest_debate.timestamp):
                         latest_debate = debates[-1]
            
            if latest_debate:
                html_logs = f"""
                <div style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; height: 300px; overflow-y: scroll; font-family: 'Consolas', monospace; font-size: 13px; color: #c9d1d9;">
                    <div style="color: #4DA6FF;">[{datetime.fromtimestamp(latest_debate.timestamp).strftime('%H:%M:%S')}] Policy stream received. Awakening Debate Room...</div>
                """
                
                for msg in latest_debate.debate_rounds:
                    if msg.role == DebateRole.BULL:
                        color = "#00ff88"
                        role_name = "牛牛 (看多派)"
                    elif msg.role == DebateRole.BEAR:
                        color = "#ff4444"
                        role_name = "空空 (看空派)"
                    else:
                        color = "#4DA6FF"
                        role_name = "风控经理"
                        
                    html_logs += f'<div style="color: {color}; margin-top: 8px;">[Agent: {role_name}] (Mood: {msg.emotion_score:+.2f}) {msg.content}</div>'
                
                html_logs += f"""
                    <br>
                    <div style="background: rgba(255, 59, 48, 0.1); border: 1px solid #FF3B30; padding: 8px; color: #FF3B30; font-weight: bold;">
                        🎯 委员会共识达成: 决定行动 {latest_debate.final_decision.get('action', 'HOLD')}
                    </div>
                </div>
                """
                st.markdown(html_logs, unsafe_allow_html=True)
            else:
                members_str = ", ".join(debate_agents)
                html = f"""
                <div style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; height: 300px; overflow-y: scroll; font-family: 'Consolas', monospace; font-size: 13px; color: #c9d1d9;">
                    <div style="color: #4DA6FF;">[Sys] 政策委员会 (Policy Committee) 集结完毕。成员: {members_str}</div>
                    <div style="color: #FFD700; margin-top: 8px;">[Agent: System] 当前系统评估处于平稳态，委员会随时待命，等待政策输入...</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
            # Fallback mock if completely disconnected
            html = """
            <div style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; height: 300px; overflow-y: scroll; font-family: 'Consolas', monospace; font-size: 13px; color: #c9d1d9;">
                <div style="color: #4DA6FF;">[Sys] 仿真尚未启动或未发现委员会成员。</div>
                <div style="color: #888;">[Agent: System] 休眠中...</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

def render_phase2(ctrl):
    st.markdown("### 阶段二：网络传染与微观异动")
    st.markdown("> **解说核心**: 中心机构超级节点响应信号转红，恐慌文本顺着图谱涟漪般扩散，散户内部System 1防线被击穿。")
    
    col_graph, col_fmri = st.columns([2, 1])
    
    with col_graph:
        st.markdown("**实时社交图谱拓扑**")
        try:
            from agents.brain import DeepSeekBrain
            import networkx as nx
            import numpy as np
            
            if ctrl and hasattr(ctrl.model, 'social_graph'):
                G = ctrl.model.social_graph.graph
            else:
                # Mock graph
                np.random.seed(42)
                G = nx.barabasi_albert_graph(100, 2)
                
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
                line=dict(width=0.5, color='rgba(150, 150, 150, 0.4)'),
                hoverinfo='none',
                mode='lines')
            
            node_x = []
            node_y = []
            colors = []
            sizes = []
            texts = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                color = '#00d4ff' # Default Blue
                size = 10
                text = f"Agent {node}"
                
                agent = ctrl.model.population.get_agent_by_id(node) if ctrl else None
                if agent:
                    # Institutional agents are larger
                    agent_type = getattr(agent, 'agent_type', 'RETAIL')
                    if agent_type != 'RETAIL':
                        size = 20
                        
                    # Get real emotion if available
                    if hasattr(DeepSeekBrain, 'thought_history') and node in DeepSeekBrain.thought_history:
                        history = DeepSeekBrain.thought_history[node]
                        if history:
                            emotion = history[-1].emotion_score
                            if emotion < -0.3:
                                color = '#FF3B30' # Red
                            elif emotion > 0.3:
                                color = '#34C759' # Green
                            else:
                                color = '#FFD60A' # Yellow
                            text += f"<br>Emotion: {emotion:+.2f}"
                            
                colors.append(color)
                sizes.append(size)
                texts.append(text)
                
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
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                     )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"图谱渲染失败: {str(e)}")
            
    with col_fmri:
        st.markdown("**典型恐慌节点监测 (fMRI)**")
        
        # Try finding the most panicked agent
        most_panicked_agent = None
        min_emotion = 0
        from agents.brain import DeepSeekBrain
        if hasattr(DeepSeekBrain, 'thought_history'):
             for agent_id, history in DeepSeekBrain.thought_history.items():
                 if history and history[-1].emotion_score < min_emotion:
                     min_emotion = history[-1].emotion_score
                     most_panicked_agent = (agent_id, history[-1])
        
        if most_panicked_agent:
            agent_id, record = most_panicked_agent
            st.markdown(f"""
            <div style="background: rgba(255, 59, 48, 0.1); border: 1px solid #FF3B30; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px;">
                <div style="font-size: 30px;">🔴</div>
                <div style="font-weight: bold; color: #e0e0e0; font-size: 18px; margin-top:5px;">Agent {agent_id}</div>
                <div style="color: #FF3B30; margin-top: 10px; font-weight: bold;">[极度恐慌状态]</div>
                <div style="color: #666; font-size: 13px; margin-top: 5px;">情绪因子: {record.emotion_score:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            html_fmri = f"""
            <div class="reasoning-box" style="height: 180px; border-color: #FF3B30; background: #161b22; font-family: 'Consolas', monospace; font-size: 13px; color: #c9d1d9; overflow-y:scroll;">
                <span style="color: #FF3B30;">[{datetime.fromtimestamp(record.timestamp).strftime('%H:%M:%S')}] (Emotion Engine) 情绪防线防破！发现强悲观社交信号。</span><br>
                <span style="color: #888;">{record.reasoning_content.replace(chr(10), '<br>')}</span><br>
                <span style="color: #FF3B30; font-weight: bold; background: rgba(255,59,48,0.2); display: inline-block; padding: 2px;">(Execution) 最终操作: {record.decision.get('action')}</span>
            </div>
            """
            st.markdown(html_fmri, unsafe_allow_html=True)
        else:
             html_fmri = """
            <div class="reasoning-box" style="height: 150px; border-color: #34C759; background: #161b22; font-family: 'Consolas', monospace; font-size: 13px;">
                <span style="color: #888;">(Social Graph Polling) 读取社交图谱时间线：环境安全检查...</span><br>
                <span style="color: #34C759;">(System 2 Alert) 暂未发现广泛恐慌源。</span><br>
                <span style="color: #c9d1d9;">(Execution) 维持现有策略观望。</span>
            </div>
            """
             st.markdown(html_fmri, unsafe_allow_html=True)

def render_phase3(ctrl):
    st.markdown("### 阶段三：订单撮合与宏观崩盘")
    st.markdown("> **解说核心**: 左侧LOB深度图被巨量卖单吞噬，宏观K线大阴线垂直俯冲触发熔断。")
    
    col_lob, col_kline = st.columns([1.2, 2])
    
    with col_lob:
        st.markdown("**实时限价订单簿深度 (LOB)**")
        
        if ctrl and hasattr(ctrl, 'market'):
             depth = ctrl.market.engine.get_order_book_depth(5)
             bids = depth.get('bids', [])
             asks = depth.get('asks', [])
             
             # The result from get_order_book_depth is already sorted (bids desc, asks asc)
             # and represents a list of dicts: {"price": float, "qty": int}
             
             html_lob = """
            <div style="background: #111; padding: 10px; border-radius: 8px; font-family: monospace; font-size: 13px; height: 350px; overflow-y:auto;">
                <div style="text-align: center; color: #34C759; font-weight: bold; background: rgba(52, 199, 89, 0.1); padding: 5px; margin-bottom: 5px; border-bottom: 1px solid #34C759;">
                    卖盘深度 (Ask)
                </div>
                <table style="width: 100%; color: #34C759; text-align: right;">
            """
             
             for i, order in enumerate(asks[:5]):
                 html_lob += f'<tr style="background: rgba(52, 199, 89, {0.8 - i*0.1});"><td>卖{i+1}</td><td>{order["price"]:.2f}</td><td>{order["qty"]}</td></tr>'
                 
             if not asks:
                 html_lob += '<tr><td>无显著卖盘</td></tr>'
                 
             html_lob += """
                </table>
                <div style="height: 2px; background: #666; margin: 10px 0;"></div>
                <table style="width: 100%; color: #FF3B30; text-align: right;">
             """
             
             for i, order in enumerate(bids[:5]):
                 html_lob += f'<tr style="background: rgba(255, 59, 48, {0.8 - i*0.1});"><td>买{i+1}</td><td>{order["price"]:.2f}</td><td>{order["qty"]}</td></tr>'

             if not bids:
                 html_lob += '<tr><td>流动性干涸 / 无买盘买单</td></tr>'
                 
             html_lob += """
                </table>
                <div style="text-align: center; color: #FF3B30; font-weight: bold; background: rgba(255, 59, 48, 0.05); padding: 5px; margin-top: 5px; border-top: 1px solid #FF3B30;">
                    买盘深度 (Bid)
                </div>
            </div>
             """
             st.markdown(html_lob, unsafe_allow_html=True)
             
        else:
             st.info("数据获取中，如果长时间没变化请确保系统正在运行。")
             
        
    with col_kline:
        st.markdown("**宏观走势：崩盘与熔断**")
        
        if st.session_state.get('market_history'):
            history = st.session_state.market_history
            recent = history[-60:] # Show last 60 candles
            
            df = pd.DataFrame(recent)
            df['color'] = np.where(df['close'] >= df['open'], '#FF3B30', '#34C759')
            
            fig_k = go.Figure(data=[go.Candlestick(
                x=df['time'], 
                open=df['open'], high=df['high'], 
                low=df['low'], close=df['close'], 
                increasing_line_color='#FF3B30', 
                decreasing_line_color='#34C759', 
                increasing_fillcolor='#FF3B30', 
                decreasing_fillcolor='#34C759'
            )])
            
            fig_k.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,10,10,0.8)', height=350, margin=dict(t=10, b=10, l=10, r=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_k, use_container_width=True)
        else:
            st.info("暂无行情数据，请等候市场第一笔交易发生。")

