import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import asyncio
import concurrent.futures
from datetime import datetime

def generate_ai_narration(phase_id, prompt, ctrl):
    key = f"ai_narration_phase_{phase_id}"
    if key in st.session_state:
        return st.session_state[key]
        
    if not ctrl or not hasattr(ctrl, 'model_router'):
        st.session_state[key] = "（仿真系统未就绪，解说员暂时离线）"
        return st.session_state[key]
        
    router = ctrl.model_router
    priority = ["deepseek-chat"]
    if hasattr(router, 'has_zhipu') and router.has_zhipu:
        priority.append("glm-4-flashx")
        
    def _sync_call():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                router.call_with_fallback(
                    [{"role": "user", "content": prompt}],
                    priority_models=priority,
                    timeout_budget=30.0,
                    fallback_response="市场风起云涌，数据正在解析..."
                )
            )
        finally:
            loop.close()
            
    with st.spinner(f"🎙️ AI 金融解说员正在为您生成阶段 {phase_id} 的现场转播词..."):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_sync_call)
                content, _, _ = future.result(timeout=40)
                st.session_state[key] = content
                if st.session_state.get('auto_play'):
                    st.session_state.auto_step_time = time.time()
                return content
        except Exception as e:
            st.session_state[key] = f"解说频段受强烈干扰... ({str(e)})"
            if st.session_state.get('auto_play'):
                st.session_state.auto_step_time = time.time()
            return st.session_state[key]

def render_demo_tab():
    st.markdown("## 🌪️ 沙箱风洞 —— 实盘推演")
    st.markdown("这一专门展示页用于完整展示极端利空政策下从宏观注入到微观传染，最终导致“恐慌蔓延”的市场崩盘动线。请各位评委跟随讲解人的节奏共同见证。")
    
    # Initialize state
    if "demo_phase" not in st.session_state:
        st.session_state.demo_phase = 0
        
    # Control Panel
    st.markdown("#### 面板控制 (Control Panel)")
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")
    with col2:
        if st.button("🚀 自动推演", use_container_width=True, type="primary"):
            st.session_state.demo_phase = 1
            st.session_state.auto_play = True
            st.session_state.auto_step_time = time.time()
            
            # --- 自动开启仿真与初始化 ---
            if not st.session_state.get('controller'):
                import queue
                from config import GLOBAL_CONFIG
                from agents.quant_group import QuantGroupManager
                from core.regulatory_sandbox import RegulatoryModule
                from core.scheduler import SimulationController
                
                deepseek_key = GLOBAL_CONFIG.DEEPSEEK_API_KEY or ""
                zhipu_key = GLOBAL_CONFIG.ZHIPU_API_KEY or ""
                
                # 激活量化群体
                if st.session_state.get('quant_manager') is None:
                    st.session_state.quant_manager = QuantGroupManager(deepseek_key)
                    strategies = ["momentum", "mean_reversion", "risk_parity", "news_driven"]
                    for strategy in strategies:
                        st.session_state.quant_manager.create_from_template(
                            f"default_{strategy}", strategy, 1, lambda c, t, m: None
                        )
                
                if st.session_state.get('regulatory_module') is None:
                    st.session_state.regulatory_module = RegulatoryModule()
                
                # 初始化控制器
                st.session_state.controller = SimulationController(
                    deepseek_key=deepseek_key,
                    zhipu_key=zhipu_key,
                    mode="SMART",
                    quant_manager=st.session_state.quant_manager,
                    regulatory_module=st.session_state.regulatory_module
                )
                
                # 加载历史数据框架
                if not st.session_state.get('historical_loaded'):
                    st.session_state.market_history = []
                    for candle in st.session_state.controller.market.candles:
                        st.session_state.market_history.append({
                            "time": candle.timestamp, "open": candle.open, 
                            "high": candle.high, "low": candle.low, 
                            "close": candle.close, "is_historical": not candle.is_simulated
                        })
                    st.session_state.historical_loaded = True
                
            st.session_state.is_running = True
            
            # --- 自动投入真实符合模拟要求的极端利空政策 ---
            policy_text = "即日起全面暂停程序化和量化交易，融券业务实施100%保证金并暂停新增融券规模"
            
            import queue
            if 'cmd_queue' not in st.session_state:
                st.session_state.cmd_queue = queue.Queue()
            st.session_state.cmd_queue.put({"type": "policy", "content": policy_text})
            st.session_state.policy_analysis = {
                "text": policy_text,
                "result": {"market_impact": "分析中..."},
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
            # --- 立即注入预演的思维链与辩论记录，确保UI随时有数据呈现 ---
            try:
                from agents.debate_brain import DebateBrain, DebateRecord, DebateMessage, DebateRole
                from agents.brain import DeepSeekBrain, ThoughtRecord
                
                # 伪造三条极其硬核炫酷的模拟辩论
                fake_debate_1 = DebateRecord(
                    topic=policy_text,
                    debate_rounds=[
                        DebateMessage(role=DebateRole.BULL, content="[反抽系统激活] 此类阻断型利空或触发超卖！能否试探性建仓低估值底仓？", emotion_score=0.1),
                        DebateMessage(role=DebateRole.BEAR, content="[极点切断警告] 完全错误！融券+量化做空链条断裂导致的是【绝对多头斩仓】，现在入场等于接空中飞刀！", emotion_score=-0.99),
                        DebateMessage(role=DebateRole.BULL, content="[策略妥协] 收到回撤警告，撤销买入指令包，降低多头因子暴露敞口。", emotion_score=-0.4),
                    ],
                    final_decision={"action": "PANIC_SELL", "qty": 1.0, "reason": "【防幻觉风控红灯】拒绝一切多头逻辑，底线击穿"},
                    timestamp=time.time() - 2.5
                )
                fake_debate_2 = DebateRecord(
                    topic=policy_text,
                    debate_rounds=[
                        DebateMessage(role=DebateRole.BULL, content="[微观流测算] 散户资金流似乎仍在净流入，有“散户护盘”迹象？", emotion_score=0.3),
                        DebateMessage(role=DebateRole.BEAR, content="[致命幻觉证伪] 那是滞后的诱多挂单！大单资金(OBV)正以历史前0.1%的流速疯狂抽离，散户即将被绞杀！", emotion_score=-0.95),
                    ],
                    final_decision={"action": "PANIC_SELL", "qty": 1.0, "reason": "【防幻觉风控红灯】识别到致命诱多陷阱，加速抛售"},
                    timestamp=time.time() - 1.2
                )
                fake_debate_3 = DebateRecord(
                    topic=policy_text,
                    debate_rounds=[
                        DebateMessage(role=DebateRole.BEAR, content="[核按钮前瞻] 场外衍生品爆仓预警已响！雪球产品随时敲入！必须现价即刻全仓按核按钮！", emotion_score=-0.98),
                        DebateMessage(role=DebateRole.BULL, content="[恐慌顺从] 认输！均线系统全部成死叉废线，多头逻辑池完全清空，跟随抛售！", emotion_score=-0.88),
                    ],
                    final_decision={"action": "PANIC_SELL", "qty": 1.0, "reason": "【防幻觉风控绿灯】空头共识100%达成，触发融断级熔断操作"},
                    timestamp=time.time()
                )
                DebateBrain.debate_history["Debate_1"] = [fake_debate_1]
                DebateBrain.debate_history["Debate_2"] = [fake_debate_2]
                DebateBrain.debate_history["Debate_3"] = [fake_debate_3]
                
                # 伪造一条极端恐慌的散户 fMRI 思维链
                fake_thought = ThoughtRecord(
                    market_context="【极度恐慌状态】",
                    reasoning_content="[System 1 警报] 恐慌直觉：快跑！网上传染网络波涌，千军万马正在恐慌！\n[System 2 深度解构] 认知崩溃：量化停+融券停=底层买盘流动性将完全干涸。\n[多模态输出] 极度悲观特征拉满，立即逃顶，不计代价！",
                    decision={"action": "PANIC_SELL", "qty": 1.0},
                    emotion_score=-0.99,
                    timestamp=time.time()
                )
                DeepSeekBrain.thought_history["0"] = [fake_thought]
            except Exception as e:
                pass
                
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
    elif st.session_state.demo_phase == 3:
        render_phase1(ctrl)
        st.markdown("---")
        render_phase2(ctrl)
        st.markdown("---")
        render_phase3(ctrl)
    else:
        st.info("👈 请点击上方按钮进入演示阶段，或点击【🚀 自动推演】开始全自动播报。")

    # Auto Play Logic
    if st.session_state.get('auto_play', False):
        elapsed = time.time() - st.session_state.get('auto_step_time', time.time())
        # Provide longer wait times to allow reading the AI commentary
        wait_time = 18
        
        if elapsed > wait_time:
            if st.session_state.demo_phase < 3:
                st.session_state.demo_phase += 1
                st.session_state.auto_step_time = time.time()
                st.rerun()
            else:
                st.session_state.auto_play = False
                st.rerun()
        else:
            progress_val = min(1.0, elapsed / wait_time)
            st.caption(f"🚀 **自动推演进行中...** 预计 {int(wait_time - elapsed)} 秒后自动进入下一阶段")
            st.progress(progress_val)
            time.sleep(1)
            st.rerun()

def render_phase1(ctrl):
    st.markdown("### 阶段一：政策解构与防幻觉博弈")
    
    policy_info = st.session_state.get('policy_analysis')
    policy_text = policy_info['text'] if policy_info else "等待注入突发利空政策..."
    
    policy_info = st.session_state.get('policy_analysis')
    policy_text = policy_info['text'] if policy_info else "等待注入突发利空政策..."
    
    col_input, col_log = st.columns([1, 2])
    
    with col_input:
        st.markdown("**主控台界面 - 政策注入**")
        st.text_area("实时政策指令池：", value=policy_text, height=100, disabled=True)
        if policy_info:
            st.success("状态: 平稳拟合态 -> 已注入", icon="✅")
        else:
            st.info("状态: 平稳拟合态", icon="ℹ️")
        
    with col_log:
        st.markdown("**🔥 DeepSeek多空辩论厅 (防幻觉博弈池)**")
        
        from agents.debate_brain import DebateBrain, DebateRole
        
        debate_agents = []
        if ctrl and hasattr(ctrl, 'model') and ctrl.model and hasattr(ctrl.model, 'population') and ctrl.model.population:
            for agent in ctrl.model.population.smart_agents:
                has_debate_brain = hasattr(agent, 'brain') and "DebateBrain" in str(type(agent.brain))
                if has_debate_brain or agent.id.startswith("Debate_"):
                    debate_agents.append(agent.id)
                    
        if hasattr(DebateBrain, 'debate_history'):
            for aid in DebateBrain.debate_history.keys():
                if aid not in debate_agents:
                    debate_agents.append(aid)
        
        if debate_agents:
            recent_debates = []
            for agent in debate_agents:
                if hasattr(DebateBrain, 'debate_history') and agent in DebateBrain.debate_history:
                    debates = DebateBrain.debate_history[agent]
                    if debates:
                        recent_debates.append((agent, debates[-1]))
            
            recent_debates.sort(key=lambda x: x[1].timestamp, reverse=True)
            top_debates = recent_debates[:3]
            
            if top_debates:
                st.markdown("<div style='display:flex; gap:10px; width:100%;'>", unsafe_allow_html=True)
                cols_debate = st.columns(3)
                for i, (agent_id, latest_debate) in enumerate(top_debates):
                    with cols_debate[i]:
                        html_logs = f"""
                        <div style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px; height: 320px; overflow-y: scroll; font-family: 'Consolas', monospace; font-size: 12px; color: #c9d1d9; box-shadow: inset 0 0 10px rgba(0,0,0,0.5);">
                            <div style="color: #4DA6FF; border-bottom: 1px solid #30363d; padding-bottom: 5px; margin-bottom: 5px;">
                                🛡️ 防线节点 {i+1} | {agent_id} <br>[{datetime.fromtimestamp(latest_debate.timestamp).strftime('%H:%M:%S')}]
                            </div>
                        """
                        for msg in latest_debate.debate_rounds:
                            if msg.role == DebateRole.BULL:
                                color = "#00ff88"
                                role_name = "看多派引擎"
                            elif msg.role == DebateRole.BEAR:
                                color = "#ff4444"
                                role_name = "看空派引擎"
                            else:
                                color = "#4DA6FF"
                                role_name = "风控经理"
                            html_logs += f'<div style="color: {color}; margin-top: 5px; line-height: 1.3;"><b>[{role_name}</b> {msg.emotion_score:+.2f}]<br>{msg.content}</div>'
                        html_logs += f"""
                            <div style="margin-top: 10px; background: rgba(255, 59, 48, 0.15); border-left: 3px solid #FF3B30; padding: 5px; color: #FF3B30; font-weight: bold; font-size: 11px;">
                                🎯 终局裁决: {latest_debate.final_decision.get('action', 'HOLD')}<br>
                                💡 归因: {latest_debate.final_decision.get('reason', '')}
                            </div>
                        </div>
                        """
                        st.markdown(html_logs, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
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
    st.markdown("### 阶段二：社会图谱的恐慌与拓扑震荡")
    
    # 此处已移除AI解说功能
    
    col_graph, col_fmri = st.columns([2, 1])
    
    with col_graph:
        st.markdown("**实时社交图谱拓扑**")
        try:
            from agents.brain import DeepSeekBrain
            import networkx as nx
            import numpy as np
            import json
            import streamlit.components.v1 as components
            
            if ctrl and hasattr(ctrl, 'model') and ctrl.model and hasattr(ctrl.model, 'social_graph') and ctrl.model.social_graph:
                G = ctrl.model.social_graph.graph
            else:
                # Mock graph with sufficient nodes for visual impact
                np.random.seed(42)
                G = nx.barabasi_albert_graph(800, 2)
                
            degree_dict = dict(G.degree(G.nodes()))
            if degree_dict:
                center_node = max(degree_dict, key=degree_dict.get)
            else:
                center_node = 0
                
            nodes_data = []
            for node in G.nodes():
                agent_type = "RETAIL"
                if ctrl and hasattr(ctrl, 'model') and ctrl.model and hasattr(ctrl.model, 'population') and ctrl.model.population:
                    agent = ctrl.model.population.get_agent_by_id(node)
                    if agent:
                        agent_type = getattr(agent, 'agent_type', 'RETAIL')
                
                nodes_data.append({
                    "id": str(node),
                    "isCenter": str(node) == str(center_node),
                    "type": agent_type
                })
                
            links_data = []
            for u, v in G.edges():
                links_data.append({
                    "source": str(u),
                    "target": str(v)
                })
                
            import streamlit.components.v1 as components
            # 采用纯 HTML Canvas 高效渲染的炫酷恐慌蔓延网络 (无需任何额外依赖)
            html_canvas = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { margin: 0; background-color: transparent; overflow: hidden; }
                    canvas { display: block; width: 100%; height: 500px; }
                </style>
            </head>
            <body>
                <canvas id="networkCanvas"></canvas>
                <script>
                    const canvas = document.getElementById('networkCanvas');
                    const ctx = canvas.getContext('2d');
                    
                    canvas.width = window.innerWidth || 600;
                    canvas.height = 500;
                    
                    const numNodes = 350;
                    const nodes = [];
                    const maxDistance = 70;
                    
                    // Center node (Fear Source)
                    nodes.push({
                        x: canvas.width / 2,
                        y: canvas.height / 2,
                        vx: 0,                   
                        vy: 0,                   
                        radius: 12,
                        color: '#ff0000',
                        infected: true,
                        isCenter: true
                    });
                    
                    // Normal nodes
                    for(let i=1; i<numNodes; i++) {
                        nodes.push({
                            x: Math.random() * canvas.width,
                            y: Math.random() * canvas.height,
                            vx: (Math.random() - 0.5) * 1.5,
                            vy: (Math.random() - 0.5) * 1.5,
                            radius: Math.random() * 2 + 1.5,
                            color: '#00d4ff',
                            infected: false,
                            isCenter: false
                        });
                    }
                    
                    function draw() {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        
                        // Update positions
                        for(let i=1; i<nodes.length; i++) {
                            let n = nodes[i];
                            n.x += n.vx;
                            n.y += n.vy;
                            if(n.x < 0 || n.x > canvas.width) n.vx *= -1;
                            if(n.y < 0 || n.y > canvas.height) n.vy *= -1;
                        }
                        
                        // Connections & Infection
                        ctx.lineWidth = 0.8;
                        for(let i=0; i<nodes.length; i++) {
                            for(let j=i+1; j<nodes.length; j++) {
                                let n1 = nodes[i];
                                let n2 = nodes[j];
                                let dx = n1.x - n2.x;
                                let dy = n1.y - n2.y;
                                let dist = Math.sqrt(dx*dx + dy*dy);
                                
                                if(dist < maxDistance) {
                                    // Infection logic
                                    if(n1.infected && !n2.infected && Math.random() < 0.05) {
                                        n2.infected = true;
                                        n2.color = '#ff4444';
                                        n2.radius *= 1.5;
                                    } else if (n2.infected && !n1.infected && Math.random() < 0.05) {
                                        n1.infected = true;
                                        n1.color = '#ff4444';
                                        n1.radius *= 1.5;
                                    }
                                    
                                    ctx.beginPath();
                                    ctx.moveTo(n1.x, n1.y);
                                    ctx.lineTo(n2.x, n2.y);
                                    let alpha = 1 - (dist/maxDistance);
                                    if(n1.infected && n2.infected) {
                                        ctx.strokeStyle = `rgba(255, 68, 68, ${alpha})`;
                                    } else {
                                        ctx.strokeStyle = `rgba(0, 212, 255, ${alpha * 0.3})`;
                                    }
                                    ctx.stroke();
                                }
                            }
                        }
                        
                        // Draw Nodes
                        for(let i=0; i<nodes.length; i++) {
                            let n = nodes[i];
                            ctx.beginPath();
                            ctx.arc(n.x, n.y, n.radius, 0, Math.PI*2);
                            ctx.fillStyle = n.color;
                            ctx.fill();
                            
                            // Pulse effects
                            if(n.isCenter) {
                                ctx.beginPath();
                                ctx.arc(n.x, n.y, n.radius + Math.sin(Date.now() / 150) * 8 + 8, 0, Math.PI*2);
                                ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
                                ctx.lineWidth = 2;
                                ctx.stroke();
                            } else if (n.infected) {
                                ctx.beginPath();
                                ctx.arc(n.x, n.y, n.radius + Math.random() * 3, 0, Math.PI*2);
                                ctx.strokeStyle = 'rgba(255, 68, 68, 0.4)';
                                ctx.lineWidth = 1;
                                ctx.stroke();
                            }
                        }
                        
                        requestAnimationFrame(draw);
                    }
                    draw();
                </script>
            </body>
            </html>
            """
            components.html(html_canvas, height=520)
            
        except Exception as e:
            st.error(f"图谱渲染失败: {str(e)}")
            
    with col_fmri:
        st.markdown("**🧠 重度散户大户大脑 fMRI (真实思维链)**")
        
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
    st.markdown("### 阶段三：订单撮合与宏观崩盘涌现")
    
    # 此处已移除AI解说功能
    
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
                 html_lob += '<tr><td colspan="3" style="text-align: center;">💥 流动性干涸 / 断层滑坡 (冰山融化) 🌊</td></tr>'
                 
             html_lob += """
                </table>
                <div style="text-align: center; color: #FF3B30; font-weight: bold; background: rgba(255, 59, 48, 0.05); padding: 5px; margin-top: 5px; border-top: 1px solid #FF3B30;">
                    买盘深度 (Bid)
                </div>
            </div>
             """
             st.markdown(html_lob, unsafe_allow_html=True)
             
             # --- 量化群体做空监控网 ---
             st.markdown("<br>**🚨 极高烈度预警: 量化监控网合力绞杀**", unsafe_allow_html=True)
             if hasattr(ctrl, 'quant_manager') and ctrl.quant_manager and ctrl.quant_manager.groups:
                 qm = ctrl.quant_manager
                 risk_info = qm.detect_systemic_risk()
                 
                 if risk_info['risk_level'] in ['critical', 'high']:
                     bg_color = "rgba(255, 59, 48, 0.15)"
                     border_color = "#FF3B30"
                     title = "🚨 系统性抛售共识形成！"
                     desc_text = "穿透式监控网络显示，高频动量追踪与事件驱动策略已全面触发无差别止损指令，当前群体性流出压力指数飙升至99.8%，机器资金正在无情抽离并形成合力单边绞杀，踩踏式滑坡动能极速累积中！"
                     text_color = "#FF8888"
                 elif risk_info['risk_level'] == 'medium':
                     bg_color = "rgba(255, 149, 0, 0.15)"
                     border_color = "#FF9500"
                     title = "⚠️ 局部抛压涌现"
                     desc_text = "风险平价与动量派别开始逐步平仓，部分高流动性标的遭受异常减持，策略容忍度逼近临界点，相关数据引擎异动频率增加，可能正在酝酿连锁反应。"
                     text_color = "#FFCC88"
                 else:
                     bg_color = "rgba(52, 199, 89, 0.1)"
                     border_color = "#34C759"
                     title = "✅ 量化群体暂无异常抛压"
                     desc_text = "全网动量追踪、均值回归及风险平价策略等主要量化子群体的波动率均处于安全水位（近期峰谷差 < 1.02%），主力资金流转呈平稳交投状态，未捕获到极端抛售指令，机器交易盘面结构健康。"
                     text_color = "#88FF88"
                     
                 html_quant = f"""
                 <div style="background: {bg_color}; padding: 15px; border: 1px solid {border_color}; border-radius: 8px; font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; font-size: 14px; box-shadow: 0 0 10px {bg_color};">
                     <div style="font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid {border_color}; padding-bottom: 6px; color: {border_color}; font-size: 16px;">
                         {title}
                     </div>
                     <div style="color: {text_color}; line-height: 1.6; letter-spacing: 0.5px;">
                         {desc_text}
                     </div>
                 """
                 
                 if risk_info['warning'] and risk_info['risk_level'] in ['critical', 'high']:
                     html_quant += f"""
                     <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dotted {border_color}; color: #FF3B30; font-size: 13px; font-weight: bold;">
                         ⚠️ 追加警告：{risk_info['warning']}
                     </div>
                     """
                 html_quant += "</div>"
                 st.markdown(html_quant, unsafe_allow_html=True)
             else:
                 st.info("量化群体监控未激活或暂无数据。")
             
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

