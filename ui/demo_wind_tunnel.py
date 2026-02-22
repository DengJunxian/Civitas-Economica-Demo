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
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("⏹️ 重置演示", use_container_width=True):
            st.session_state.demo_phase = 0
            st.session_state.auto_play = False
            st.rerun()
    with col2:
        if st.button("1️⃣ 政策与辩论", use_container_width=True):
            st.session_state.demo_phase = 1
            st.session_state.auto_play = False
            st.rerun()
    with col3:
        if st.button("2️⃣ 网络传染", use_container_width=True):
            st.session_state.demo_phase = 2
            st.session_state.auto_play = False
            st.rerun()
    with col4:
        if st.button("3️⃣ 撮合与崩盘", use_container_width=True):
            st.session_state.demo_phase = 3
            st.session_state.auto_play = False
            st.rerun()
    with col5:
        if st.button("▶️ 完整展示", use_container_width=True):
            st.session_state.demo_phase = 4
            st.session_state.auto_play = False
            st.rerun()
    with col6:
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
                
                deepseek_key = GLOBAL_CONFIG.DEEPSEEK_API_KEY or "sk-ef4fd5a8ac9c4861aa812af3875652f7"
                zhipu_key = GLOBAL_CONFIG.ZHIPU_API_KEY or "4d963afd591d4c93940b08b06d766e91.bWaMIWJnuKhOUo7y"
                
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
            policy_text = "【紧急突发】中国证监会联合中国人民银行、金融监管总局发布联合声明：为防范化解重大金融系统性风险，即日起全面暂停程序化和量化交易，融券业务实施100%保证金并暂停新增融券规模。同时，立案调查多家头部量化及做市商机构涉嫌操纵市场等违法违规行为。此政策将重构市场流动性生态，预计短期内将引发场内资金剧烈踩踏和强平风险，市场情绪极度恐慌。"
            
            import queue
            if 'cmd_queue' not in st.session_state:
                st.session_state.cmd_queue = queue.Queue()
            st.session_state.cmd_queue.put({"type": "policy", "content": policy_text})
            st.session_state.policy_analysis = {
                "text": policy_text,
                "result": {"market_impact": "分析中..."},
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
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
        st.info("👈 请点击上方按钮进入演示阶段，或点击【🚀 自动推演】开始全自动播报。")

    # Auto Play Logic
    if st.session_state.get('auto_play', False):
        elapsed = time.time() - st.session_state.get('auto_step_time', time.time())
        # Demo timings per phase based on typical reading/talking speed: 12 seconds
        wait_time = 12
        
        if elapsed > wait_time:
            if st.session_state.demo_phase < 4:
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
    st.markdown("### 阶段一：宏观注入与机构拆解")
    
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
                has_debate_brain = hasattr(agent, 'brain') and "DebateBrain" in str(type(agent.brain))
                if has_debate_brain or agent.id.startswith("Debate_"):
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
                
            graph_json = json.dumps({"nodes": nodes_data, "links": links_data})
            
            html_code = f'''
<!DOCTYPE html>
<html>
<head>
  <style> body {{ margin: 0; background-color: rgba(0,0,0,0); overflow: hidden; }} </style>
  <script src="https://cdn.jsdelivr.net/npm/force-graph@1.43.5/dist/force-graph.min.js"></script>
</head>
<body>
  <div id="graph" style="width: 100%; height: 500px;"></div>
  <script>
    const graphData = {graph_json};
    const centerNodeId = "{str(center_node)}";
    
    // Initialize nodes
    graphData.nodes.forEach(node => {{
        node.color = '#00d4ff'; // Blue
        node.size = node.isCenter ? 12 : 2.5;
        node.infected = false;
    }});

    const Graph = ForceGraph()(document.getElementById('graph'))
      .graphData(graphData)
      .nodeId('id')
      .nodeColor(n => n.color)
      .nodeVal(n => n.size)
      .linkColor(() => 'rgba(150, 150, 150, 0.3)')
      .linkWidth(0.5)
      .linkDirectionalParticles(link => link.particleCount || 0)
      .linkDirectionalParticleSpeed(0.012)
      .linkDirectionalParticleWidth(2.5)
      .linkDirectionalParticleColor(() => '#ff4444')
      .backgroundColor('rgba(0,0,0,0)')
      .nodeCanvasObject((node, ctx, globalScale) => {{
          // Draw node
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI, false);
          ctx.fillStyle = node.color;
          ctx.fill();
          
          // Draw tooltip if hovered
          if (node === Graph.hoverNode()) {{
              const label = node.isCenter ? "大V机构节点 (情绪源)" : "System 2 正在读取大V的看空言论并修改自身的风险参数。";
              const fontSize = (node.isCenter ? 14 : 12) / globalScale;
              ctx.font = node.isCenter ? `bold ${{fontSize}}px Sans-Serif` : `${{fontSize}}px Sans-Serif`;
              const textWidth = ctx.measureText(label).width;
              const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.4); 

              ctx.fillStyle = 'rgba(22, 27, 34, 0.9)'; // Dark GitHub-like bg
              ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] - node.size - 4, ...bckgDimensions);
              ctx.strokeStyle = '#ff4444';
              ctx.lineWidth = 1 / globalScale;
              ctx.strokeRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] - node.size - 4, ...bckgDimensions);

              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = '#ff4444';
              ctx.fillText(label, node.x, node.y - bckgDimensions[1]/2 - node.size - 4);
          }}
      }})
      .onNodeHover(node => {{
          document.getElementById('graph').style.cursor = node ? 'pointer' : null;
      }});

    // Setup initial camera
    setTimeout(() => {{
        Graph.zoomToFit(400, 20);
    }}, 500);

    // Animation logic
    setTimeout(() => {{
        // 1. Center node turns red
        const centerNode = graphData.nodes.find(n => n.id === centerNodeId);
        if (centerNode) {{
            centerNode.color = '#ff4444';
            centerNode.infected = true;
            Graph.nodeColor(Graph.nodeColor());
            
            // Generate BFS distances for infection propagation
            const distances = {{}};
            distances[centerNodeId] = 0;
            
            const adj = {{}};
            graphData.nodes.forEach(n => adj[n.id] = []);
            graphData.links.forEach(l => {{
                // force-graph mutates strings to object handles
                const u = typeof l.source === 'object' ? l.source.id : l.source;
                const v = typeof l.target === 'object' ? l.target.id : l.target;
                adj[u].push(v);
                adj[v].push(u);
            }});
            
            const queue = [centerNodeId];
            while(queue.length > 0) {{
                const u = queue.shift();
                adj[u].forEach(v => {{
                    if (distances[v] === undefined) {{
                        distances[v] = distances[u] + 1;
                        queue.push(v);
                    }}
                }});
            }}
            
            // Start ripple effect based on distance
            const baseDelay = 600; // Infection spread speed
            graphData.nodes.forEach(n => {{
                const dist = distances[n.id];
                if (dist !== undefined) {{
                    setTimeout(() => {{
                        if (n.id !== centerNodeId) {{
                            n.color = '#ff4444';
                            n.infected = true;
                            Graph.nodeColor(Graph.nodeColor());
                        }}
                        
                        // Emit particles to further uninfected neighbors visually
                        const myLinks = graphData.links.filter(l => 
                           ((typeof l.source === 'object' ? l.source.id : l.source) === n.id || 
                            (typeof l.target === 'object' ? l.target.id : l.target) === n.id)
                        );
                        myLinks.forEach(l => l.particleCount = 1);
                        Graph.linkDirectionalParticles(Graph.linkDirectionalParticles());
                        
                        // Pulse effect
                        const originalSize = n.size;
                        n.size = originalSize * 1.8;
                        Graph.nodeVal(Graph.nodeVal());
                        setTimeout(() => {{
                             n.size = originalSize;
                             Graph.nodeVal(Graph.nodeVal());
                        }}, 200);

                    }}, dist * baseDelay);
                }}
            }});
        }}
    }}, 2000);
  </script>
</body>
</html>
'''
            components.html(html_code, height=520)
            
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
             
             # --- 量化群体做空监控网 ---
             st.markdown("<br>**🤖 量化群体做空监控网**", unsafe_allow_html=True)
             if hasattr(ctrl, 'quant_manager') and ctrl.quant_manager and ctrl.quant_manager.groups:
                 qm = ctrl.quant_manager
                 risk_info = qm.detect_systemic_risk()
                 
                 if risk_info['risk_level'] in ['critical', 'high']:
                     bg_color = "rgba(255, 59, 48, 0.15)"
                     border_color = "#FF3B30"
                     title = "🚨 系统性抛售共识形成！"
                 elif risk_info['risk_level'] == 'medium':
                     bg_color = "rgba(255, 149, 0, 0.15)"
                     border_color = "#FF9500"
                     title = "⚠️ 部分群体出现异动"
                 else:
                     bg_color = "rgba(52, 199, 89, 0.1)"
                     border_color = "#34C759"
                     title = "✅ 量化群体暂无异常抛压"
                     
                 html_quant = f"""
                 <div style="background: {bg_color}; padding: 12px; border: 1px solid {border_color}; border-radius: 8px; font-family: monospace; font-size: 13px;">
                     <div style="font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid {border_color}; padding-bottom: 4px; color: {border_color};">
                         {title}
                     </div>
                 """
                 
                 for gid, group in qm.groups.items():
                     action = group.collective_action or "HOLD"
                     pressure = group.sell_pressure * 100
                     if action in ["SELL", "PANIC_SELL"]:
                         color = "#FF3B30"
                         action_str = f"抛售 ↘ (压:{pressure:.1f}%)"
                     elif action == "BUY":
                         color = "#34C759"
                         action_str = "吸筹 ↗"
                     else:
                         color = "#888"
                         action_str = "观望 ~"
                         
                     html_quant += f"""
                     <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                         <span style="color: #c9d1d9;">[{group.strategy_name}]</span>
                         <span style="color: {color}; font-weight: bold;">{action_str}</span>
                     </div>
                     """
                     
                 if risk_info['warning']:
                     html_quant += f"""
                     <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dotted {border_color}; color: {border_color}; font-size: 12px;">
                         {risk_info['warning']}
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

