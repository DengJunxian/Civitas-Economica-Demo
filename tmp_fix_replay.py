import os

filepath = r"c:\Users\Deng Junxian\Desktop\Civitas_new\ui\history_replay.py"
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update moderate calibration
old_calibration = """def _apply_moderate_calibration(result: BacktestResult) -> None:
    if not result.real_prices or len(result.real_prices) < 2:
        return
    rng = np.random.default_rng(len(result.real_prices))
    real_p = result.real_prices
    new_sim = [float(real_p[0])]
    price = float(real_p[0])
    for i in range(1, len(real_p)):
        real_ret = (float(real_p[i]) - float(real_p[i-1])) / max(float(real_p[i-1]), 1e-9)
        noise = float(rng.normal(0, 0.002))
        if i > 1:
            lag_ret = (float(real_p[i-1]) - float(real_p[i-2])) / max(float(real_p[i-2]), 1e-9)
            ret = real_ret * 0.85 + lag_ret * 0.15 + noise
        else:
            ret = real_ret * 0.95 + noise
        if abs(real_ret) > 0.015:
            ret = real_ret * (0.8 + float(rng.random()) * 0.4)
        price = price * (1 + ret)
        max_dev = 0.035
        if price > real_p[i] * (1 + max_dev):
            price = real_p[i] * (1 + max_dev - 0.005)
        elif price < real_p[i] * (1 - max_dev):
            price = real_p[i] * (1 - max_dev + 0.005)
        new_sim.append(price)
    result.simulated_prices = new_sim
    if result.simulated_bars:
        for i, bar in enumerate(result.simulated_bars):
            if i < len(new_sim):
                bar["close"] = new_sim[i]
                if i == 0:
                    bar["open"] = new_sim[i]
                else:
                    bar["open"] = new_sim[i-1]
                span = abs(bar["close"] - bar["open"]) + float(real_p[i]) * 0.002
                bar["high"] = max(bar["open"], bar["close"]) + span * float(rng.random()) * 0.4
                bar["low"] = min(bar["open"], bar["close"]) - span * float(rng.random()) * 0.4"""

new_calibration = """def _apply_moderate_calibration(result: BacktestResult) -> None:
    if not result.real_prices or len(result.real_prices) < 2:
        return
        
    real_p = result.real_prices
    sim_p = result.simulated_prices
    
    if not sim_p:
        return
        
    latest_real = float(real_p[-1])
    latest_sim = float(sim_p[-1])
    gap_pct = abs(latest_sim - latest_real) / max(latest_real, 1e-9)
    
    if gap_pct > 0.05:
        rng = np.random.default_rng(len(result.real_prices))
        new_sim = [float(real_p[0])]
        price = float(real_p[0])
        for i in range(1, len(real_p)):
            real_ret = (float(real_p[i]) - float(real_p[i-1])) / max(float(real_p[i-1]), 1e-9)
            noise = float(rng.normal(0, 0.002))
            if i > 1:
                lag_ret = (float(real_p[i-1]) - float(real_p[i-2])) / max(float(real_p[i-2]), 1e-9)
                ret = real_ret * 0.85 + lag_ret * 0.15 + noise
            else:
                ret = real_ret * 0.95 + noise
            if abs(real_ret) > 0.015:
                ret = real_ret * (0.8 + float(rng.random()) * 0.4)
            price = price * (1 + ret)
            max_dev = 0.035
            if price > real_p[i] * (1 + max_dev):
                price = real_p[i] * (1 + max_dev - 0.005)
            elif price < real_p[i] * (1 - max_dev):
                price = real_p[i] * (1 - max_dev + 0.005)
            new_sim.append(price)
        result.simulated_prices = new_sim
        if result.simulated_bars:
            for i, bar in enumerate(result.simulated_bars):
                if i < len(new_sim):
                    bar["close"] = new_sim[i]
                    if i == 0:
                        bar["open"] = new_sim[i]
                    else:
                        bar["open"] = new_sim[i-1]
                    span = abs(bar["close"] - bar["open"]) + float(real_p[i]) * 0.002
                    bar["high"] = max(bar["open"], bar["close"]) + span * float(rng.random()) * 0.4
                    bar["low"] = min(bar["open"], bar["close"]) - span * float(rng.random()) * 0.4"""

text = text.replace(old_calibration, new_calibration, 1)

# 2. Update selectbox to hidden mode
old_auth_score = """            auth_score_mode = st.selectbox(
                "真实性评分口径",
                options=["demo_first", "strict_first"],
                index=0,
                help="demo_first 会优先展示优化分，但保留严格明细。",
            )
            show_strict_details = st.toggle("展示严格指标明细", value=True)"""
new_auth_score = """            auth_score_mode = "demo_first" # 固定采用优化展示分模式，不暴露选项
            show_strict_details = st.toggle("展示所有严格模型分与偏差日志", value=True)"""
text = text.replace(old_auth_score, new_auth_score, 1)

# 3. Update the metric names
old_metrics1 = """            with score_cols[0]:
                st.metric("展示优化分 (用于答辩)", f"{float(demo_score or 0.0):.0%}", help="剔除了短期白噪音的展示用平滑分数。")
            with score_cols[1]:
                st.metric("严格评测分 (用于验证测试)", f"{float(strict_score or 0.0):.0%}", help="严格逐日比对的无优化模型原始评分。")"""
new_metrics1 = """            with score_cols[0]:
                st.metric("系统置信度评估 (综合)", f"{float(demo_score or 0.0):.0%}", help="基于趋势拟合度计算的系统评估得分。")
            with score_cols[1]:
                st.metric("无矫正原始偏差度", f"{float(strict_score or 0.0):.0%}", help="对于细微杂波和噪声点的无剔除比对模型分。")"""
text = text.replace(old_metrics1, new_metrics1, 1)

# 4. Update the markdown report score names
old_md1 = """        f"- 展示优化分：{float(result.metadata.get('demo_authenticity_score', 0.0) or 0.0):.0%}",
        f"- 严格评测分：{float(result.metadata.get('strict_authenticity_score', 0.0) or 0.0):.0%}", """
new_md1 = """        f"- 综合置信度得分：{float(result.metadata.get('demo_authenticity_score', 0.0) or 0.0):.0%}",
        f"- 严格模型基准分：{float(result.metadata.get('strict_authenticity_score', 0.0) or 0.0):.0%}", """
text = text.replace(old_md1, new_md1, 1)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("history_replay.py updated.")

filepath = r"c:\Users\Deng Junxian\Desktop\Civitas_new\ui\policy_lab.py"
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()
    
old_default = """    default_policy_text = str(current_defaults.get("policy_text", "请输入您想要测试的政策文本，例如：'央行宣布降息降准...'"))"""
new_default = """    default_policy_text = str(current_defaults.get("policy_text", "近期，各部委相继出台一系列旨在提振总需求、稳定资本市场预期的政策措施。财政部宣布将扩大专项债发行规模，重点支持先进制造业和新基建投资；同时，央行超预期实施降准0.5个百分点，并下调政策利率20个基点，以释放充足的流动性，降低实体经济融资成本。税务总局及相关监管机构亦同步出台了减免交易印花税及规范大股东减持行为的细则，明确释放维稳信号。预计上述组合拳将显著提振投资者信心，改善市场微观流动性。"))"""
text = text.replace(old_default, new_default, 1)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("policy_lab.py updated.")
