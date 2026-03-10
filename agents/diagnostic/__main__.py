# file: agents/diagnostic/__main__.py
import asyncio

from agents.diagnostic.report_agent import ReportAgent
from agents.persona import Persona
from agents.trader_agent import TraderAgent


async def async_main():
    print("=====================================================")
    print("   Civitas 自治复盘探针 (ReportAgent / ReACT) 终端   ")
    print("=====================================================")
    print("正在初始化虚拟测试环境...\n")

    # 构建 Mock 市场环境，并挂载 LOB 日志
    class MockMarketEnv:
        last_price = 3200.5
        trend = "震荡"
        panic_level = 0.6
        current_time = 1718000000.0
        lob_logs = [
            {"step": 14, "event": "margin_call", "agent_id": "Retail_Loss", "detail": "保证金不足警告"},
            {"step": 15, "event": "liquidation", "agent_id": "Retail_Loss", "detail": "触发强平，先爆仓"},
            {"step": 15, "event": "follow_sell", "agent_id": "Inst_Profit", "detail": "观察到流动性恶化后减仓"},
        ]

    market_env = MockMarketEnv()

    # 构建 Mock Agents Map
    agents_map = {}

    p1 = Persona(name="Retail_Loss", risk_preference="激进")
    a1 = TraderAgent(agent_id="Retail_Loss", persona=p1)
    a1.cash_balance = 5000
    a1.total_pnl = -45000
    a1.brain.state.confidence = 20
    agents_map["Retail_Loss"] = a1

    p2 = Persona(name="Inst_Profit", risk_preference="稳健")
    a2 = TraderAgent(agent_id="Inst_Profit", persona=p2)
    a2.cash_balance = 80000
    a2.total_pnl = 15000
    a2.brain.state.confidence = 85
    agents_map["Inst_Profit"] = a2

    # 初始化自治复盘 Agent
    try:
        report_agent = ReportAgent(agents_map=agents_map, market_env=market_env)
    except Exception as e:
        print(f"初始化 ReportAgent 失败 (请检查 config.yaml 中 API 配置): {e}")
        return

    print("环境就绪。已注册复盘工具 (Tool-Calling):")
    print(" - query_lob_log")
    print(" - search_agent_memory")
    print(" - send_interview\n")

    print("示例提问：排查第15个时间步是谁先爆仓的？")
    print("输入 'exit' 退出。")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n[You] > ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input.strip():
                continue

            print("\n[ReportAgent] 正在进行 ReACT 复盘推理...")
            response = await report_agent.chat(user_input)
            print("-" * 50)
            print(f"[ReportAgent]\n{response}")
            print("-" * 50)
        except KeyboardInterrupt:
            break


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
