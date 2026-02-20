
import asyncio
from core.scheduler import SimulationController, SimulationMode

def verify_agent_configuration(mode_name="FAST"):
    print(f"\n--- Testing Mode: {mode_name} ---")
    
    try:
        controller = SimulationController(
            deepseek_key="dummy", 
            mode=mode_name
        )
        
        real_agents = []
        sim_agents = []
        for a in controller.model.agents:
            # CivitasAgent -> core (TraderAgent) -> use_llm
            if hasattr(a, 'core') and getattr(a.core, 'use_llm', False):
                real_agents.append(a)
            else:
                sim_agents.append(a)
        
        print(f"Total Agents: {len(controller.model.agents)}")
        print(f"Real API Agents (use_llm=True): {len(real_agents)}")
        for ra in real_agents:
             print(f"  - {ra.unique_id}")
        print(f"Simulation Agents (use_llm=False): {len(sim_agents)}")
        
        # Check Top 5 Limit
        expected_real = 5 # 5 from loop (National Team is rule-based, not LLM)
        assert len(real_agents) == expected_real, f"Expected {expected_real} real agents, got {len(real_agents)}"
        
        # Check Model Priority
        print("Checking Model Priorities for Real Agents:")
        for agent in real_agents:
            brain = agent.core.brain
            # Need to consider that brain might store priority as None if not set?
            # But we set it explicitly.
            priority = getattr(brain, 'model_priority', [])
            print(f"  Agent {agent.unique_id} ({agent.investor_type.name}): {priority}")
            
            # Check specific logic
            if mode_name == "FAST":
                assert "deepseek-chat" in priority, f"FAST mode should use chat, got {priority}"
            elif mode_name == "DEEP":
                assert "deepseek-reasoner" in priority, f"DEEP mode should use reasoner, got {priority}"
            elif mode_name == "SMART":
                # Agent 0 (or 1) uses Reasoner, others Chat
                # In CivitasModel, idx=0 gets Reasoner.
                # Based on previous output, Agent 1 corresponds to idx=0.
                uid = str(agent.unique_id)
                is_first_agent = uid == "1" or uid == "0" or uid.endswith("_0")
                
                if is_first_agent:
                    assert "deepseek-reasoner" in priority, f"Agent {agent.unique_id} missing reasoner in SMART mode. Got: {priority}"
                else:
                    assert "deepseek-chat" in priority, f"SMART mode others should use chat, got {priority}"
                    
        print(f"Mode {mode_name} Verified Successfully!")
        return True
        
    except Exception as e:
        print(f"Mode {mode_name} Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    modes = ["FAST", "SMART", "DEEP"]
    for m in modes:
        verify_agent_configuration(m)
