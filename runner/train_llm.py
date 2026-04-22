import os
import json
import time
import sys
from dotenv import load_dotenv

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.client import GlobalCrisisEnv
from agent.planner_llm import LLMPlanner
from agent.memory import Memory
from agent.reflection import generate_reflection

load_dotenv()

def log_episode(data):
    """Appends episode results to a JSONL file for portfolio tracking."""
    log_path = "logs/episode_scores.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

def run_phase_2b_training(episodes=5, task_id="hard"):
    print(f"--- Starting Phase 2B: LLM Integration ({task_id}) ---")
    planner = LLMPlanner()
    memory = Memory()
    
    with GlobalCrisisEnv() as env:
        for ep in range(episodes):
            print(f"\n[EPISODE {ep+1}/{episodes}] Starting...")
            obs = env.reset(task_id=task_id)
            
            total_reward = 0
            bottleneck_ever_active = False
            bottlenecks_cleared = 0
            bottleneck_occurrences = 0
            
            while not obs.done:
                if obs.transport_demand > 5:
                    bottleneck_ever_active = True
                    bottleneck_occurrences += 1

                # Normalize state for memory consistency
                state_dict = {
                    "fuel_available": obs.fuel_available,
                    "hospital_demand": obs.hospital_demand,
                    "emergency_demand": obs.emergency_demand,
                    "transport_demand": obs.transport_demand,
                    "residential_demand": obs.residential_demand,
                    "bottleneck": obs.transport_demand > 5
                }
                state_str = json.dumps(state_dict)

                # 1. Get LLM Action
                action, thought = planner.decide_action(obs, task_id)
                
                # 2. Step Environment
                prev_transport_demand = obs.transport_demand
                obs = env.step(action)
                
                if prev_transport_demand > 5 and obs.transport_demand <= 5:
                    bottlenecks_cleared += 1

                # 3. Reflection & Memory Update
                reflection = generate_reflection(state_str, action, obs.reward)
                memory.add(state_str, reflection)
                
                total_reward += obs.reward

            final_score = round(total_reward / 5.0, 4)
            bottleneck_cleared_success = (bottlenecks_cleared == bottleneck_occurrences) if bottleneck_ever_active else True

            # 4. Log for Portfolio
            log_data = {
                "phase": "2B",
                "difficulty": task_id,
                "score": final_score,
                "bottleneck_cleared": bottleneck_cleared_success,
                "fuel_wasted": 0, 
                "agent_type": "llm_with_memory",
                "timestamp": time.time()
            }
            log_episode(log_data)
            
            print(f"Episode {ep+1} Finished. Score: {final_score} | Bottleneck Cleared: {bottleneck_cleared_success}")

if __name__ == "__main__":
    run_phase_2b_training(episodes=3, task_id="hard")
