import sys
import os
import json
import random
import requests

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.client import GlobalCrisisEnv
from agent.planner import decide_action

# Configuration
TARGET_PER_TASK = 60
EXPERT_RATIO = 0.7
NEAR_EXPERT_RATIO = 0.3
WASTE_THRESHOLD = 0.1
OUTPUT_PATH = "logs/expert_trajectories.jsonl"
TASKS = ["easy", "medium", "hard"]
MAX_FAIL_SAFES = 200

# Task-specific scoring ceilings
TASK_THRESHOLDS = {
    "easy": 0.150,
    "medium": 0.175,
    "hard": 0.128
}

SCORE_MIN = 0.115

SYSTEM_PROMPT = (
    "You are a Geopolitical Crisis Logistics AI. Your goal is to stabilize "
    "hospital, emergency, and transport demands with minimal fuel waste."
)

def build_llama_instruction(state: dict, action: dict, reasoning: str) -> dict:
    user_prompt = (
        "Given the current crisis state, allocate fuel optimally to "
        "maximize stability while minimizing waste.\n\n"
        "State: " + json.dumps(state) + "\n"
        "What is the optimal fuel allocation for this step?"
    )
    assistant_response = {
        "reasoning": reasoning,
        "action": action
    }
    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "assistant": json.dumps(assistant_response)
    }

def run_expert_generation():
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    
    counts = {t: {"expert": 0, "near_expert": 0} for t in TASKS}
    initial_demands = {"hospital": 40, "emergency": 30, "transport": 20, "residential": 15}
    starting_fuels = {"easy": 160, "medium": 120, "hard": 80}

    print("Phase 2A Re-Balancing Started")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        pass

    try:
        with GlobalCrisisEnv() as env:
            for task in TASKS:
                consecutive_failures = 0
                current_expert_threshold = TASK_THRESHOLDS[task]
                target_experts = int(60 * 0.7)
                target_nears = int(60 * 0.3)

                while (counts[task]["expert"] + counts[task]["near_expert"]) < 60:
                    if counts[task]["expert"] < target_experts:
                        noise_level = 0.0
                    else:
                        noise_level = 0.1

                    if consecutive_failures >= 200:
                        current_expert_threshold = max(0.12, current_expert_threshold - 0.005)
                        print("Relaxing threshold for " + task)
                        consecutive_failures = 0

                    try:
                        obs = env.reset(task_id=task)
                    except Exception as e:
                        print("Connection Error: " + str(e))
                        return

                    initial_fuel = starting_fuels[task]
                    total_waste = 0
                    trajectory = []
                    cumulative_reward = 0.0

                    for step in range(1, 6):
                        remaining_steps = 5 - step
                        
                        state_capture = {
                            "initial_fuel": initial_fuel,
                            "remaining_steps": remaining_steps,
                            "fuel_available": obs.fuel_available,
                            "fuel_ratio": round(obs.fuel_available / initial_fuel, 3),
                            "hospital_demand": obs.hospital_demand,
                            "emergency_demand": obs.emergency_demand,
                            "transport_demand": obs.transport_demand,
                            "residential_demand": obs.residential_demand,
                            "hospital_ratio": round(obs.hospital_demand / 40, 3),
                            "emergency_ratio": round(obs.emergency_demand / 30, 3),
                            "transport_ratio": round(obs.transport_demand / 20, 3),
                            "residential_ratio": round(obs.residential_demand / 15, 3),
                            "bottleneck": obs.transport_demand > 5,
                            "step_fraction": round(step / 5.0, 1),
                            "cumulative_reward": round(cumulative_reward, 4)
                        }

                        action, reasoning = decide_action(obs, randomness=noise_level)

                        waste = (
                            max(0, action["fuel_to_hospital"] - obs.hospital_demand) +
                            max(0, action["fuel_to_emergency"] - obs.emergency_demand) +
                            max(0, action["fuel_to_transport"] - obs.transport_demand) +
                            max(0, action["fuel_to_residential"] - obs.residential_demand)
                        )
                        total_waste += waste

                        try:
                            obs = env.step(action)
                        except:
                            break
                            
                        cumulative_reward += obs.reward
                        trajectory.append({
                            "step": step,
                            "state": state_capture,
                            "action": action,
                            "reasoning": reasoning,
                            "reward": round(obs.reward, 4),
                            "instruction": build_llama_instruction(state_capture, action, reasoning)
                        })

                    if len(trajectory) < 5:
                        consecutive_failures += 1
                        continue

                    episode_score = round(cumulative_reward / 5.0, 4)
                    waste_ratio = total_waste / initial_fuel
                    
                    if episode_score >= 0.115 and waste_ratio <= 0.1:
                        quality = "expert" if episode_score >= current_expert_threshold else "near_expert"
                        
                        if quality == "expert" and counts[task]["expert"] >= target_experts:
                            consecutive_failures += 1
                            continue
                        if quality == "near_expert" and counts[task]["near_expert"] >= target_nears:
                            consecutive_failures += 1
                            continue

                        for point in trajectory:
                            point["state"]["outcome_score"] = episode_score

                        record = {
                            "task": task,
                            "quality": quality,
                            "score": episode_score,
                            "waste": total_waste,
                            "trajectory": trajectory
                        }
                        
                        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record) + "\n")
                        
                        counts[task][quality] += 1
                        print("Saved " + task + " " + quality + " - " + str(episode_score))
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1

    except Exception as e:
        print("Loop Error: " + str(e))

if __name__ == "__main__":
    run_expert_generation()
