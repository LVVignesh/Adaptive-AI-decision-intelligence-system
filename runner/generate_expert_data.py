import sys
import os
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.client import GlobalCrisisEnv
from agent.planner import decide_action

# ── Configuration ───────────────────────────────────────────────────────────
TARGET_PER_TASK  = 60           # aim for 60 quality episodes per task (Easy/Med/Hard)
SCORE_MIN        = 0.12         # minimum score to keep any data (Near-Expert)
EXPERT_THRESHOLD = 0.18         # scores >=0.18 are labeled 'expert'
WASTE_THRESHOLD  = 0.1          # max 10% total fuel waste
OUTPUT_PATH      = "logs/expert_trajectories.jsonl"
TASKS            = ["easy", "medium", "hard"]
MAX_FAIL_SAFES   = 100          # if after 100 resets no expert found, relax threshold
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a Geopolitical Crisis Logistics AI. Your goal is to stabilize "
    "hospital, emergency, and transport demands with minimal fuel waste."
)

def build_llama_instruction(state: dict, action: dict, reasoning: str) -> dict:
    """Format a state+action+reasoning triplet for elite fine-tuning."""
    # Enriched Prompt suggested by review
    user_prompt = (
        f"Given the current crisis state, allocate fuel optimally to "
        f"maximize stability while minimizing waste.\n\n"
        f"State: {json.dumps(state)}\n"
        f"What is the optimal fuel allocation for this step?"
    )
    assistant_response = {
        "reasoning": reasoning,
        "action": action
    }
    return {
        "system":    SYSTEM_PROMPT,
        "user":      user_prompt,
        "assistant": json.dumps(assistant_response)
    }

def run_expert_generation():
    os.makedirs("logs", exist_ok=True)
    
    # Initialize counts
    counts = {t: {"expert": 0, "near_expert": 0} for t in TASKS}
    
    # Baseline demands for normalization
    initial_demands = {
        "hospital": 40, "emergency": 30, "transport": 20, "residential": 15
    }
    starting_fuels = {"easy": 160, "medium": 120, "hard": 80}

    print(f"Starting Phase 2A ELITE Expert Generation (70/30 Distribution)...\n")
    print(f"Thresholds: Expert >= {EXPERT_THRESHOLD}, Near-Expert >= {SCORE_MIN}")

    # Overwrite dataset for purity
    with open(OUTPUT_PATH, "w") as f:
        pass

    with GlobalCrisisEnv() as env:
        for task in TASKS:
            consecutive_failures = 0
            current_expert_threshold = EXPERT_THRESHOLD

            while (counts[task]["expert"] + counts[task]["near_expert"]) < TARGET_PER_TASK:
                # ── Distribution Control ──────────────────────────────────────
                # Target: 70% Expert, 30% Near-Expert
                total_for_task = counts[task]["expert"] + counts[task]["near_expert"]
                
                # If we have enough near-experts relative to experts, hunt for pure experts
                expert_ratio = counts[task]["expert"] / (total_for_task + 1e-6)
                needs_expert = expert_ratio < 0.7
                
                # Apply noise only if we explicitly want to fill 'near-expert' capacity
                noise_level = 0.1 if (not needs_expert and random.random() > 0.5) else 0.0

                if consecutive_failures >= MAX_FAIL_SAFES:
                    current_expert_threshold = max(0.15, current_expert_threshold - 0.01)
                    print(f"⚠️  Task {task}: High failure rate. Relaxing threshold to {current_expert_threshold:.2f}")
                    consecutive_failures = 0

                obs = env.reset(task_id=task)
                
                initial_fuel = starting_fuels[task]
                total_waste = 0
                trajectory = []
                cumulative_reward = 0.0

                step = 0
                while not obs.done:
                    step += 1
                    remaining_steps = 5 - step
                    
                    # ── Capture ELITE state ─────────────────────────────────────
                    state_capture = {
                        "total_fuel_initial": initial_fuel,
                        "remaining_steps":    remaining_steps,
                        "fuel_available":     obs.fuel_available,
                        "fuel_ratio":        round(obs.fuel_available / initial_fuel, 3),
                        "hospital_demand":    obs.hospital_demand,
                        "emergency_demand":   obs.emergency_demand,
                        "transport_demand":   obs.transport_demand,
                        "residential_demand": obs.residential_demand,
                        "hospital_ratio":    round(obs.hospital_demand / initial_demands["hospital"], 3),
                        "emergency_ratio":   round(obs.emergency_demand / initial_demands["emergency"], 3),
                        "transport_ratio":   round(obs.transport_demand / initial_demands["transport"], 3),
                        "residential_ratio": round(obs.residential_demand / initial_demands["residential"], 3),
                        "bottleneck_active":  obs.transport_demand > 5,
                        "step_fraction":      round(step / 5.0, 1),
                        "cumulative_reward":  round(cumulative_reward, 4)
                    }

                    action, reasoning = decide_action(obs, randomness=noise_level)

                    # ── Waste Calculation ───────────────────────────────────────
                    step_waste = (
                        max(0, action["fuel_to_hospital"] - obs.hospital_demand) +
                        max(0, action["fuel_to_emergency"] - obs.emergency_demand) +
                        max(0, action["fuel_to_transport"] - obs.transport_demand) +
                        max(0, action["fuel_to_residential"] - obs.residential_demand)
                    )
                    total_waste += step_waste

                    obs = env.step(action)
                    cumulative_reward += obs.reward

                    trajectory.append({
                        "step": step,
                        "state": state_capture,
                        "action": action,
                        "reasoning": reasoning,
                        "reward": round(obs.reward, 4),
                        "instruction": build_llama_instruction(state_capture, action, reasoning)
                    })

                episode_score = round(cumulative_reward / 5.0, 4)
                
                # ── Apply Strict Filters ──────────────────────────────────────
                waste_ratio = total_waste / initial_fuel
                
                if episode_score >= SCORE_MIN and waste_ratio <= WASTE_THRESHOLD:
                    quality = "expert" if episode_score >= current_expert_threshold else "near_expert"
                    
                    # Check if we still need this quality type for balancing
                    if quality == "near_expert" and not needs_expert and total_for_task > 10:
                        # If we have enough near-experts, stay in 'expert hunt'
                        consecutive_failures += 1
                        continue

                    # Enrich trajectory steps with the FINAL score (outcome association)
                    for point in trajectory:
                        point["state"]["episode_score_outcome"] = episode_score

                    record = {
                        "task_id": task,
                        "quality_level": quality,
                        "episode_score": episode_score,
                        "total_waste": total_waste,
                        "trajectory": trajectory
                    }
                    
                    with open(OUTPUT_PATH, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    
                    counts[task][quality] += 1
                    total_saved = sum(sum(q.values()) for q in counts.values())
                    print(f"[SAVED] [{task.upper()}] Score: {episode_score} Quality: {quality} (Expert Count: {counts[task]['expert']})")
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

    print(f"\nPhase 2A ELITE Complete. Final Counts:")
    for task, qs in counts.items():
        print(f"  - {task}: {qs['expert']} experts, {qs['near_expert']} near_experts")


if __name__ == "__main__":
    run_expert_generation()
