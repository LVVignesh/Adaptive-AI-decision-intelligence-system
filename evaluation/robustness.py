# evaluation/robustness.py
import sys
import os
import argparse
import time
import json
import csv
from datetime import datetime
import numpy as np

# Ensure root of the workspace is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.client import GlobalCrisisEnv
from agent.planner_llm import LLMPlanner
from evaluation.config import DEFAULT_EPISODES, RUNS_DIR, EVAL_DIR

def run_robustness_test(episodes, fuel_limits):
    print(f"\n==================================================")
    print(f"            ROBUSTNESS TESTING START              ")
    print(f"==================================================")
    
    planner = LLMPlanner()
    results = {}
    
    with GlobalCrisisEnv() as env:
        for fuel in fuel_limits:
            print(f"\nTesting with Initial Fuel Limit: {fuel}")
            scores = []
            total_waste = 0
            bottleneck_attempts = 0
            bottleneck_cleared = 0
            start_time = time.time()
            
            for ep in range(episodes):
                # Set seed for reproducibility
                seed = 2000 + ep
                obs = env.reset(task_id="hard", fuel=fuel, seed=seed)
                cum_reward = 0.0
                planner.history = []
                
                for step in range(1, 6):
                    if obs.done:
                        break
                    
                    # Bottleneck tracking
                    had_bottleneck = obs.transport_demand > 5
                    if had_bottleneck:
                        bottleneck_attempts += 1
                        
                    # Guarded configuration
                    action, thought, invalid_flag, json_retry_count, _ = planner.decide_action(
                        obs,
                        task_id="hard",
                        use_memory=True,
                        use_experts=True,
                        use_history=True,
                        use_guardrails=True
                    )
                    
                    step_waste = sum(max(0, action[k] - getattr(obs, f"{k.split('_')[-1]}_demand")) for k in action)
                    total_waste += step_waste
                    
                    obs = env.step(action)
                    cum_reward += obs.reward
                    
                    if had_bottleneck and obs.transport_demand == 0:
                        bottleneck_cleared += 1
                        
                    time.sleep(1.0)
                    
                scores.append(cum_reward / 5.0)
                
            elapsed_time = time.time() - start_time
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            score_std = np.std(scores)
            # 95% CI: mean ± 1.96 * (std / sqrt(n))
            score_ci95 = 1.96 * (score_std / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
            bottleneck_rate = (bottleneck_cleared / bottleneck_attempts * 100) if bottleneck_attempts > 0 else 100.0
            
            results[fuel] = {
                "avg_score": float(avg_score),
                "max_score": float(max_score),
                "score_std": float(score_std),
                "score_ci95": float(score_ci95),
                "waste": int(total_waste),
                "bottleneck_rate": float(bottleneck_rate),
                "runtime": float(elapsed_time),
                "fuel_limit": fuel
            }
            
            # Save run metadata log
            run_data = {
                "agent": "guarded_robustness",
                "episodes": episodes,
                "score": avg_score,
                "waste": total_waste,
                "runtime": elapsed_time,
                "fuel": float(fuel),
                "timestamp": datetime.now().isoformat(),
                "hardware": "CPU (Local API)"
            }
            run_file = os.path.join(RUNS_DIR, f"run_robustness_fuel_{fuel}_{int(time.time())}.json")
            with open(run_file, "w") as f:
                json.dump(run_data, f, indent=4)
                
            print(f"Fuel {fuel} Results -> Avg Score: {avg_score:.4f} | Bottleneck Clear: {bottleneck_rate:.1f}% | Waste: {total_waste}")

    return results

def main():
    parser = argparse.ArgumentParser(description="AI Engineering Quality Evaluation Platform - Robustness Testing")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes to run per limit")
    args = parser.parse_args()

    fuel_limits = [40, 60, 80, 100]
    
    results = run_robustness_test(args.episodes, fuel_limits)
    
    # Save to CSV
    csv_file = os.path.join(EVAL_DIR, "robustness_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fuel_limit", "avg_score", "score_std", "score_ci95", "max_score", "waste", "bottleneck_rate", "runtime"])
        for fuel in fuel_limits:
            res = results[fuel]
            writer.writerow([
                fuel,
                res["avg_score"],
                res["score_std"],
                res["score_ci95"],
                res["max_score"],
                res["waste"],
                res["bottleneck_rate"],
                res["runtime"]
            ])
            
    print(f"\nRobustness test completed. Summary saved to: {csv_file}")

if __name__ == "__main__":
    main()
