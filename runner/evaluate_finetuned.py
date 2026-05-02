# runner/evaluate_finetuned.py
import sys
import os
import json
import re
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.client import GlobalCrisisEnv
from agent.planner_llm import LLMPlanner

class FinetunedPlanner:
    def __init__(self, model_path="outputs/llama3_crisis_lora", base_model="unsloth/llama-3-8b-Instruct-bnb-4bit"):
        print(f"Loading tokenizer from {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        print(f"Loading fine-tuned model from {model_path}...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()
        self.history = []

    def _build_system_prompt(self):
        return (
            "You are a Geopolitical Crisis Logistics AI. Your goal is to stabilize hospital, emergency, and transport demands with minimal fuel waste."
        )

    def decide_action(self, obs, initial_fuel, remaining_steps, cumulative_reward, task_id="hard"):
        state_dict = {
            "initial_fuel": initial_fuel,
            "remaining_steps": remaining_steps,
            "fuel_available": obs.fuel_available,
            "fuel_ratio": round(obs.fuel_available / initial_fuel, 3),
            "hospital_demand": obs.hospital_demand,
            "emergency_demand": obs.emergency_demand,
            "transport_demand": obs.transport_demand,
            "residential_demand": obs.residential_demand,
            "hospital_ratio": round(obs.hospital_demand / 40.0, 3),
            "emergency_ratio": round(obs.emergency_demand / 30.0, 3),
            "transport_ratio": round(obs.transport_demand / 20.0, 3),
            "residential_ratio": round(obs.residential_demand / 15.0, 3),
            "bottleneck": obs.transport_demand > 5,
            "step_fraction": round((5 - remaining_steps) / 5.0, 2),
            "cumulative_reward": round(cumulative_reward, 4)
        }

        user_prompt = (
            f"Given the current crisis state, allocate fuel optimally to maximize stability while minimizing waste.\n\n"
            f"State: {json.dumps(state_dict)}\n"
            f"What is the optimal fuel allocation for this step?"
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        try:
            # Model should directly output the target JSON {"reasoning": "...", "action": {...}}
            parsed = json.loads(response)
            action = parsed.get("action", {})
            thought = parsed.get("reasoning", "No reasoning parsed.")
            
            # Very lightweight fallback just in case, but no heavy guards
            required_keys = ["fuel_to_hospital", "fuel_to_emergency", "fuel_to_transport", "fuel_to_residential"]
            for k in required_keys:
                if k not in action:
                    action[k] = 0
            
            invalid_flag = False

        except json.JSONDecodeError:
            print(f"[ERROR] Fine-tuned model produced invalid JSON: {response}")
            action = {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": 0, "fuel_to_residential": 0}
            thought = "Failed to parse JSON."
            invalid_flag = True

        print(f"\n[FT THOUGHT] {thought}")
        print(f"[FT ACTION] {action}")
        return action, thought, invalid_flag

def evaluate_agent(planner_instance, episodes=10, task_id="hard"):
    scores = []
    wastes = []
    bottlenecks_cleared = 0
    invalids = 0
    
    with GlobalCrisisEnv() as env:
        for ep in range(episodes):
            obs = env.reset(task_id=task_id)
            total_fuel_used = 0
            fuel_wasted = 0
            ep_invalid = 0
            
            # Reconstruct variables missing from raw obs
            starting_fuels = {"easy": 160, "medium": 120, "hard": 80}
            initial_fuel = starting_fuels[task_id]
            cumulative_reward = 0.0
            
            for step in range(1, 6):
                if obs.done:
                    break
                remaining_steps = 5 - step
                
                # Get Action
                if hasattr(planner_instance, "decide_action"):
                    # Handle different return signatures based on planner type
                    if isinstance(planner_instance, FinetunedPlanner):
                        res = planner_instance.decide_action(obs, initial_fuel, remaining_steps, cumulative_reward, task_id)
                    else:
                        res = planner_instance.decide_action(obs, task_id)
                        
                    if len(res) == 3:
                        action, thought, invalid_flag = res
                    else:
                        action, thought, invalid_flag, retries, r_p = res
                else:
                    action = {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": 0, "fuel_to_residential": 0}
                    invalid_flag = True

                if invalid_flag:
                    ep_invalid += 1
                
                # Math out waste
                step_fuel_used = sum(action.values())
                step_wasted = max(0, action.get("fuel_to_hospital", 0) - obs.hospital_demand) + \
                              max(0, action.get("fuel_to_emergency", 0) - obs.emergency_demand) + \
                              max(0, action.get("fuel_to_transport", 0) - obs.transport_demand) + \
                              max(0, action.get("fuel_to_residential", 0) - obs.residential_demand)
                
                obs = env.step(action)
                cumulative_reward += obs.reward
                total_fuel_used += step_fuel_used
                fuel_wasted += step_wasted

            scores.append(cumulative_reward / 5.0)
            wastes.append(fuel_wasted)
            invalids += ep_invalid
            
            if obs.transport_demand <= 5:
                bottlenecks_cleared += 1

    return {
        "avg_score": np.mean(scores),
        "max_score": np.max(scores),
        "avg_waste": np.mean(wastes),
        "bottleneck_clear_rate": bottlenecks_cleared / episodes,
        "invalid_actions": invalids
    }

def main():
    print("--- Phase 3 Evaluation Pipeline ---")
    
    results = {}
    
    print("\n[1/2] Evaluating Phase 2B Agent (Guarded)...")
    phase2_planner = LLMPlanner()
    results["Phase_2B"] = evaluate_agent(phase2_planner, episodes=5)
    
    print("\n[2/2] Evaluating Fine-Tuned Agent (Unguarded)...")
    try:
        ft_planner = FinetunedPlanner()
        results["Finetuned"] = evaluate_agent(ft_planner, episodes=5)
    except Exception as e:
        print(f"Could not load fine-tuned model: {e}")
        print("Note: You must train the model first using runner/fine_tune.py on Colab/GPU.")

    print("\n--- Final Comparison ---")
    for agent, metrics in results.items():
        print(f"\n{agent}:")
        print(f"  Avg Score: {metrics['avg_score']:.4f}")
        print(f"  Max Score: {metrics['max_score']:.4f}")
        print(f"  Avg Waste: {metrics['avg_waste']:.2f}")
        print(f"  Bottleneck Clear Rate: {metrics['bottleneck_clear_rate'] * 100}%")
        print(f"  Invalid Actions: {metrics['invalid_actions']}")

if __name__ == "__main__":
    main()
