import sys, os, json, torch, traceback
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.client import GlobalCrisisEnv

class FinetunedHybridPlanner:
    def __init__(self, model_path="outputs/llama3_crisis_lora", base_model="unsloth/llama-3-8b-Instruct-bnb-4bit"):
        print("Loading Fine-Tuned Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if os.path.exists(model_path):
            print(f"Applying LoRA adapter from {model_path}...")
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            print("Warning: Model weights not found locally. Using base model.")
            self.model = base
            
        self.model.eval()
        print("Model ready!")

    def decide_action(self, obs, initial_fuel, remaining_steps, cumulative_reward, task_id="hard"):
        # Zero Fuel Guard
        if obs.fuel_available <= 0:
            print("\n[THOUGHT] Fuel exhausted. Waiting for next step.")
            return {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": 0, "fuel_to_residential": 0}, "Fuel exhausted.", False, 0, True

        state_dict = {
            "fuel_available": obs.fuel_available,
            "hospital_demand": obs.hospital_demand,
            "emergency_demand": obs.emergency_demand,
            "transport_demand": obs.transport_demand,
            "residential_demand": obs.residential_demand,
        }
        messages = [
            {"role": "system", "content": "You are a Crisis Logistics AI. Output only valid JSON with keys: reasoning, action."},
            {"role": "user", "content": f"State: {json.dumps(state_dict)}\\nAllocate fuel optimally."}
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        invalid_flag = False
        json_retry_count = 0
        action = {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": 0, "fuel_to_residential": 0}
        thought = "JSON Error"

        try:
            parsed = json.loads(response)
            action_raw = parsed.get("action", {})
            thought = parsed.get("reasoning", "")
            
            # Merge with default 0 to ensure all keys exist
            action = {k: action_raw.get(k, 0) for k in action}
        except:
            invalid_flag = True
            json_retry_count = 1
            print("[ERROR] JSON Parsing failed. Using fallback.")
            
        # ==========================================
        # GUARDRAILS APPLICATION (Deterministic Logic)
        # ==========================================
        fuel_available = obs.fuel_available
        
        # 1. Clamp allocations to not exceed sector demands to eliminate waste
        action["fuel_to_hospital"] = min(action["fuel_to_hospital"], obs.hospital_demand)
        action["fuel_to_emergency"] = min(action["fuel_to_emergency"], obs.emergency_demand)
        action["fuel_to_transport"] = min(action["fuel_to_transport"], obs.transport_demand)
        action["fuel_to_residential"] = min(action["fuel_to_residential"], obs.residential_demand)
        
        # Ensure no negative values
        for k in action: action[k] = max(0, action[k])

        # 2. Soft Cap Pacing
        max_this_step = fuel_available * 0.6 if fuel_available > 10 else fuel_available
        total_requested = sum(action.values())
        
        if total_requested > max_this_step:
            print(f"[GUARD] Scaling action from {total_requested} to {int(max_this_step)} for pacing.")
            scale = max_this_step / total_requested
            for key in action:
                action[key] = int(action[key] * scale)
                
        # 3. Priority Guard: If bottleneck is active but LLM ignored it
        if obs.transport_demand > 5 and action["fuel_to_transport"] < min(obs.transport_demand, int(max_this_step)):
            print("[GUARD] Enforcing PRIORITY RULE for Transport Bottleneck.")
            invalid_flag = True
            needed = min(obs.transport_demand, int(max_this_step))
            action = {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": needed, "fuel_to_residential": 0}

        return action, thought, invalid_flag, json_retry_count, True

def evaluate_agent(planner_instance, episodes=5, task_id="hard"):
    scores = []
    total_waste = 0
    total_bottleneck_clear_attempts = 0
    successful_bottleneck_clears = 0
    total_invalid_actions = 0
    total_json_retries = 0

    with GlobalCrisisEnv() as env:
        for ep in range(episodes):
            obs = env.reset(task_id=task_id)
            cum_reward = 0.0
            print(f"\\n--- Episode {ep+1} ---")
            for step in range(1, 6):
                if obs.done: break
                
                # Check bottleneck state before action
                had_bottleneck = obs.transport_demand > 5
                if had_bottleneck:
                    total_bottleneck_clear_attempts += 1
                    
                action, _, invalid_flag, json_retry_count, _ = planner_instance.decide_action(obs, 80, 5-step, cum_reward, task_id)
                
                if invalid_flag: total_invalid_actions += 1
                total_json_retries += json_retry_count
                
                # Estimate waste before step
                step_waste = max(0, sum(action.values()) - (obs.hospital_demand + obs.emergency_demand + obs.transport_demand + obs.residential_demand))
                total_waste += step_waste
                
                obs = env.step(action)
                cum_reward += obs.reward
                
                # Check bottleneck resolution
                if had_bottleneck and obs.transport_demand == 0:
                    successful_bottleneck_clears += 1
                
            scores.append(cum_reward / 5.0)
            print(f"  Episode {ep+1} Score: {cum_reward/5.0:.4f}")
            
    bottleneck_rate = (successful_bottleneck_clears / total_bottleneck_clear_attempts * 100) if total_bottleneck_clear_attempts > 0 else 100.0

    return {
        "avg_score": np.mean(scores),
        "best_score": np.max(scores),
        "fuel_waste": total_waste,
        "bottleneck_clear_rate": bottleneck_rate,
        "invalid_actions": total_invalid_actions,
        "json_retries": total_json_retries
    }

def main():
    print("=== Round 2 ReThesis: Hybrid Agent Evaluation Pipeline ===")
    try:
        hybrid_planner = FinetunedHybridPlanner()
        results = evaluate_agent(hybrid_planner)
        print("\\n=== FINAL HYBRID AGENT METRICS ===")
        print(f"Average Score:          {results['avg_score']:.4f}")
        print(f"Best Episode Score:     {results['best_score']:.4f}")
        print(f"Total Fuel Waste:       {results['fuel_waste']}")
        print(f"Bottleneck Clear Rate:  {results['bottleneck_clear_rate']:.1f}%")
        print(f"Invalid Guard Triggers: {results['invalid_actions']}")
        print(f"JSON Parsing Errors:    {results['json_retries']}")
        print("==================================")
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
