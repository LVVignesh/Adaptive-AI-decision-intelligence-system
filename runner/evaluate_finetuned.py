import sys, os, json, torch, traceback
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.client import GlobalCrisisEnv

class FinetunedPlanner:
    def __init__(self, model_path="outputs/llama3_crisis_lora", base_model="unsloth/llama-3-8b-Instruct-bnb-4bit"):
        print("Loading Fine-Tuned Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Load base model in 4-bit (standard HF way for local execution)
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

        # Use the string-first tokenization fix to avoid BatchEncoding errors
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

        try:
            parsed = json.loads(response)
            return parsed.get("action", {}), parsed.get("reasoning", ""), False
        except:
            return {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": 0, "fuel_to_residential": 0}, "JSON Error", True

def evaluate_agent(planner_instance, episodes=5, task_id="hard"):
    scores = []
    with GlobalCrisisEnv() as env:
        for ep in range(episodes):
            obs = env.reset(task_id=task_id)
            cum_reward = 0.0
            for step in range(1, 6):
                if obs.done: break
                action, _, _ = planner_instance.decide_action(obs, 80, 5-step, cum_reward, task_id)
                obs = env.step(action)
                cum_reward += obs.reward
            scores.append(cum_reward / 5.0)
            print(f"  Episode {ep+1}: {cum_reward/5.0:.4f}")
    return {"avg_score": np.mean(scores)}

def main():
    print("--- Phase 3 Evaluation Pipeline ---")
    try:
        ft_planner = FinetunedPlanner()
        results = evaluate_agent(ft_planner)
        print(f"\\n=== Fine-Tuned Agent Avg Score: {results['avg_score']:.4f} ===")
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()
