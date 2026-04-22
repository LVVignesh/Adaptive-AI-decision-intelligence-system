import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv
from agent.expert_provider import ExpertProvider
from agent.memory import Memory

load_dotenv()

class LLMPlanner:
    def __init__(self, model=None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self.expert_provider = ExpertProvider()
        self.memory = Memory()
        self.history = []  # Stores last 2 steps: {"state": ..., "action": ..., "reward": ...}

    def _build_system_prompt(self):
        return (
            "You are a Geopolitical Crisis Logistics Agent. You are in a 5-STEP STRATEGY GAME.\n"
            "STRATEGIC OBJECTIVE: Maximize total mission success over 5 steps. Do NOT act greedily per step. Plan fuel usage across all steps.\n\n"
            "CRITICAL RULES:\n"
            "1. STEP-WISE PLANNING: You have 5 steps. Do NOT use all fuel in one step. Distribute fuel across steps.\n"
            "2. PRIORITY RULE: If transport demand > 5, this is a BOTTLENECK. You MUST allocate to transport FIRST before anything else.\n"
            "3. HIERARCHY: After transport bottleneck is cleared, prioritize Hospital > Emergency > Residential.\n"
            "4. MATH & WASTE: Never exceed available fuel. NEVER allocate more fuel than a sector's demand. Avoid any unnecessary fuel usage.\n"
            "5. REASONING: You MUST provide a reasoning (THOUGHT) for every step. Empty reasoning is not allowed.\n\n"
            "Respond using this EXACT format:\n"
            "[THOUGHT] Your detailed step-by-step reasoning.\n"
            "[ACTION] {\"fuel_to_hospital\": int, \"fuel_to_emergency\": int, \"fuel_to_transport\": int, \"fuel_to_residential\": int}\n"
        )

    def _get_context(self, task_id, current_state_dict):
        # 1. Expert Examples
        experts = self.expert_provider.get_top_examples(task_id, k=2)
        expert_context = "\n".join([self.expert_provider.format_example_for_prompt(e) for e in experts])

        # 2. Memory Reflections
        state_str = json.dumps(current_state_dict)
        memories = self.memory.query(state_str, k=2)
        memory_context = "\n".join(memories) if memories else "No relevant memories found."

        # 3. Recent History (Last 2 steps)
        history_context = ""
        for i, h in enumerate(self.history[-2:]):
            history_context += f"Previous Step {i+1}:\nState: {json.dumps(h['state'])}\nAction: {json.dumps(h['action'])}\n"

        return expert_context, memory_context, history_context

    def decide_action(self, obs, task_id):
        # Convert observation object to normalized dict
        current_state = {
            "fuel_available": obs.fuel_available,
            "hospital_demand": obs.hospital_demand,
            "emergency_demand": obs.emergency_demand,
            "transport_demand": obs.transport_demand,
            "residential_demand": obs.residential_demand,
            "bottleneck": obs.transport_demand > 5
        }

        invalid_flag = False
        json_retry_count = 0
        reasoning_present = False

        # FIX 4: ZERO FUEL GUARD
        if obs.fuel_available <= 0:
            print("\n[THOUGHT] Fuel exhausted. Waiting for next step.")
            return {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": 0, "fuel_to_residential": 0}, "Fuel exhausted.", False, 0, True

        expert_ctx, memory_ctx, history_ctx = self._get_context(task_id, current_state)

        user_prompt = f"""
--- EXPERT DEMONSTRATIONS ---
{expert_ctx}

--- PAST REFLECTIONS ---
{memory_ctx}

--- RECENT HISTORY ---
{history_ctx}

--- CURRENT STATE ---
{json.dumps(current_state)}

Task: Decide the optimal fuel allocation for the current state.
Remember: Clear bottlenecks (Transport > 5) immediately.
"""

        max_retries = 2
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self._build_system_prompt()},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    temperature=0.1,
                )
                response = chat_completion.choices[0].message.content
                
                # Extract Thought and Action
                thought_match = re.search(r"\[THOUGHT\](.*?)(?=\[ACTION\]|$)", response, re.S)
                action_match = re.search(r"\[ACTION\]\s*(\{.*?\})", response, re.S)

                thought = thought_match.group(1).strip() if thought_match else ""
                if not thought or thought.lower() == "no thought provided.":
                    thought = "No thought provided."
                    reasoning_present = False
                else:
                    reasoning_present = True

                action_json = action_match.group(1).strip() if action_match else "{}"

                print(f"\n[THOUGHT] {thought}")
                
                try:
                    action = json.loads(action_json)
                    required_keys = ["fuel_to_hospital", "fuel_to_emergency", "fuel_to_transport", "fuel_to_residential"]
                    if not all(k in action for k in required_keys):
                        raise ValueError("Missing keys in JSON")
                    
                    # FIX 1 & 5 & 6: HARD CONSTRAINT LAYER, SOFT CAP & DEMAND CLAMPING
                    fuel_available = obs.fuel_available
                    
                    # Clamp allocations to not exceed sector demands to eliminate waste
                    action["fuel_to_hospital"] = min(action["fuel_to_hospital"], obs.hospital_demand)
                    action["fuel_to_emergency"] = min(action["fuel_to_emergency"], obs.emergency_demand)
                    action["fuel_to_transport"] = min(action["fuel_to_transport"], obs.transport_demand)
                    action["fuel_to_residential"] = min(action["fuel_to_residential"], obs.residential_demand)
                    
                    # Ensure no negative values
                    for k in action: action[k] = max(0, action[k])

                    max_this_step = fuel_available * 0.6 if fuel_available > 10 else fuel_available
                    total_requested = sum(action.values())
                    
                    if total_requested > max_this_step:
                        print(f"[GUARD] Scaling action from {total_requested} to {int(max_this_step)} for pacing.")
                        scale = max_this_step / total_requested
                        for key in action:
                            action[key] = int(action[key] * scale)
                            
                    # Priority Guard: If bottleneck is active but LLM ignored it
                    if obs.transport_demand > 5 and action["fuel_to_transport"] < min(obs.transport_demand, max_this_step):
                        print("[GUARD] Enforcing PRIORITY RULE for Transport Bottleneck.")
                        invalid_flag = True
                        # Reallocate to transport first
                        needed = min(obs.transport_demand, int(max_this_step))
                        action = {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": needed, "fuel_to_residential": 0}

                    # Valid Action Achieved
                    print(f"[ACTION] {json.dumps(action)}")
                    self.history.append({"state": current_state, "action": action})
                    return action, thought, invalid_flag, json_retry_count, reasoning_present

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[ERROR] JSON Parsing failed on attempt {attempt+1}: {e}.")
                    if attempt < max_retries - 1:
                        json_retry_count += 1
                        print("Retrying...")
                        continue
                    else:
                        print("Using fallback action.")
                        invalid_flag = True
                        action = self._get_fallback_action(current_state)
                        print(f"[ACTION] {json.dumps(action)}")
                        self.history.append({"state": current_state, "action": action})
                        return action, thought, invalid_flag, json_retry_count, reasoning_present

            except Exception as e:
                # Catch rate limits and wait
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    print(f"[CRITICAL WARNING] Rate Limit hit. Waiting 3 seconds... (Attempt {attempt+1})")
                    time.sleep(3)
                    if attempt < max_retries - 1:
                        continue
                print(f"[CRITICAL ERROR] API call failed: {e}")
                invalid_flag = True
                action = self._get_fallback_action(current_state)
                return action, "Error in API call.", invalid_flag, json_retry_count, False

    def _get_fallback_action(self, state):
        """Safe fallback: distribute fuel proportionally or clear bottleneck."""
        fuel = state["fuel_available"]
        if state["transport_demand"] > 5:
            # Priority: Clear bottleneck
            t_alloc = min(fuel, state["transport_demand"])
            return {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": t_alloc, "fuel_to_residential": 0}
        
        # Simple proportional split
        total_demand = state["hospital_demand"] + state["emergency_demand"] + state["transport_demand"] + state["residential_demand"]
        if total_demand == 0: return {"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": 0, "fuel_to_residential": 0}
        
        ratio = fuel / total_demand
        return {
            "fuel_to_hospital": int(state["hospital_demand"] * ratio),
            "fuel_to_emergency": int(state["emergency_demand"] * ratio),
            "fuel_to_transport": int(state["transport_demand"] * ratio),
            "fuel_to_residential": int(state["residential_demand"] * ratio)
        }
