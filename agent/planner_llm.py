import os
import json
import re
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
            "You are a Geopolitical Crisis Logistics Agent. Your mission is to stabilize "
            "hospital, emergency, transport, and residential demands with minimal fuel waste.\n"
            "Prioritize: Hospital > Emergency > Transport > Residential.\n"
            "Critical: If transport demand > 5, it's a bottleneck; you MUST allocate enough fuel "
            "to clear it or global efficiency will drop.\n\n"
            "Respond using this EXACT format:\n"
            "[THOUGHT] Your step-by-step reasoning.\n"
            "[ACTION] {\"fuel_to_hospital\": int, \"fuel_to_emergency\": int, \"fuel_to_transport\": int, \"fuel_to_residential\": int}\n"
            "Note: The [ACTION] must be strict JSON. Do not include anything else in the JSON block."
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

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.1, # Keep it deterministic as requested
            )
            response = chat_completion.choices[0].message.content
            
            # Extract Thought and Action
            thought_match = re.search(r"\[THOUGHT\](.*?)(?=\[ACTION\]|$)", response, re.S)
            action_match = re.search(r"\[ACTION\]\s*(\{.*?\})", response, re.S)

            thought = thought_match.group(1).strip() if thought_match else "No thought provided."
            action_json = action_match.group(1).strip() if action_match else "{}"

            print(f"\n[THOUGHT] {thought}")
            
            try:
                action = json.loads(action_json)
                # Validation / Basic cleaning
                required_keys = ["fuel_to_hospital", "fuel_to_emergency", "fuel_to_transport", "fuel_to_residential"]
                if not all(k in action for k in required_keys):
                    raise ValueError("Missing keys in JSON")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[ERROR] JSON Parsing failed: {e}. Using fallback action.")
                action = self._get_fallback_action(current_state)

            print(f"[ACTION] {json.dumps(action)}")
            
            # Record in history (will be updated with reward later if needed, but for now just state/action)
            self.history.append({"state": current_state, "action": action})
            return action, thought

        except Exception as e:
            print(f"[CRITICAL ERROR] Groq API call failed: {e}")
            return self._get_fallback_action(current_state), "Error in API call."

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
