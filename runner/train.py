import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.client import GlobalCrisisEnv
from agent.memory import Memory
from agent.planner import decide_action
from agent.reflection import generate_reflection
import pandas as pd

memory = Memory()
scores = []

EPISODES = 20

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

with GlobalCrisisEnv() as env:
    for ep in range(EPISODES):
        obs = env.reset(task_id="hard")

        total_reward = 0

        while not obs.done:
            # simple state string for the baseline
            state_str = f"Transport Demand: {obs.transport_demand}, Fuel: {obs.fuel_available}, Hospital: {obs.hospital_demand}, Emergency: {obs.emergency_demand}"

            past = memory.query(state_str)

            action = decide_action(obs, past)

            obs = env.step(action)

            reflection = generate_reflection(state_str, action, obs.reward)

            memory.add(state_str, reflection)

            total_reward += obs.reward

        # Average the 5 step rewards
        score = total_reward / 5.0
        scores.append(score)

        print(f"Episode {ep} -> Score: {score:.3f}")

# Save results
df = pd.DataFrame({"episode": list(range(EPISODES)), "score": scores})
df.to_csv("logs/scores.csv", index=False)
print("Finished training. Results saved to logs/scores.csv")
