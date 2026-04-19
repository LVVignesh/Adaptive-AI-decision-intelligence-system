import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.client import GlobalCrisisEnv

try:
    with GlobalCrisisEnv() as env:
        print("Resetting env...")
        obs = env.reset(task_id="easy")
        print(f"Initial Obs: {obs}")
        
        action = {
            "fuel_to_hospital": 20,
            "fuel_to_emergency": 0,
            "fuel_to_transport": 20,
            "fuel_to_residential": 0
        }
        print(f"Stepping with action: {action}")
        obs = env.step(action)
        print(f"Post-step Obs: {obs}")
        print(f"Reward: {obs.reward}")
except Exception as e:
    print(f"ERROR: {e}")
