import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.client import GlobalCrisisEnv

def test_max_score():
    with GlobalCrisisEnv() as env:
        # Easy mode has 160 fuel, enough to satisfy 105 demand comfortably.
        obs = env.reset(task_id="easy")
        total_reward = 0
        
        # Step 1: satisfying everything at once
        action = {
            "fuel_to_hospital": 40,
            "fuel_to_emergency": 30,
            "fuel_to_transport": 20,
            "fuel_to_residential": 15
        }
        obs = env.step(action)
        print(f"Step 1 Reward: {obs.reward}")
        total_reward += obs.reward
        
        # Steps 2-5: empty
        for i in range(2, 6):
            obs = env.step({"fuel_to_hospital": 0, "fuel_to_emergency": 0, "fuel_to_transport": 0, "fuel_to_residential": 0})
            print(f"Step {i} Reward: {obs.reward}")
            total_reward += obs.reward
            
        final_score = total_reward / 5.0
        print(f"TOTAL SCORE: {total_reward}")
        print(f"AVERAGE (MISSION) SCORE: {final_score}")
        print(f"MESSAGE: {obs.message}")

if __name__ == "__main__":
    test_max_score()
