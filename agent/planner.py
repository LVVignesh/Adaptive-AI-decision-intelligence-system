import random

def decide_action(observation, past_reflections):
    # SUPER simple baseline (we improve later)

    # If memory exists and we remembered "too low reward", we could try a fixed safe baseline
    if past_reflections and any("Good decision" in r for r in past_reflections):
        # In a real heuristic we would parse the best action, but here we just adopt a "safer" fixed allocation
        return {
            "fuel_to_hospital": 25,
            "fuel_to_emergency": 25,
            "fuel_to_transport": 25,
            "fuel_to_residential": 5
        }

    # Otherwise random (exploration)
    return {
        "fuel_to_hospital": random.randint(10, 40),
        "fuel_to_emergency": random.randint(10, 30),
        "fuel_to_transport": random.randint(10, 30),
        "fuel_to_residential": random.randint(0, 20)
    }
