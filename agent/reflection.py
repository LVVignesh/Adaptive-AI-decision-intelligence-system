def generate_reflection(state, action, reward):
    # Extremely simple heuristic reflection for the baseline.
    if float(reward) < 0.2:
        return f"Bad decision: {action}. Likely wasted fuel or ignored bottleneck."
    else:
        return f"Good decision: {action}. effectively allocated fuel."
