# ===========================================================================
# PHASE 2A: Rule-Based Expert Planner (Bootstrap — TEMPORARY)
# ---------------------------------------------------------------------------
# Strategy derived from the real environment mechanics:
#
#   Hard mode: 80 fuel | 5 steps | demands: H=40, E=30, T=20, R=15
#   Bottleneck: if transport_demand > 5 → 40% efficiency on all other sectors
#
#   KEY INSIGHT: Must PACE fuel across all 5 steps. Never spend everything
#   in step 1. Each step = ~16 units budget on average.
#
# PHASE 2B: Replace the body of `decide_action()` with an LLM call.
#           The function signature must remain IDENTICAL.
# ===========================================================================

def decide_action(obs, past_reflections=None, randomness=0.0):
    """
    Decide fuel allocation for this step based on current observation.

    Strategy:
      1. Pace fuel: spend at most (fuel_available / steps_remaining) budget per step
      2. If transport bottleneck active (demand > 5): allocate enough to clear it
      3. Allocate remaining by priority: Hospital > Emergency > Transport > Residential
      4. Never over-allocate (no waste penalty)
      5. Apply ±randomness to allow for "near_expert" variations.

    PHASE 2B SWAP POINT — replace the body with: return llm_decide_action(obs, past_reflections)
    """
    import random

    fuel        = obs.fuel_available
    t_demand    = obs.transport_demand
    h_demand    = obs.hospital_demand
    e_demand    = obs.emergency_demand
    r_demand    = obs.residential_demand

    step_budget = max(16, fuel // 3)
    remaining = min(fuel, step_budget)

    allocs = {
        "fuel_to_hospital": 0,
        "fuel_to_emergency": 0,
        "fuel_to_transport": 0,
        "fuel_to_residential": 0
    }

    # Concise reasoning tracking
    reasons = []

    # ── Rule 1: If bottleneck active, prioritize clearing transport ─────────
    if t_demand > 5:
        amt = min(t_demand, remaining)
        allocs["fuel_to_transport"] = amt
        remaining -= amt
        reasons.append("Critical: Clearing transport bottleneck to restore global logistics efficiency")

    # ── Rule 2: Allocate remaining budget by weighted priority ────────────
    # Hospital first
    if remaining > 0 and h_demand > 0:
        amt = min(h_demand, remaining)
        allocs["fuel_to_hospital"] = amt
        remaining -= amt
        reasons.append(f"High Priority: Allocating {amt} units to stabilize critical hospital systems")

    # Emergency second
    if remaining > 0 and e_demand > 0:
        amt = min(e_demand, remaining)
        allocs["fuel_to_emergency"] = amt
        remaining -= amt
        reasons.append("Supporting emergency response teams for regional safety")

    # Transport maintenance
    if remaining > 0 and t_demand > 0 and t_demand <= 5:
        amt = min(t_demand, remaining)
        allocs["fuel_to_transport"] = amt
        remaining -= amt
        reasons.append("Maintaining transport infrastructure to prevent future bottlenecks")

    # Residential last
    if remaining > 0 and r_demand > 0:
        amt = min(r_demand, remaining)
        allocs["fuel_to_residential"] = amt
        remaining -= amt
        reasons.append("Managing residential energy distribution for civil stability")

    # Final logic for idle steps
    if sum(allocs.values()) == 0:
        if any(d > 0 for d in [h_demand, e_demand, t_demand, r_demand]):
            reasons.append("Resource management: Conserving fuel for future high-demand steps")
        else:
            reasons.append("Strategic stability: All demands satisfied, holding remaining reserves")

    # ── Rule 3: Apply controlled randomness (±10%) for Near-Expert ──────────
    if randomness > 0:
        reasons.append(f"Applying {int(randomness*100)}% strategic variance for exploration")
        total_fuel = sum(allocs.values())
        if total_fuel > 0:
            for k in allocs:
                noise = 1.0 + random.uniform(-randomness, randomness)
                allocs[k] = int(allocs[k] * noise)
            
            # Re-normalize to ensure we don't exceed budget or available fuel
            new_total = sum(allocs.values())
            if new_total > fuel: 
                factor = fuel / new_total
                for k in allocs: allocs[k] = int(allocs[k] * factor)

    reasoning = "; ".join(reasons) if reasons else "Maintaining baseline crisis stability"
    return allocs, reasoning
