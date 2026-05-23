# Failure Analysis Report

This report documents representative failure modes observed in the sequential crisis logistics environment. It classifies these failures, explains their technical root causes, and details the deterministic guardrail and runtime mitigations implemented in the `LLMPlanner` to ensure high-precision operations.

---

## 🔍 Failure Profile Catalog

### 1. Over-Allocation Failure
- **Scenario**: Representative Failure Scenario (Baseline Evaluation)
- **Agent**: `baseline` (Raw LLM)
- **Failure**: The agent outputted an allocation plan requesting a total of 110 units of fuel, while the global reserve stood at 80 units. This triggered a `LOGISTICS OVERLOAD` on the simulator, resulting in 0 actual fuel shipped and a step reward of `1e-6`.
- **Root Cause**: Autoregressive LLMs cannot guarantee mathematical summation consistency across outputs. Without hard constraints, they generate numbers that look superficially plausible but violate physical conservation of resources.
- **Mitigation**: Implemented an runtime **Reserves Guard**. If total allocations exceed the current reserve, the system dynamically scales down the allocation proportionally to match the available limit exactly.

---

### 2. Pacing Collapse (Greedy Exhaustion)
- **Scenario**: Representative Failure Scenario (Memory-only Evaluation)
- **Agent**: `memory` (LLM with memory reflection, no guardrails)
- **Failure**: The agent allocated 75 units of fuel on Step 1 of a 5-step game. While this cleared immediate demands, it left only 5 units of fuel for the remaining 4 steps, causing systemic collapse and near-zero rewards in later steps.
- **Root Cause**: Temporal horizon misalignment. LLMs are biased towards greedily resolving high-intensity demands in the current prompt context, failing to execute long-term multi-step budgeting.
- **Mitigation**: Implemented a **Soft Cap Pacing Guard**. On any step before the final planning step, the agent's total allocation is capped at `60%` of its available fuel reserve (unless the reserve is extremely low, e.g. < 10 units), forcing resource conservation.

---

### 3. Transport Bottleneck Miss
- **Scenario**: Representative Failure Scenario (Baseline/Memory Evaluation)
- **Agent**: `memory` (LLM with memory, no guardrails)
- **Failure**: Transport demand was 12 units (active bottleneck threshold is > 5). The agent allocated 10 to Hospital and 0 to Transport. Consequently, a `LOGISTICS BOTTLENECK` occurred, dropping overall sector delivery efficiency to 40% for the step.
- **Root Cause**: Priority logic distraction. When presented with multiple high-demand sectors in a single context window, the model defaults to basic prompt-level weights (e.g. prioritizing hospitals due to semantic associations) and ignores complex conditional rules.
- **Mitigation**: Implemented a **Priority Bottleneck Guard**. Before executing any plan, if `transport_demand > 5` and the LLM's transport allocation is insufficient, the guard overrides the plan, reallocating fuel to clear the transport bottleneck first.

---

### 4. Stochastic JSON Formatting Errors
- **Scenario**: Representative Failure Scenario (Guarded LLM)
- **Agent**: `guarded` (during initialization)
- **Failure**: The model generated markdown-style blocks, text wrapping around the JSON, or malformed braces:
  `[ACTION] {"fuel_to_hospital": 20, "fuel_to_emergency": 15... (truncated)`
  This resulted in JSON parsing crashes.
- **Root Cause**: Stochastic temperature fluctuations. Even at `temperature=0.1`, long generation prompts or complex reasoning text can introduce formatting tokens that break strict JSON structural requirements.
- **Mitigation**: Implemented a two-pronged defense:
  1. **Regex Extraction & Cleaners**: Explicitly strip wrapper tags and isolate JSON sub-braces.
  2. **Retry Loop with Fallback**: Allow up to 2 API attempts on parsing failures, falling back to a safe, proportional demand-based heuristic if both fail.

---

### 5. Demand Oversubscription (Guard Intervention)
- **Scenario**: Representative Failure Scenario (Guarded Agent)
- **Agent**: `guarded`
- **Failure**: The agent attempted to allocate 18 units of fuel to the Emergency sector, where the actual demand was only 8 units. This would have resulted in 10 units of fuel being completely wasted.
- **Root Cause**: Memory mismatch or estimation drift. The model failed to compute the exact demand offset after preceding steps, predicting allocations based on outdated context.
- **Mitigation**: Implemented a **Demand Clamping Guard**. The code dynamically intersects the LLM's requested allocation with the current sector demand:
  `action[k] = min(requested, actual_demand)`
  This eliminates excess fuel waste entirely, forcing the remaining fuel to stay in the reserve for subsequent steps.
