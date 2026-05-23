# Evaluation Methodology & Assumptions

This document outlines the key assumptions, fallback behaviors, and methodological decisions underlying the evaluation framework.

---

## Historical Metrics & Data Origin

### Source of Baseline Data
- **Historical benchmark values** for agents (`baseline`, `memory`, `guarded`, `finetuned`, `hybrid`) originate from Phase 3 Re-Thesis evaluation conducted on GPU infrastructure.
- These values are stored in `evaluation/historical_results.json` with metadata:
  - `version`: Format version for future updates
  - `generated_at`: Timestamp of JSON creation
  - `source`: Attribution to original evaluation phase

### When Historical Metrics Are Used
Historical metrics serve as **fallback references** in two scenarios:

1. **GPU-Dependent Models**: When fine-tuned LoRA weights (`finetuned`, `hybrid` agents) are unavailable due to:
   - Missing GPU/CUDA environment
   - Absence of checkpoint files
   - Incomplete model loading setup

2. **Mixed Evaluation Paradigm**: The `ablation.py` script uses:
   - **Live evaluation** (CPU-based) for `baseline`, `no_memory`, `no_guardrails`, `no_finetune` configurations
   - **Historical reference** for the `full_hybrid` configuration (GPU-dependent)

This mixed approach enables comparison while maintaining reproducibility and scientific integrity.

---

## Evaluation Constraints

### Episode Count
- **Default**: 5 episodes per configuration (configurable via `--episodes` CLI argument)
- **Rationale**: Balances statistical stability with runtime efficiency
- **Implication**: Results reflect aggregate performance across 5 random seed variants; larger episode counts would reduce stochastic variance

### Random Seed Strategy
Seeds are deterministically derived from evaluation type to ensure reproducibility:

```
Benchmark Evaluation: seed = 1000 + episode_index
Robustness Testing:   seed = 2000 + episode_index
Ablation Studies:     seed = 3000 + episode_index
```

This ensures:
- Consistent behavior across local runs
- Reproducible stochastic simulation states
- Isolation of variability to agent decision-making (not environment randomness)

### Environment Simulator
- **Task Difficulty**: Hard mode (80 fuel units, 5-step horizon)
- **Sector Dynamics**: Stochastic demand patterns generated per episode via environment-side seeding
- **Bottleneck Threshold**: Transport demand > 5 units triggers bottleneck detection
- **Reward Structure**: Cumulative sector satisfaction per step (normalized by 5.0 for per-episode score)

---

## Model & Agent Configurations

### API-Based Planner
- **Model**: `llama-3.1-8b-instant` (hosted on Groq Cloud)
- **API Rate Limits**: 1-second delay between API calls to respect service constraints
- **Determinism**: Temperature set to 0.0 for reproducible outputs (implementation detail in agent code)

### Fine-Tuned Models
- **Base Model**: `unsloth/llama-3-8b-Instruct-bnb-4bit` (4-bit quantized)
- **Adapter**: PEFT LoRA weights at `outputs/llama3_crisis_lora/`
- **Availability**: Required for `finetuned` and `hybrid` agent benchmarks
- **Fallback**: Historical metrics used if GPU/checkpoint unavailable

### Agent Configurations
All variations share common underlying architecture but differ in feature toggles:

| Config | use_memory | use_experts | use_history | use_guardrails | Source |
|--------|-----------|------------|------------|---------------|--------|
| baseline | False | False | False | False | Live Evaluation |
| memory | True | False | True | False | Live Evaluation |
| guarded | True | True | True | True | Live Evaluation |
| finetuned | N/A | N/A | N/A | N/A | Historical Reference |
| hybrid | N/A | N/A | N/A | N/A | Historical Reference |

---

## No Model Retraining

**Critical Assumption**: All benchmark results use fixed, pre-trained model weights from Phase 3. This evaluation framework does **not** perform any model retraining, fine-tuning, or weight updates.

Implications:
- Results reflect a snapshot of agent performance at a specific point in model development
- Recent improvements to underlying LLMs (e.g., Llama 3.1 updates) are not reflected
- LoRA adapter weights remain unchanged across all evaluation runs

---

## Robustness Testing

### Fuel Constraint Variations
Four fuel limit scenarios are evaluated: `[40, 60, 80, 100]` units

- **40 units**: Severely constrained (50% below standard hard mode)
- **60 units**: Moderately constrained (75% of standard)
- **80 units**: Standard hard mode
- **100 units**: Relaxed constraints (125% above standard)

Each constraint uses the **guarded** agent configuration (all components enabled).

### Statistical Measures
For each fuel constraint, the following metrics are computed across N episodes:

- `avg_score`: Mean of normalized cumulative rewards
- `score_std`: Standard deviation of avg_scores across episodes
- `score_ci95`: 95% confidence interval margin (1.96 × score_std / √N)
- `max_score`: Maximum observed episode reward
- `waste`: Total fuel over-allocation across all episodes and steps
- `bottleneck_rate`: Percentage of bottleneck scenarios successfully cleared

---

## Ablation Study Methodology

### Component Isolation
Five configurations are evaluated:

1. **FULL_HYBRID (Historical)**: Reference baseline using Phase 3 benchmark
2. **BASELINE**: No components (raw LLM, no memory, no guardrails)
3. **NO_MEMORY**: Remove ChromaDB memory context retrieval
4. **NO_GUARDRAILS**: Remove deterministic safety constraints
5. **NO_FINETUNE**: Use Groq API planner only (equivalent to guarded configuration)

### Interpretation
Ablation results show **correlations** between component removal and performance change. These correlations do not imply causation in a strict scientific sense but indicate component importance within the current architecture.

---

## Limitations & Generalizability

### Scope Limitations
1. **Environment Specificity**: Results are specific to the fuel-constrained logistics simulator. Generalization to other domains is not supported.
2. **Fixed Horizon**: All evaluations use a 5-step decision horizon. Performance under extended horizons is unknown.
3. **Deterministic Components**: Guardrails implement deterministic rules that may not be optimal in all scenarios.

### Statistical Limitations
1. **Small Episode Count**: Default N=5 episodes provides limited statistical power. Confidence intervals are relatively wide.
2. **No Baseline Variance**: Historical metrics represent single point estimates, not distributions. Uncertainty bounds for historical data are unavailable.
3. **Seed Determinism**: While reproducible, fixed seed strategies reduce observable variance and may hide algorithm sensitivity.

### Model Limitations
1. **Static Weights**: Models are not adapted to specific domains or updated mid-evaluation.
2. **Context Window**: LLM context length constrains planning horizon (implementation detail).
3. **Quantization**: Fine-tuned models use 4-bit quantization, which introduces computational bias relative to full-precision versions.

---

## Reproducibility Assumptions

### Required Environment
- **Python**: 3.12.x
- **GPU** (optional): NVIDIA T4/V100/A100 for fine-tuned model execution; CPU-only supports API-based agents
- **API Access**: Active Groq API key with quota for llama-3.1-8b-instant inference

### Stability Guarantees
- API-based agents: Fully reproducible across platforms (same seed → same output)
- Fine-tuned agents: Reproducible only on identical GPU hardware (CUDA/cuDNN version sensitivity)
- Simulator: Fully reproducible via seed control

### Known Variability Sources
- ChromaDB memory retrieval: Order may vary if index is rebuilt (mitigated by fixed seed strategy)
- Floating-point arithmetic: Minor numerical differences across architectures (< 1e-6 relative error)
- API response time: Transient network delays affect runtime measurements but not decision outputs

---

## Future Extension Points

This framework is designed for future expansion:

1. **Additional Constraints**: New fuel limits or time horizons can be added to `robustness.py`
2. **New Agent Variants**: Additional ablation configurations via `LLMPlanner` parameter combinations
3. **Alternative Models**: Framework supports swapping `llama-3.1-8b` for other Groq-hosted models
4. **Synthetic Benchmarks**: `benchmark.py` can be extended to synthetic task families beyond Phase 3 hard mode

---

## References

- `evaluation/config.py`: Loads historical metrics and configures defaults
- `evaluation/historical_results.json`: Source of truth for Phase 3 benchmark values
- `evaluation/reproducibility.md`: Step-by-step execution instructions
- `evaluation/failure_analysis.md`: Technical failure modes and mitigation strategies
