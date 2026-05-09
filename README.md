# 🛡️ Adaptive Crisis AI: Strategic Resource Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![ChromaDB](https://img.shields.io/badge/Memory-ChromaDB-green.svg)](https://www.trychroma.com/)

An advanced AI decision-intelligence system designed to master complex logistics constraints in high-stakes crisis environments. This project demonstrates a complete **Self-Improving Intelligence Pipeline**: from rule-based bootstrapping to synthetic data generation and LLM fine-tuning.

---

## 🚀 The Architecture

The system is built in three distinct phases of evolution:

### Phase 1: The Memory Foundation ✅
- **Agent:** Heuristic Planner with persistent memory.
- **Engine:** Integrated **ChromaDB** to store state-action reflections.
- **Outcome:** A baseline system that "remembers" past mistakes and successes via vector similarity search.

### Phase 2: Guardrails & Expert Data Collection ✅
- **Phase 2A:** **Expert Data Generation**.
    - Developed a precision rule-based expert planner managing a 5-step horizon.
    - Captured **180 high-quality trajectories** (900+ transition steps) formatted in **Llama-3 Instruction JSONL**.
- **Phase 2B:** **LLM Integration & Constraints (The Guarded Agent)**.
    - Swapped heuristic logic for a Groq-powered **Llama-3.1-8b** decision agent.
    - Built a **Deterministic Constraint Layer** guaranteeing 0% fuel waste and enforcing strict pacing (max 60% fuel per step).
    - **Benchmark Results:** Achieved **0 Waste** and a **100% Bottleneck Clear Rate**, proving the system is exceptionally safe and stable.

### Phase 3: Knowledge Distillation (Fine-Tuning) ✅
- **Objective:** Distill the Phase 2 agent's logic into model weights to remove reliance on hard-coded guards.
- **Pipeline:** Built a complete **MLOps Training Pipeline**.
    - Processed 900+ examples into ChatML format.
    - Fine-Tuned **Llama-3-8B-Instruct** using **Unsloth** and **LoRA** (Rank 16, 4-bit Quantization) on a cloud GPU (Google Colab T4).
    - Built a local Evaluation pipeline to benchmark the fine-tuned model against the guarded baseline.
- **Outcome:** Successfully created a specialized logistics adapter ready for cloud deployment.

---

## 📈 Final Benchmark Results: The Hybrid Agent (Round 2 Re-Thesis)

The final architecture combines the **Phase 3 Fine-Tuned Brain** with the **Phase 2B Deterministic Guardrails**. This represents the ultimate evolution of the system: Neural reasoning stabilized by symbolic logic.

| Metric | Hybrid Agent (Fine-Tuned + Guards) | Status |
| :--- | :--- | :--- |
| **Average Strategy Score** | **0.1315** | ✅ **Optimal** |
| **Total Fuel Waste** | **0.0%** | 🛡️ **Guaranteed** |
| **Bottleneck Clear Rate** | **100.0%** | 🚀 **Resilient** |
| **JSON Stability** | **100%** | 💎 **Robust** |
| **Guardrail Interventions** | **3** | 🧠 **Stabilized** |

> **Key Insight:** By wrapping the fine-tuned model in a deterministic constraint layer, we achieve a **20x performance gain** over the base model while maintaining 100% safety. This Hybrid Agent is the definitive version presented for the Round 2 Re-Thesis.

### 🖼️ Evidence: Hybrid Agent Evaluation (Google Colab)

#### Round 2: Hybrid Agent (Fine-Tuned + Guardrails)
![Hybrid Results Part 1](assests/hybrid_fine%20tuned_colab_output_3.png)
![Hybrid Results Part 2](assests/hybrid_fine_tuned_colab_output_4.png)

#### Previous Phase: Pure Fine-Tuned Agent
![Phase 3 Results Part 1](assests/fine_tuned_colab_results_1.png)
![Phase 3 Results Part 2](assests/fine_tuned_colab_results_2.png)

---

## 🧠 Core Engineering Insight: Hybrid Intelligence

This project demonstrates a critical principle in production AI:
*   **Neural Reasoning (LLM)**: Excellent for flexible decision-making and pattern recognition.
*   **Deterministic Constraints (Guards)**: Essential for safety, pacing, and 100% adherence to business/logic rules.

**The Result:** A hybrid system combining these two layers achieves a performance level and safety profile that neither approach could reach alone. This mirrors the architecture of real-world autonomous systems in safety-critical domains like logistics and energy.

---

## 🛠️ Technology Stack
- **Environment:** Custom `GlobalCrisisEnv` Simulator (FastAPI / Uvicorn).
- **Agentic Logic:** LangGraph / Groq API.
- **Vector Database:** ChromaDB (State-Reflection persistent memory).
- **Fine-Tuning:** Unsloth, HuggingFace Transformers, PEFT (LoRA), BitsAndBytes.

## 🏁 How to Run the Pipeline

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Phase 2B Agent (Production Guarded Agent):**
   Requires `GROQ_API_KEY` in `.env`:
   ```bash
   python runner/train_llm.py
   ```
3. **Format Data for Fine-Tuning:**
   ```bash
   python runner/format_dataset.py
   ```
4. **Fine-Tune in the Cloud (Google Colab):**
   Upload `finetuning_dataset.jsonl` and run the script on a T4 GPU:
   ```bash
   python runner/fine_tune.py
   ```
5. **Evaluate Models:**
   *Note: Requires a dedicated NVIDIA GPU to run the 4-bit Llama-3 model locally.*
   ```bash
   python runner/evaluate_finetuned.py
   ```

---

*This project is a flagship demonstration of full-lifecycle Adaptive AI engineering for decision-intelligence roles.*
