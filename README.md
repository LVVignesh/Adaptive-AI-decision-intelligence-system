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

### Phase 2: Strategic Intelligence (Current) ✅
- **Phase 2A (Complete):** **Expert Data Generation**.
    - Developed a high-precision, rule-based expert planner that manages a 5-step horizon.
    - Captures **180 high-quality trajectories** balanced across Easy, Medium, and Hard difficulties.
    - **Dataset:** 900+ transition steps formatted in **Llama-3 Instruction JSONL**.
- **Phase 2B (Complete):** **LLM Integration & Memory**.
    - Swapped heuristic logic for a Groq-powered **Llama-3.1-8b** decision agent.
    - Implemented **Few-Shot Learning** using top trajectories from the Phase 2A dataset.
    - Integrated **ChromaDB** for step-by-step agent reflections and learning.
- **Goal:** Fine-tuning an open-source Llama-3 model using **LoRA**.
- **Outcome:** A standalone, high-performance strategic model that identifies logistics bottlenecks and optimizes fuel distribution with expert-level precision.

---

## 📊 Dataset Insights (Phase 2A Output)

We successfully harvested 180 expert-grade episodes with the following constraints:
- **Zero-Waste Policy:** Rejects any episode with >10% fuel waste.
- **Priority Logic:** Explicitly identifies and resolves the **Transport Bottleneck** (40% global efficiency penalty).
- **Reasoning:** Every action is paired with a strategic "thought process" (e.g., *"Stabilizing critical hospital demand before residential supply"*).

| Task Difficulty | Target Episodes | Expert Tier (≥0.18) | Near-Expert (≥0.12) |
| :--- | :--- | :--- | :--- |
| **Easy** | 60 | 0 | 60 |
| **Medium** | 60 | 10 | 50 |
| **Hard** | 60 | 0 | 60 |

---

## 🛠️ Technology Stack
- **Environment:** Custom `GlobalCrisisEnv` simulator (FastAPI/Uvicorn).
- **Vector Database:** ChromaDB (State-Reflection persistent storage).
- **Generator Agent:** Python-based Heuristic Expert.
- **Future Fine-tuning:** Llama-3 (HF Transformers / PEFT / LoRA).

## 🏁 Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Memory:**
   Ensure ChromaDB is initialized by running Phase 1:
   ```bash
   python runner/train.py
   ```
3. **Generate Expert Data:**
   ```bash
   python runner/generate_expert_data.py
   ```
4. **Run Phase 2B Agent (LLM + Memory):**
   Requires `GROQ_API_KEY` in `.env`:
   ```bash
   python runner/train_llm.py
   ```

---

*This project is a flagship demonstration of Adaptive AI engineering for decision-intelligence roles.*
