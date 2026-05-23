# Reproducibility Guide

This guide documents the technical environment, hardware configurations, model parameters, and execution instructions necessary to fully reproduce the benchmarks, robustness, and ablation studies of the Crisis Logistics agent.

---

## 💻 Environment Configurations

### 1. Hardware Specifications
- **Local Execution (CPU/API Evaluation)**: 
  - CPU: 8-core x86_64 Processor (e.g., Intel Core i7 / AMD Ryzen 7)
  - Memory: 16 GB RAM or higher
  - Connection: Active broadband internet connection (required for Groq LLM API and ChromaDB persistent layer metadata verification)
- **Model Fine-Tuning & Local Execution (GPU Evaluation)**:
  - GPU: NVIDIA T4 / V100 / A100 GPU (16GB VRAM minimum)
  - System: Google Colab High-RAM instance (25GB RAM)

### 2. Software & Tooling
- **Python Version**: `3.12.x`
- **Operating System**: Windows / Linux / macOS (tested on Windows 11)

### 3. Core Dependencies (Versions)
Below are the exact packages used during the research phase:
- `openenv-core==0.2.3` — Environment wrapper
- `chromadb==1.5.8` — Memory retrieval backend
- `groq==1.2.0` — LLM API execution client
- `openai==2.32.0` — Chat interface backend compatibility
- `pandas==3.0.2` — Data transformation and CSV handling
- `matplotlib==3.10.8` — Chart plotting
- `numpy==2.4.4` — Vector calculations & statistical helpers
- `python-dotenv==1.2.2` — Environment variables loader
- `uvicorn==0.44.0` — Fast API server runner
- `gradio==6.12.0` — Human-in-the-loop dashboard interface

---

## 🔑 Model & API Variables

### 1. Model Definitions
- **API Planner**: `llama-3.1-8b-instant` (hosted on Groq Cloud)
- **Local Fine-Tuned Model**:
  - Base: `unsloth/llama-3-8b-Instruct-bnb-4bit` (4-bit quantized Instruct model)
  - Adapter: PEFT LoRA adapter checkpointed at `outputs/llama3_crisis_lora/`

### 2. Random Seeds
To ensure structural generation stability inside the `GlobalCrisisEnv` simulator (specifically for sector demand noise and state transitions), fixed random seeds are applied across runs:
- **Benchmark Evaluation**: `1000 + ep` (where `ep` is the 0-indexed episode number)
- **Robustness Testing**: `2000 + ep`
- **Ablation Studies**: `3000 + ep`

---

## 🛠️ Step-by-Step Execution Instructions

To reproduce the complete evaluation suite, follow the instructions below:

### Step 1: Environment Setup
Initialize a Python 3.12 environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install groq
```

### Step 2: Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your-groq-api-key-here
LLM_MODEL=llama-3.1-8b-instant
SIMULATOR_URL=http://127.0.0.1:7860
```

### Step 3: Run the Logistics Simulator Server
Start the simulation backend. This must remain running in the background for all subsequent evaluations:
```bash
python -m server.app
```
*Note: Verify that the server is listening at http://127.0.0.1:7860.*

### Step 4: Run the Evaluation Suite
You can execute all evaluation scripts. Use the `--episodes` CLI argument to define the scale of the runs (recommended minimum = 5 episodes for baseline evaluation):
```bash
# 1. Run the system benchmarking
python evaluation/benchmark.py --episodes 5

# 2. Run robustness tests across fuel limits
python evaluation/robustness.py --episodes 5

# 3. Run the ablation studies
python evaluation/ablation.py --episodes 5
```

**Note on Episode Count**: The default of 5 episodes balances statistical stability with runtime efficiency. Larger episode counts improve confidence intervals and reduce stochastic variance, but increase total evaluation time. See evaluation/config.py::DEFAULT_EPISODES for current settings.

### Step 5: Compile Report & Charts
Generate the visualization plots and compile the final summary report:
```bash
python evaluation/report.py
```
This command compiles and creates:
- `evaluation/benchmark_results.csv`
- `evaluation/robustness_results.csv`
- `evaluation/ablation_results.csv`
- `evaluation/evaluation_report.md` (containing complete tables and performance analysis)
- `experiments/experiment_summary.md` (metadata catalog for audit tracking)
- Performance charts saved in `evaluation/`
