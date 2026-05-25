#!/usr/bin/env python3
"""
Adaptive Crisis AI — Decision Intelligence Demo
HuggingFace Spaces MVP Interface

This Gradio app provides:
- Overview of project architecture and results
- Benchmark dashboard with performance comparisons
- Agent explorer for historical metrics analysis
"""

import json
import os
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================================
# CONFIGURATION & DATA LOADING
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
HISTORICAL_RESULTS_PATH = EVALUATION_DIR / "historical_results.json"
BENCHMARK_CSV_PATH = EVALUATION_DIR / "benchmark_results.csv"
ABLATION_CSV_PATH = EVALUATION_DIR / "ablation_results.csv"


def load_historical_results():
    """Load historical results from JSON, with error handling."""
    try:
        if HISTORICAL_RESULTS_PATH.exists():
            with open(HISTORICAL_RESULTS_PATH, "r") as f:
                data = json.load(f)
            return data.get("results", {})
        return {}
    except Exception as e:
        print(f"Warning: Failed to load historical results: {e}")
        return {}


def load_benchmark_csv():
    """Load benchmark CSV, with error handling."""
    try:
        if BENCHMARK_CSV_PATH.exists():
            return pd.read_csv(BENCHMARK_CSV_PATH)
        return None
    except Exception as e:
        print(f"Warning: Failed to load benchmark CSV: {e}")
        return None


def load_ablation_csv():
    """Load ablation CSV, with error handling."""
    try:
        if ABLATION_CSV_PATH.exists():
            return pd.read_csv(ABLATION_CSV_PATH)
        return None
    except Exception as e:
        print(f"Warning: Failed to load ablation CSV: {e}")
        return None


# Load data at startup
HISTORICAL_DATA = load_historical_results()
BENCHMARK_DF = load_benchmark_csv()
ABLATION_DF = load_ablation_csv()


# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

def create_overview_tab():
    """Create the Overview tab with project summary."""
    
    overview_md = """
# 🎯 Adaptive Crisis AI — Strategic Resource Allocation

## Project Summary
**Adaptive Crisis AI** is an intelligent decision support system designed for crisis management and resource allocation under uncertainty. 
The system combines memory-augmented planning, guardrails enforcement, and fine-tuned decision-making to optimize crisis response strategies.

### Key Innovation
- **Hybrid Intelligence Architecture**: Combines retrieval-augmented memory, guardrails enforcement, and fine-tuned optimization
- **Zero-Waste Guardrails**: Ensures 0% resource waste with 100% bottleneck identification
- **Adaptive Planning**: Learns from historical crisis responses to improve decisions

---

## 📊 System Architecture

### Core Components
1. **Agent Module** (`agent/`)
   - `planner.py` — Decision planning engine
   - `planner_llm.py` — LLM-powered planning
   - `expert_provider.py` — Expert trajectory provider
   - `memory.py` — Memory management system
   - `reflection.py` — Self-reflection mechanism

2. **Evaluation Framework** (`evaluation/`)
   - Comprehensive benchmarking suite
   - Robustness testing
   - Ablation studies

3. **Runner Module** (`runner/`)
   - Training orchestration
   - Model fine-tuning
   - Evaluation scripts

### Agent Variants
- **Baseline**: Simple heuristic decision-making (Score: 0.03)
- **Memory**: Adds retrieval-augmented memory (Score: 0.08)
- **Guarded**: Enforces zero-waste guardrails (Score: 0.115)
- **Fine-tuned**: LLM fine-tuned on expert trajectories (Score: 0.088)
- **Hybrid**: Combines all components (Score: **0.1315** ⭐)

---

## 📈 Evaluation Summary

### Performance Metrics
| Agent | Avg Score | Max Score | Waste | Bottleneck Rate | Runtime (s) |
|-------|-----------|-----------|-------|-----------------|-------------|
| Baseline | 0.031 | 0.061 | 5% | 100% | 6.4 |
| Memory | 0.073 | 0.073 | 10% | 100% | 11.3 |
| Guarded | 0.123 | 0.123 | **0%** | 100% | 12.1 |
| Fine-tuned | 0.088 | 0.110 | 35% | 45% | 5.2 |
| **Hybrid** | **0.132** | **0.142** | **0%** | 100% | 5.8 |

### Key Findings
✅ **Hybrid achieves 4.3x improvement** over baseline (0.132 vs 0.031)
✅ **Zero-waste guarantee** maintained while improving performance
✅ **Efficient runtime** (5.8s vs 12.1s for guarded alone)
✅ **100% bottleneck identification** across all decision points

---

## 📚 Source Code
All code is available in the repository:
- **Agent Logic**: `agent/` directory
- **Evaluation Scripts**: `evaluation/` directory  
- **Training Pipeline**: `runner/` directory
- **Server**: `server/` directory (FastAPI backend)

Use the **Agent Explorer** tab to see detailed metrics for each agent variant.
"""
    
    return gr.Markdown(overview_md)


# ============================================================================
# TAB 2: BENCHMARK DASHBOARD
# ============================================================================

def create_benchmark_tab():
    """Create the Benchmark Dashboard tab."""
    
    with gr.Group():
        gr.Markdown("## 📊 Benchmark Dashboard")
        
        if BENCHMARK_DF is not None:
            # Benchmark table
            with gr.Row():
                gr.Dataframe(
                    value=BENCHMARK_DF,
                    label="Benchmark Results",
                    interactive=False,
                    wrap=True
                )
            
            # Benchmark comparison charts
            with gr.Row():
                # Score comparison
                score_chart = go.Figure(data=[
                    go.Bar(
                        x=BENCHMARK_DF["agent"],
                        y=BENCHMARK_DF["avg_score"],
                        name="Avg Score",
                        marker_color="lightblue"
                    ),
                    go.Bar(
                        x=BENCHMARK_DF["agent"],
                        y=BENCHMARK_DF["max_score"],
                        name="Max Score",
                        marker_color="darkblue"
                    )
                ])
                score_chart.update_layout(
                    title="Agent Performance Comparison",
                    xaxis_title="Agent",
                    yaxis_title="Score",
                    barmode="group",
                    height=400
                )
                gr.Plot(value=score_chart)
            
            # Efficiency metrics
            with gr.Row():
                efficiency_chart = go.Figure(data=[
                    go.Bar(
                        x=BENCHMARK_DF["agent"],
                        y=BENCHMARK_DF["waste"],
                        name="Waste %",
                        marker_color="coral"
                    ),
                    go.Bar(
                        x=BENCHMARK_DF["agent"],
                        y=BENCHMARK_DF["bottleneck_rate"],
                        name="Bottleneck Rate %",
                        marker_color="lightgreen"
                    )
                ])
                efficiency_chart.update_layout(
                    title="Resource Efficiency Metrics",
                    xaxis_title="Agent",
                    yaxis_title="Percentage (%)",
                    barmode="group",
                    height=400
                )
                gr.Plot(value=efficiency_chart)
            
            # Runtime comparison
            with gr.Row():
                runtime_chart = go.Figure(data=[
                    go.Bar(
                        x=BENCHMARK_DF["agent"],
                        y=BENCHMARK_DF["runtime"],
                        marker_color="mediumpurple"
                    )
                ])
                runtime_chart.update_layout(
                    title="Runtime Comparison",
                    xaxis_title="Agent",
                    yaxis_title="Runtime (seconds)",
                    height=400
                )
                gr.Plot(value=runtime_chart)
        else:
            gr.Markdown("⚠️ Benchmark data not available")
        
        # Ablation results
        if ABLATION_DF is not None:
            gr.Markdown("---")
            gr.Markdown("### Ablation Study Results")
            gr.Dataframe(
                value=ABLATION_DF,
                label="Ablation Results",
                interactive=False,
                wrap=True
            )
        else:
            gr.Markdown("⚠️ Ablation data not available")


# ============================================================================
# TAB 3: AGENT EXPLORER
# ============================================================================

def get_agent_metrics(agent_name):
    """Get metrics for a specific agent from historical results."""
    if agent_name in HISTORICAL_DATA:
        metrics = HISTORICAL_DATA[agent_name]
        metrics_text = f"""
### {agent_name.title()} Agent

#### Performance Metrics
- **Average Score**: {metrics.get('avg_score', 'N/A')}
- **Max Score**: {metrics.get('max_score', 'N/A')}
- **Resource Waste**: {metrics.get('waste', 'N/A')}%
- **Bottleneck Rate**: {metrics.get('bottleneck_rate', 'N/A')}%
- **Runtime**: {metrics.get('runtime', 'N/A')} seconds

#### Description
"""
        
        descriptions = {
            "baseline": "Simple heuristic-based decision making without any advanced features.",
            "memory": "Adds retrieval-augmented memory for better decision context.",
            "guarded": "Enforces zero-waste guardrails on all decisions.",
            "finetuned": "Uses LLM fine-tuned on expert crisis response trajectories.",
            "hybrid": "Combines memory, guardrails, and fine-tuning for optimal performance.",
        }
        
        metrics_text += descriptions.get(agent_name, "No description available.")
        return metrics_text
    return f"No data available for {agent_name}"


def create_agent_explorer_tab():
    """Create the Agent Explorer tab."""
    
    agents = list(HISTORICAL_DATA.keys()) if HISTORICAL_DATA else [
        "baseline", "memory", "guarded", "finetuned", "hybrid"
    ]
    
    with gr.Group():
        gr.Markdown("## 🤖 Agent Explorer")
        
        with gr.Row():
            agent_dropdown = gr.Dropdown(
                choices=agents,
                value=agents[0] if agents else None,
                label="Select Agent",
                interactive=True
            )
        
        with gr.Row():
            metrics_display = gr.Markdown(
                get_agent_metrics(agents[0] if agents else None)
            )
        
        # Connect dropdown to metrics display
        agent_dropdown.change(
            fn=get_agent_metrics,
            inputs=agent_dropdown,
            outputs=metrics_display
        )
        
        # Comparison visualization
        if HISTORICAL_DATA:
            gr.Markdown("---")
            gr.Markdown("### All Agents Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for agent_name, metrics in HISTORICAL_DATA.items():
                comparison_data.append({
                    "Agent": agent_name.title(),
                    "Avg Score": metrics.get("avg_score", 0),
                    "Max Score": metrics.get("max_score", 0),
                    "Waste %": metrics.get("waste", 0),
                    "Bottleneck %": metrics.get("bottleneck_rate", 0),
                    "Runtime (s)": metrics.get("runtime", 0),
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            gr.Dataframe(value=comparison_df, interactive=False, wrap=True)
            
            # Comparison chart
            with gr.Row():
                comparison_chart = go.Figure(data=[
                    go.Bar(
                        x=comparison_df["Agent"],
                        y=comparison_df["Avg Score"],
                        name="Avg Score",
                        marker_color="skyblue"
                    )
                ])
                comparison_chart.update_layout(
                    title="Agent Performance Comparison",
                    xaxis_title="Agent",
                    yaxis_title="Average Score",
                    height=400
                )
                gr.Plot(value=comparison_chart)


# ============================================================================
# BUILD GRADIO APP
# ============================================================================

def create_app():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="Adaptive Crisis AI Demo",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # 🛡️ Adaptive Crisis AI — Decision Intelligence Demo
            
            A strategic resource allocation system for crisis management using hybrid intelligence.
            Explore the project overview, benchmark results, and individual agent metrics.
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("📖 Overview"):
                create_overview_tab()
            
            with gr.TabItem("📊 Benchmark Dashboard"):
                create_benchmark_tab()
            
            with gr.TabItem("🤖 Agent Explorer"):
                create_agent_explorer_tab()
        
        gr.Markdown(
            """
            ---
            **Learn More**: [GitHub Repository](https://github.com/LVVignesh/Adaptive-AI-decision-intelligence-system) | 
            **Paper**: Phase 3 Re-Thesis Results | 
            **License**: MIT
            """
        )
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
