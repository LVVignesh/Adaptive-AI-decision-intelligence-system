# Evaluation Framework Changelog

All notable changes to the evaluation framework are documented in this file.

---

## [1.1.0] - 2026-05-22 - Research Quality Cleanup

### Added

- **Evaluation Traceability**: Added `source` column to all evaluation CSVs:
  - `benchmark_results.csv`: `source` field (live_evaluation | historical_reference)
  - `ablation_results.csv`: `source` field (live_evaluation | historical_reference)
  - Enables identification of which metrics come from Phase 3 historical data vs. current CPU evaluation

- **Statistical Improvements**: Enhanced robustness testing with distribution metrics:
  - `score_std`: Standard deviation of scores across episodes per fuel constraint
  - `score_ci95`: 95% confidence interval margin (1.96 × std / √N)
  - Provides uncertainty quantification for robustness results

- **Documentation Files**:
  - `evaluation/assumptions.md`: Comprehensive documentation of methodology, assumptions, limitations, and reproducibility constraints
  - `evaluation/historical_results.json`: JSON-formatted historical metrics with metadata (version, generated_at, source)

- **Evaluation Methodology Section**: New README section (`## Evaluation Methodology`) with subsections:
  - Benchmark Evaluation (description + run command)
  - Robustness Testing (description + run command)
  - Ablation Studies (description + run command)
  - Results & Reporting (description + run command)
  - Reproducibility (reference to detailed guide)

### Changed

- **Data Source Migration**: 
  - Removed hardcoded `HISTORICAL_METRICS` dictionary from `evaluation/config.py`
  - Created `evaluation/historical_results.json` as single source of truth
  - Added `load_historical_metrics()` utility function for dynamic loading
  - All evaluation scripts updated to use new loading mechanism

- **Ablation Configuration Labels**:
  - (Internally) `full_hybrid` → displayed as `FULL HYBRID (Historical)` in reports
  - Clarifies that this benchmark uses Phase 3 historical data
  - (Internally) `no_finetune` → displayed as `GUARDED REFERENCE` in reports
  - Better reflects that this is the proper live-evaluation reference configuration

- **Scientific Language Improvements**: Softened causal claims in `evaluation/report.py`:
  - "Guardrails are the Primary Driver" → "Guardrails show the largest observed performance contribution under current evaluation setup"
  - "causes" → "correlates with"
  - "proves" → "suggests"
  - "lifts the system score" → "is associated with performance improvement"
  - "helps" → "correlates with"
  - Changes apply to generated markdown in `evaluation_report.md`

- **Robustness CSV Structure**: Reordered columns for clarity:
  - Before: `fuel_limit, avg_score, max_score, waste, bottleneck_rate, runtime`
  - After: `fuel_limit, avg_score, score_std, score_ci95, max_score, waste, bottleneck_rate, runtime`

- **Documentation Improvements**:
  - `evaluation/reproducibility.md`: 
    - Replaced "minimum 5 for benchmark truthfulness" with "recommended minimum = 5 episodes"
    - Added note on episode count tradeoffs (larger counts improve confidence, increase runtime)
  
  - `evaluation/failure_analysis.md`:
    - Replaced specific "Episode X, Step Y" references with "Representative Failure Scenario"
    - Removes false specificity about which exact episode/step in reproducible runs
    - Preserved all technical failure descriptions and mitigation explanations

- **Report Template**: Updated `evaluation_report.md` structure:
  - Before: Introduction → Benchmarks → Robustness → Ablation → Conclusion
  - After: Introduction → Benchmarks → Robustness → Ablation → Limitations → Conclusion
  - New "Limitations" section documents:
    - Historical reference usage in FULL HYBRID benchmark
    - Environment-specific results (not universal claims)
    - Limited episode count with recommendation for larger studies
    - Constraint scope boundaries (fuel limits [40, 60, 80, 100])
    - Architecture-specific component importance

### Fixed

- **Data Integrity**: Ensured all benchmark values remain unchanged during migration:
  - Historical values: Exactly preserved from Phase 3 (no rounding, no modifications)
  - Live evaluation: No changes to simulator logic or agent configuration
  - All CSV output values identical to pre-1.1.0 format

- **Reproducibility**: Improved traceability of evaluation sources:
  - Source tracking in CSVs enables audit trail for performance claims
  - `assumptions.md` documents all methodology constraints and limitations

### No Changes

✅ **Preserved**:
- Benchmark values: All numerical scores unchanged
- Agent architecture: No modifications to decision-making logic
- Simulator: No changes to environment rules or dynamics
- Model weights: All models use Phase 3 frozen weights (no retraining)
- Evaluation methodology: Same metrics computed, same reward structure
- Hardware assumptions: CPU/GPU configurations identical

### Migration Notes

- **For Users**: Backward compatibility maintained via `HISTORICAL_METRICS()` function wrapper in `config.py`
- **For Scripts**: No code changes required if using through standard entry points (`benchmark.py`, `robustness.py`, `ablation.py`)
- **For Developers**: New `load_historical_metrics()` function recommended for direct historical data access

### Verification

Run the verification suite to confirm no values changed:

```bash
# 1. Verify config loads correctly
python -c "from evaluation.config import load_historical_metrics; print(load_historical_metrics())"

# 2. Run all evaluation scripts (5 episodes each)
python evaluation/benchmark.py --episodes 5
python evaluation/robustness.py --episodes 5
python evaluation/ablation.py --episodes 5

# 3. Verify source column in CSVs
python -c "import pandas as pd; print(pd.read_csv('evaluation/benchmark_results.csv')[['agent', 'source']])"
python -c "import pandas as pd; print(pd.read_csv('evaluation/ablation_results.csv')[['config', 'source']])"

# 4. Verify score_std in robustness CSV
python -c "import pandas as pd; print(pd.read_csv('evaluation/robustness_results.csv')[['fuel_limit', 'avg_score', 'score_std', 'score_ci95']])"

# 5. Generate final report
python evaluation/report.py

# 6. Check report contains Limitations section
grep -A 5 "## ⚠️ Limitations" evaluation/evaluation_report.md
```

---

## [1.0.0] - 2026-03-XX - Initial Evaluation Framework

- Initial release with core evaluation suite:
  - `benchmark.py`: Agent performance benchmarking
  - `robustness.py`: Fuel constraint sensitivity testing
  - `ablation.py`: Component contribution analysis
  - `report.py`: Automated chart generation and report compilation
  - `reproducibility.md`: Environment and execution documentation
  - `failure_analysis.md`: Failure mode catalog and mitigations
