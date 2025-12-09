# 2.5 Multi-Agent System for Exploratory Data Analysis

This folder contains a sophisticated multi-agent system designed to automate and enhance exploratory data analysis (EDA). The system orchestrates multiple specialized agents to perform comprehensive data analysis.

## Architecture

The system uses a 5-agent orchestration pattern:

1. **Planner Agent** (gemini-2.5-pro)
   - Analyzes the user's EDA question
   - Decomposes analysis into specific tasks
   - Creates structured analysis plan

2. **Data Inspector Agent** (gemini-2.5-flash)
   - Examines data structure and quality
   - Reports shape, types, missing values
   - Identifies data quality issues
   - Produces summary statistics

3. **Statistician Agent** (gemini-2.5-flash)
   - Performs correlation analysis
   - Conducts hypothesis testing
   - Computes confidence intervals
   - Identifies statistical significance

4. **Visualizer Agent** (gemini-2.5-flash)
   - Creates appropriate visualizations
   - Generates distribution, scatter, box plots
   - Produces correlation heatmaps
   - Saves publication-quality plots

5. **Synthesizer Agent** (gemini-2.5-pro)
   - Combines all findings
   - Creates executive summary
   - Identifies key insights
   - Recommends next steps

## Data Flow

```
User Question
    ↓
Planner → Analysis Plan
    ↓
┌─────────────────────────────────┐
│  Parallel Analysis              │
├─────────────────────────────────┤
│ Inspector → Data Characteristics │
│ Statistician → Statistical Tests │
│ Visualizer → Charts & Plots     │
└─────────────────────────────────┘
    ↓
Synthesizer → Comprehensive Report
```

## Running

### Interactive Gradio App

```bash
uv run --env-file .env gradio src/2_frameworks/5_eda_multi_agent/app.py
```

This launches an interactive chat interface where you can ask natural language EDA questions about the dataset.

**Example Questions:**
- "What are the main characteristics and distributions of this dataset?"
- "Are there any significant correlations between numeric variables?"
- "What are the missing data patterns and potential data quality issues?"
- "What patterns and anomalies exist in the data?"
- "Can you provide a comprehensive overview suitable for stakeholders?"

### CLI Version

```bash
uv run --env-file .env src/2_frameworks/5_eda_multi_agent/cli.py
```

This runs the analysis non-interactively and logs all results to stdout.

## Key Features

### 1. Intelligent Task Decomposition
- Planner breaks down complex EDA questions into specific, actionable tasks
- Ensures comprehensive coverage of relevant analysis dimensions

### 2. Code Interpretation
- All agents use the E2B code interpreter for programmatic analysis
- Enables sophisticated statistical tests and visualizations
- Data is not persisted across tool calls (agents handle re-uploads)

### 3. Structured Output
- Planner provides structured JSON output for orchestration
- Each agent produces clear, actionable findings
- Synthesizer creates readable narrative reports

### 4. Langfuse Tracing
- Complete trace of all agent calls and reasoning
- Tracks which agents were called and in what order
- Enables debugging and optimization
- Integrates with Langfuse for analysis and cost tracking

### 5. Multi-Model Strategy
- Uses expensive, capable models (gemini-2.5-pro) for planning and synthesis
- Uses faster, cheaper models (gemini-2.5-flash) for specialized analysis
- Optimizes cost while maintaining quality

## Example Output

```
📋 EDA Strategy Created

**Reasoning:** To comprehensively understand this dataset, I'll focus on:
1. Data structure and quality assessment
2. Distribution analysis for all variables
3. Correlation patterns between features
4. Identification of outliers and anomalies

**Planned Tasks:**
- Inspection: 4 tasks
- Statistical Analysis: 3 tasks
- Visualizations: 5 tasks
- Pattern Detection: 2 tasks

[Analysis proceeds with 5 sequential steps...]

✨ EDA Report

## Executive Summary
This dataset contains 1,000 observations across 15 variables. Overall data quality
is good with <5% missing values except in the 'notes' field (45% missing).

## Key Findings
1. Three main clusters identified in the price distribution
2. Strong correlation (r=0.87) between feature_x and feature_y
3. 12 outliers detected in the value column (>3σ)
...
```

## Configuration

The system uses:
- **Environment variables** from `.env` for API keys
- **Dataset path**: `tests/tool_tests/example_files/final_unified_dataset_sample.csv`
- **Code interpreter timeout**: 60 seconds
- **LLM models**: Gemini 2.5 series

## Cost Optimization

The system balances cost and quality:
- **Planner**: Pro model (better at complex reasoning)
- **Inspection/Analysis**: Flash models (fast, 70% cheaper)
- **Synthesizer**: Pro model (better at synthesis)
- **Typical cost**: ~$0.50-1.00 per comprehensive EDA

## Extending the System

To add new analysis capabilities:

1. Create new specialized agent with custom instructions
2. Add new method following `_run_*` pattern
3. Integrate into main orchestration loop
4. Update planner prompt to include new task type

Example:

```python
# In agents_shared.py

anomaly_detection_agent = agents.Agent(
    name="AnomalyDetectorAgent",
    instructions="Detect and classify anomalies in the dataset...",
    tools=[agents.function_tool(code_interpreter.run_code)],
    model=agents.OpenAIChatCompletionsModel(...)
)

# Then in app.py
async def _run_anomaly_detection(dataset_path: str, tasks: list[str]) -> str:
    # Implementation
    pass
```

## Troubleshooting

**Issue**: "Dataset not found"
- Ensure `tests/tool_tests/example_files/final_unified_dataset_sample.csv` exists
- Update `dataset_path` variable if using different dataset

**Issue**: "Code interpreter timeout"
- Complex analyses may exceed 60-second timeout
- Increase `timeout_seconds` in CodeInterpreter initialization

**Issue**: "Low-quality visualizations"
- The visualizer agent may need guidance on specific plot types
- Update agent instructions with more specific requirements

## Files

- `app.py` - Gradio interactive interface
- `cli.py` - Command-line interface
- `agents_shared.py` - Shared agent definitions and orchestration utilities
- `README.md` - This file

## References

- [OpenAI Agents SDK](https://github.com/openai/agents-sdk)
- [Multi-Agent Architecture (Efficient Pattern)](../2_multi_agent/efficient.py)
- [Code Interpreter Tool](../../utils/tools/code_interpreter.py)
- [Langfuse Integration](../../utils/langfuse/)
