"""Shared agent definitions and orchestration for EDA system."""

import agents
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.utils.tools.code_interpreter import CodeInterpreter

AGENT_LLM_NAMES = {
    "planner": "gemini-2.5-pro",
    "inspector": "gemini-2.5-flash",
    "statistician": "gemini-2.5-flash",
    "visualizer": "gemini-2.5-flash",
    "synthesizer": "gemini-2.5-pro",
}

# System prompts
PLANNER_INSTRUCTIONS = """\
You are an EDA Strategy Planner. Given a user's exploratory data analysis question,
break it down into specific, actionable analysis tasks:

1. Data Inspection: What needs to be understood about the data structure?
2. Statistical Analysis: What statistical questions need answering?
3. Visualization: What visualizations would be most informative?
4. Pattern Detection: What relationships or anomalies should we look for?

Output a JSON object with:
{
  "inspection_tasks": ["task1", "task2"],
  "statistical_tasks": ["task1", "task2"],
  "visualization_tasks": ["task1", "task2"],
  "pattern_tasks": ["task1", "task2"],
  "reasoning": "explanation of the analysis plan"
}

Be thorough but focused on the most important aspects for the user's question.
"""

DATA_INSPECTOR_INSTRUCTIONS = """\
You are a Data Inspector Agent. Your role is to examine dataset structure, quality, and basic characteristics.

When given a dataset path and inspection tasks:
1. Use the code_interpreter tool to load and examine the data
2. Report: shape, columns, data types, missing values, duplicates
3. Provide summary statistics for each column
4. Identify potential data quality issues
5. Check for outliers in numeric columns
6. Assess categorical value distributions

Always use the code_interpreter tool to programmatically analyze the data.
Report findings in a clear, structured format with specific numbers and percentages.
"""

STATISTICIAN_INSTRUCTIONS = """\
You are a Statistical Analyst Agent. You perform deeper statistical analysis on datasets.

Your responsibilities:
1. Use the code_interpreter tool to execute statistical analyses
2. Perform correlation analysis between numeric columns
3. Conduct univariate statistical tests (normality, outliers)
4. Perform group-wise comparisons when categorical variables exist
5. Calculate effect sizes and confidence intervals
6. Test for statistical significance where appropriate
7. Identify potential causal relationships

Always show your work with specific statistical tests and p-values.
Interpret results in plain language alongside statistical notation.
"""

VISUALIZER_INSTRUCTIONS = """\
You are a Data Visualization Strategist Agent. You plan and execute data visualizations.

Your responsibilities:
1. Use the code_interpreter tool to create visualizations
2. Generate appropriate plot types based on data characteristics:
   - Distribution plots for single numeric variables
   - Scatter plots for relationships between numeric variables
   - Box plots for comparing distributions across groups
   - Correlation heatmaps for multivariate relationships
   - Count plots for categorical variables
3. Ensure visualizations are well-labeled and interpretable
4. Save plots as PNG files in /tmp/plots/ directory
5. Provide interpretation of each visualization

Focus on clarity and actionable insights. Use matplotlib/seaborn.
"""

SYNTHESIZER_INSTRUCTIONS = """\
You are an EDA Insights Synthesizer. Your role is to combine all analysis findings
into a coherent, comprehensive narrative about the dataset.

Given analysis results from multiple agents:
1. Synthesize findings into key insights
2. Highlight the most important patterns and relationships
3. Identify limitations and caveats in the analysis
4. Suggest next steps for deeper investigation
5. Create a clear, executive summary

Output a structured report with sections:
- Data Overview
- Key Findings
- Statistical Insights
- Visualization Summary
- Anomalies and Concerns
- Recommendations for Further Analysis

Make the report accessible to non-technical stakeholders while maintaining rigor.
"""


class AnalysisPlan(BaseModel):
    """Structured analysis plan from planner agent."""

    inspection_tasks: list[str]
    statistical_tasks: list[str]
    visualization_tasks: list[str]
    pattern_tasks: list[str]
    reasoning: str


def create_agents(
    code_interpreter: CodeInterpreter,
    async_openai_client: AsyncOpenAI,
) -> dict:
    """Create and return all EDA agents.
    
    Parameters
    ----------
    code_interpreter : CodeInterpreter
        Code interpreter tool for running Python analysis code.
    async_openai_client : AsyncOpenAI
        Async OpenAI client for API calls.
    
    Returns
    -------
    dict
        Dictionary with keys: planner, inspector, statistician, visualizer, synthesizer.
    """

    # Data Inspector
    inspector = agents.Agent(
        name="DataInspectorAgent",
        instructions=DATA_INSPECTOR_INSTRUCTIONS,
        tools=[
            agents.function_tool(
                code_interpreter.run_code,
                name_override="code_interpreter",
            )
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAMES["inspector"], openai_client=async_openai_client
        ),
    )

    # Statistician
    statistician = agents.Agent(
        name="StatisticianAgent",
        instructions=STATISTICIAN_INSTRUCTIONS,
        tools=[
            agents.function_tool(
                code_interpreter.run_code,
                name_override="code_interpreter",
            )
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAMES["statistician"], openai_client=async_openai_client
        ),
    )

    # Visualizer
    visualizer = agents.Agent(
        name="VisualizerAgent",
        instructions=VISUALIZER_INSTRUCTIONS,
        tools=[
            agents.function_tool(
                code_interpreter.run_code,
                name_override="code_interpreter",
            )
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAMES["visualizer"], openai_client=async_openai_client
        ),
    )

    # Planner with structured output
    planner = agents.Agent(
        name="PlannerAgent",
        instructions=PLANNER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAMES["planner"], openai_client=async_openai_client
        ),
        output_type=AnalysisPlan,
    )

    # Synthesizer
    synthesizer = agents.Agent(
        name="SynthesizerAgent",
        instructions=SYNTHESIZER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAMES["synthesizer"], openai_client=async_openai_client
        ),
    )

    return {
        "planner": planner,
        "inspector": inspector,
        "statistician": statistician,
        "visualizer": visualizer,
        "synthesizer": synthesizer,
    }
