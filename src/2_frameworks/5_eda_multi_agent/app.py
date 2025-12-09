"""Multi-Agent System for Exploratory Data Analysis.

Architecture:
- Planner Agent: Breaks down EDA questions into specific analysis tasks
- Data Inspector Agent: Examines data structure, types, and basic statistics
- Statistical Analyst Agent: Performs statistical analysis and hypothesis testing
- Visualization Strategist Agent: Plans and executes data visualizations
- Insights Synthesizer Agent: Combines all findings into coherent narrative

Log traces to LangFuse for observability and evaluation.
"""

import asyncio
import contextlib
import json
import signal
import sys
from pathlib import Path

import agents
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.utils import (
    Configs,
    oai_agent_stream_to_gradio_messages,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client
from src.utils.tools.code_interpreter import CodeInterpreter

from agents_share import (
    AnalysisPlan,
    create_agents,
)

load_dotenv(verbose=True)
set_up_logging()

configs = Configs.from_env_var()
async_openai_client = AsyncOpenAI()

# Initialize code interpreter for data analysis
code_interpreter = CodeInterpreter(
    local_files=[
        Path("tests/tool_tests/example_files/final_unified_dataset_sample.csv"),
    ],
    timeout_seconds=60,
)

# Create all agents
agents_dict = create_agents(code_interpreter, async_openai_client)
planner_agent = agents_dict["planner"]
data_inspector_agent = agents_dict["inspector"]
statistician_agent = agents_dict["statistician"]
visualizer_agent = agents_dict["visualizer"]
synthesizer_agent = agents_dict["synthesizer"]


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown.
    
    Parameters
    ----------
    signum : int
        Signal number.
    frame : object
        Current stack frame.
    """
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


async def _run_planner(question: str) -> AnalysisPlan:
    """Create analysis plan using planner agent.
    
    Parameters
    ----------
    question : str
        User's EDA question.
    
    Returns
    -------
    AnalysisPlan
        Structured analysis plan with tasks and reasoning.
    """
    with langfuse_client.start_as_current_span(
        name="create_analysis_plan", input=question
    ) as planner_span:
        response = await agents.Runner.run(planner_agent, input=question)
        plan = response.final_output_as(AnalysisPlan)
        planner_span.update(output=plan)
    return plan


async def _run_data_inspection(dataset_path: str, tasks: list[str]) -> str:
    """Run data inspection tasks.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset CSV file.
    tasks : list[str]
        List of inspection tasks to perform.
    
    Returns
    -------
    str
        Data inspection findings.
    """
    with langfuse_client.start_as_current_span(
        name="data_inspection", input={"dataset": dataset_path, "tasks": tasks}
    ) as span:
        prompt = f"""
Analyze this dataset: {dataset_path}

Focus on these inspection tasks:
{json.dumps(tasks, indent=2)}

Load the data, examine it thoroughly, and provide detailed findings about:
- Data shape and structure
- Data types and missing values
- Summary statistics
- Data quality issues
- Distribution of key variables
"""
        response = await agents.Runner.run(data_inspector_agent, input=prompt)
        result = response.final_output
        span.update(output=result)
    return result


async def _run_statistical_analysis(dataset_path: str, tasks: list[str]) -> str:
    """Run statistical analysis tasks.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset CSV file.
    tasks : list[str]
        List of statistical analysis tasks to perform.
    
    Returns
    -------
    str
        Statistical analysis findings with test results.
    """
    with langfuse_client.start_as_current_span(
        name="statistical_analysis", input={"dataset": dataset_path, "tasks": tasks}
    ) as span:
        prompt = f"""
Perform statistical analysis on: {dataset_path}

Execute these analysis tasks:
{json.dumps(tasks, indent=2)}

Use the code interpreter to:
- Calculate correlations and dependencies
- Perform hypothesis tests
- Identify statistical significance
- Compute confidence intervals
- Detect anomalies and outliers

Provide detailed statistical findings with test statistics and p-values.
"""
        response = await agents.Runner.run(statistician_agent, input=prompt)
        result = response.final_output
        span.update(output=result)
    return result


async def _run_visualizations(dataset_path: str, tasks: list[str]) -> str:
    """Run visualization tasks.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset CSV file.
    tasks : list[str]
        List of visualization tasks to perform.
    
    Returns
    -------
    str
        Summary of visualizations created and their interpretations.
    """
    with langfuse_client.start_as_current_span(
        name="create_visualizations", input={"dataset": dataset_path, "tasks": tasks}
    ) as span:
        prompt = f"""
Create visualizations for: {dataset_path}

Create these visualizations:
{json.dumps(tasks, indent=2)}

Use the code interpreter to:
1. Load the data
2. Create appropriate plots for each task
3. Save all plots as PNG files in /tmp/plots/ directory
4. Provide interpretation of each visualization

Include proper labels, titles, and legends. Make plots publication-quality.
"""
        response = await agents.Runner.run(visualizer_agent, input=prompt)
        result = response.final_output
        span.update(output=result)
    return result


async def _run_synthesizer(
    question: str,
    inspection_results: str,
    statistical_results: str,
    visualization_results: str,
) -> str:
    """Synthesize all findings into coherent narrative.
    
    Parameters
    ----------
    question : str
        Original user's EDA question.
    inspection_results : str
        Data inspection findings.
    statistical_results : str
        Statistical analysis findings.
    visualization_results : str
        Visualization summary.
    
    Returns
    -------
    str
        Comprehensive EDA report with all findings synthesized.
    """
    with langfuse_client.start_as_current_span(
        name="synthesize_findings", input=question
    ) as span:
        prompt = f"""
Original Question: {question}

Here are all the analysis findings:

INSPECTION RESULTS:
{inspection_results}

STATISTICAL FINDINGS:
{statistical_results}

VISUALIZATION SUMMARY:
{visualization_results}

Please synthesize these findings into a comprehensive EDA report with:
1. Executive Summary
2. Data Overview
3. Key Findings and Insights
4. Statistical Significance
5. Important Patterns and Relationships
6. Anomalies and Data Quality Issues
7. Limitations and Caveats
8. Recommendations for Further Analysis

Make the report clear, actionable, and accessible to stakeholders.
"""
        response = await agents.Runner.run(synthesizer_agent, input=prompt)
        result = response.final_output
        span.update(output=result)
    return result


async def perform_eda(question: str, history: list[ChatMessage]):
    """Main EDA analysis orchestrator.
    
    This function orchestrates the multi-agent EDA system:
    1. Planner creates analysis strategy
    2. Data Inspector examines dataset
    3. Statistician performs analysis
    4. Visualizer creates charts
    5. Synthesizer creates final report
    
    Parameters
    ----------
    question : str
        User's EDA question.
    history : list[ChatMessage]
        Chat history for Gradio interface.
    
    Yields
    ------
    list[ChatMessage]
        Updated chat history with analysis progress and results.
    """
    dataset_path = "tests/tool_tests/example_files/final_unified_dataset_sample.csv"

    # Validate dataset exists
    if not Path(dataset_path).exists():
        yield history + [
            ChatMessage(
                role="assistant",
                content=f"❌ Error: Dataset not found at {dataset_path}",
            )
        ]
        return

    history.append(ChatMessage(role="user", content=question))
    yield history

    with langfuse_client.start_as_current_span(name="EDA-Multi-Agent-Trace") as root_span:
        root_span.update(input=question)

        # Step 1: Planner creates analysis strategy
        history.append(
            ChatMessage(
                role="assistant",
                content="📋 **Step 1/5: Creating EDA Strategy**\nAnalyzing your question and planning analysis tasks...",
            )
        )
        yield history

        plan = await _run_planner(question)

        history[-1] = ChatMessage(
            role="assistant",
            content=f"""📋 **EDA Strategy Created**

**Reasoning:** {plan.reasoning}

**Planned Tasks:**
- Inspection: {len(plan.inspection_tasks)} tasks
- Statistical Analysis: {len(plan.statistical_tasks)} tasks
- Visualizations: {len(plan.visualization_tasks)} tasks
- Pattern Detection: {len(plan.pattern_tasks)} tasks

Proceeding with analysis...""",
        )
        yield history

        # Step 2: Data Inspector examines dataset
        history.append(
            ChatMessage(
                role="assistant",
                content="🔍 **Step 2/5: Inspecting Data Structure**\nExamining dataset characteristics...",
            )
        )
        yield history

        inspection_results = await _run_data_inspection(
            dataset_path, plan.inspection_tasks + plan.pattern_tasks
        )

        history[-1] = ChatMessage(
            role="assistant",
            content=f"""🔍 **Data Inspection Complete**

{inspection_results[:1000]}...""",
        )
        yield history

        # Step 3: Statistical Analyst performs analysis
        history.append(
            ChatMessage(
                role="assistant",
                content="📊 **Step 3/5: Conducting Statistical Analysis**\nPerforming statistical tests and computing metrics...",
            )
        )
        yield history

        statistical_results = await _run_statistical_analysis(
            dataset_path, plan.statistical_tasks
        )

        history[-1] = ChatMessage(
            role="assistant",
            content=f"""📊 **Statistical Analysis Complete**

{statistical_results[:1000]}...""",
        )
        yield history

        # Step 4: Visualization Strategist creates plots
        history.append(
            ChatMessage(
                role="assistant",
                content="📈 **Step 4/5: Creating Visualizations**\nGenerating plots and charts...",
            )
        )
        yield history

        visualization_results = await _run_visualizations(
            dataset_path, plan.visualization_tasks
        )

        history[-1] = ChatMessage(
            role="assistant",
            content=f"""📈 **Visualizations Created**

{visualization_results[:1000]}...""",
        )
        yield history

        # Step 5: Synthesizer creates final report
        history.append(
            ChatMessage(
                role="assistant",
                content="✨ **Step 5/5: Synthesizing Findings**\nCombining all insights into comprehensive report...",
            )
        )
        yield history

        final_report = await _run_synthesizer(
            question, inspection_results, statistical_results, visualization_results
        )

        history[-1] = ChatMessage(
            role="assistant",
            content=f"""✨ **EDA Report**

{final_report}""",
        )
        root_span.update(output=final_report)
        yield history


# Create Gradio interface
demo = gr.ChatInterface(
    perform_eda,
    title="🔬 Multi-Agent Exploratory Data Analysis",
    type="messages",
    examples=[
        "What are the main characteristics and distributions of this dataset?",
        "Are there any significant correlations between numeric variables?",
        "What are the missing data patterns and potential data quality issues?",
        "What patterns and anomalies exist in the data? How should we investigate them?",
        "Can you provide a comprehensive overview suitable for stakeholders?",
    ],
)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
