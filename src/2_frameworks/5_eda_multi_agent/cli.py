"""CLI version of Multi-Agent EDA System.

Provides a command-line interface to the multi-agent EDA system
for batch analysis without an interactive interface.
"""

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.utils import (
    Configs,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client
from src.utils.tools.code_interpreter import CodeInterpreter

from .agents_shared import create_agents

load_dotenv(verbose=True)
set_up_logging()

logger = logging.getLogger(__name__)

configs = Configs.from_env_var()
async_openai_client = AsyncOpenAI()

# Initialize code interpreter
code_interpreter = CodeInterpreter(
    local_files=[Path("tests/tool_tests/example_files/")],
    timeout_seconds=60,
)

# Create agents
agents_dict = create_agents(code_interpreter, async_openai_client)
planner_agent = agents_dict["planner"]
data_inspector_agent = agents_dict["inspector"]
statistician_agent = agents_dict["statistician"]
visualizer_agent = agents_dict["visualizer"]
synthesizer_agent = agents_dict["synthesizer"]


async def _run_planner(question: str):
    """Create analysis plan using planner agent.
    
    Parameters
    ----------
    question : str
        User's EDA question.
    
    Returns
    -------
    AnalysisPlan
        Structured analysis plan.
    """
    import agents
    
    with langfuse_client.start_as_current_span(
        name="create_analysis_plan", input=question
    ) as span:
        response = await agents.Runner.run(planner_agent, input=question)
        plan = response.final_output_as(response.final_output.__class__)
        span.update(output=plan)
    return plan


async def _run_data_inspection(dataset_path: str, tasks: list[str]) -> str:
    """Run data inspection tasks.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset.
    tasks : list[str]
        List of inspection tasks.
    
    Returns
    -------
    str
        Inspection results.
    """
    import agents
    import json
    
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
        Path to the dataset.
    tasks : list[str]
        List of statistical tasks.
    
    Returns
    -------
    str
        Statistical analysis results.
    """
    import agents
    import json
    
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
        Path to the dataset.
    tasks : list[str]
        List of visualization tasks.
    
    Returns
    -------
    str
        Visualization results.
    """
    import agents
    import json
    
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
        User's EDA question.
    inspection_results : str
        Data inspection findings.
    statistical_results : str
        Statistical analysis findings.
    visualization_results : str
        Visualization summary.
    
    Returns
    -------
    str
        Comprehensive EDA report.
    """
    import agents
    
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


async def _main(question: str):
    """Run EDA analysis from CLI.
    
    Parameters
    ----------
    question : str
        The EDA question to analyze.
    """
    dataset_path = "tests/tool_tests/example_files/final_unified_dataset_sample.csv"

    # Validate dataset
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    logger.info(f"Question: {question}")
    logger.info(f"Dataset: {dataset_path}")

    setup_langfuse_tracer()

    with langfuse_client.start_as_current_span(name="EDA-CLI-Analysis") as root_span:
        root_span.update(input=question)

        # Step 1: Plan
        logger.info("Step 1/5: Creating analysis strategy...")
        plan = await _run_planner(question)
        logger.info(f"Plan: {plan}")

        # Step 2: Inspect
        logger.info("Step 2/5: Inspecting data...")
        inspection_results = await _run_data_inspection(
            dataset_path, plan.inspection_tasks + plan.pattern_tasks
        )
        logger.info(f"Inspection results:\n{inspection_results}")

        # Step 3: Analyze
        logger.info("Step 3/5: Running statistical analysis...")
        statistical_results = await _run_statistical_analysis(
            dataset_path, plan.statistical_tasks
        )
        logger.info(f"Statistical results:\n{statistical_results}")

        # Step 4: Visualize
        logger.info("Step 4/5: Creating visualizations...")
        visualization_results = await _run_visualizations(
            dataset_path, plan.visualization_tasks
        )
        logger.info(f"Visualization results:\n{visualization_results}")

        # Step 5: Synthesize
        logger.info("Step 5/5: Synthesizing findings...")
        final_report = await _run_synthesizer(
            question, inspection_results, statistical_results, visualization_results
        )

        logger.info("=" * 80)
        logger.info("FINAL EDA REPORT")
        logger.info("=" * 80)
        logger.info(final_report)

        root_span.update(output=final_report)


if __name__ == "__main__":
    query = (
        "What are the main characteristics, distributions, and relationships "
        "in this dataset? Are there any anomalies or data quality issues?"
    )

    asyncio.run(_main(query))
