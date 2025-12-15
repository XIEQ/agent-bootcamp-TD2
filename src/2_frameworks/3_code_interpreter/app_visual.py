"""Data Visualization Agent using Code Interpreter.

An AI agent that creates various types of graphs and charts based on
user-provided CSV files. Uses E2B sandbox for secure code execution
and LangFuse for observability.

You will need your E2B API Key.
"""

import base64
import json
import tempfile
from pathlib import Path

import agents
import gradio as gr
from agents.items import ToolCallOutputItem
from dotenv import load_dotenv
from e2b_code_interpreter import AsyncSandbox
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.utils import (
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client
from src.utils.tools.code_interpreter import (
    _upload_files,
    _enumerate_files,
    CodeInterpreterOutput,
    _CodeInterpreterOutputError,
)


load_dotenv(verbose=True)

set_up_logging()

# Store for generated images (temp file paths)
_image_store: list[str] = []


class VisualCodeInterpreterOutput(BaseModel):
    """Output from code interpreter with image support."""
    stdout: list[str]
    stderr: list[str]
    error: _CodeInterpreterOutputError | None = None
    images: list[str] = []  # base64 encoded images


class VisualCodeInterpreter:
    """Code Interpreter that captures and returns images from matplotlib."""

    def __init__(
        self,
        local_files=None,
        timeout_seconds: int = 60,
        template_name: str | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.local_files = []
        self.template_name = template_name

        if local_files:
            for _path in local_files:
                self.local_files.extend(_enumerate_files(_path))

    async def run_code(self, code: str) -> str:
        """Run Python code and capture any generated images."""
        sbx = await AsyncSandbox.create(
            timeout=self.timeout_seconds, template=self.template_name
        )
        await _upload_files(sbx, self.local_files)

        try:
            result = await sbx.run_code(
                code, on_error=lambda error: print(error.traceback)
            )

            # Parse stdout/stderr
            base_response = CodeInterpreterOutput.model_validate_json(
                result.logs.to_json()
            )

            # Extract images from results (E2B captures plt.show() output)
            images_base64 = []
            if result.results:
                for res in result.results:
                    # Check for PNG images
                    if hasattr(res, 'png') and res.png:
                        images_base64.append(res.png)
                    # Check for JPEG images
                    elif hasattr(res, 'jpeg') and res.jpeg:
                        images_base64.append(res.jpeg)

            response = VisualCodeInterpreterOutput(
                stdout=list(base_response.stdout),
                stderr=list(base_response.stderr),
                images=images_base64,
            )

            error = result.error
            if error is not None:
                response.error = _CodeInterpreterOutputError.model_validate_json(
                    error.to_json()
                )

            return response.model_dump_json()
        finally:
            await sbx.kill()

VISUALIZATION_AGENT_INSTRUCTIONS = """\
You are a data visualization expert agent. Your primary task is to help users \
create beautiful, informative graphs and charts from their CSV data.

The `code_interpreter` tool executes Python commands. \
Please note that data is not persisted. Each time you invoke this tool, \
you will need to run import and define all variables from scratch.

## Your Capabilities:
1. **Data Exploration**: First examine the CSV structure (columns, data types, sample rows)
2. **Chart Types**: Create various visualizations including:
   - Line charts (for time series, trends)
   - Bar charts (for categorical comparisons)
   - Scatter plots (for relationships between variables)
   - Histograms (for distribution analysis)
   - Pie charts (for proportion/composition)
   - Box plots (for statistical distribution)
   - Heatmaps (for correlation matrices)
   - Stacked/grouped bar charts
   - Area charts
   - Violin plots

## Workflow:
1. First, load and inspect the CSV file to understand its structure
2. Ask clarifying questions if the user's visualization request is ambiguous
3. Generate clean, well-labeled visualizations with:
   - Clear titles
   - Axis labels
   - Legends when appropriate
   - Appropriate color schemes
4. Display the plot using plt.show() - the image will be captured and shown automatically

## Code Guidelines:
- Use `pandas` for data manipulation
- Use `matplotlib` and/or `seaborn` for visualization
- Always include `plt.tight_layout()` before displaying
- Use `plt.show()` to display the plot (the image will be captured automatically)
- DO NOT use plt.savefig() - just use plt.show()
- Handle missing data appropriately (dropna or fillna as needed)

## Available Files:
You can access files in the local filesystem. Look for CSV files and explore \
their contents before creating visualizations.

Recommended packages: Pandas, Matplotlib, Seaborn, Numpy.

You can also run Jupyter-style shell commands (e.g., `!ls`) to explore files.
"""

AGENT_LLM_NAME = "gemini-2.5-flash"
async_openai_client = AsyncOpenAI()
code_interpreter = VisualCodeInterpreter(
    local_files=[
        Path("sandbox_content/"),
        Path("tests/tool_tests/example_files/final_unified_dataset_sample.csv"),
    ],
    timeout_seconds=60,  # Longer timeout for complex visualizations
)


def _extract_images_from_tool_output(item) -> list[ChatMessage]:
    """Extract base64 images from tool output and return as Gradio messages."""
    messages = []
    if isinstance(item, ToolCallOutputItem):
        try:
            output_str = item.raw_item.get("output", "{}")
            output_data = json.loads(output_str)
            images = output_data.get("images", [])

            for img_base64 in images:
                # Save base64 image to temp file for Gradio
                img_data = base64.b64decode(img_base64)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    f.write(img_data)
                    temp_path = f.name
                    _image_store.append(temp_path)

                # Use file path dict format for Gradio chat images
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content={"path": temp_path},
                    )
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error extracting images: {e}")

    return messages


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    visualization_agent = agents.Agent(
        name="Data Visualization Agent",
        instructions=VISUALIZATION_AGENT_INSTRUCTIONS,
        tools=[
            agents.function_tool(
                code_interpreter.run_code,
                name_override="code_interpreter",
            )
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME, openai_client=async_openai_client
        ),
    )

    with langfuse_client.start_as_current_span(name="Visualization-Agent-Trace") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(visualization_agent, input=question)
        async for _item in result_stream.stream_events():
            # Get standard messages
            new_messages = oai_agent_stream_to_gradio_messages(_item)
            gr_messages += new_messages

            # Check for images in tool outputs
            if hasattr(_item, 'item'):
                image_messages = _extract_images_from_tool_output(_item.item)
                gr_messages += image_messages

            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="📊 Data Visualization Agent",
    description="Upload CSV data and ask me to create visualizations! I can generate bar charts, line graphs, scatter plots, histograms, and more.",
    type="messages",
    examples=[
        # Data exploration
        "What columns are available in final_unified_dataset_sample.csv? Show me the first few rows.",
        # Bar chart examples
        "Create a bar chart showing the count of transactions by card_brand in final_unified_dataset_sample.csv",
        "Create a grouped bar chart comparing average transaction amount by card_type and card_brand",
        # Histogram/distribution
        "Create a histogram of the total_debt in final_unified_dataset_sample.csv",
        "Show me the distribution of credit_score with a histogram",
        # Pie chart
        "Create a pie chart showing the proportion of transactions by merchant_state (top 10 states)",
        # Scatter plot
        "Create a scatter plot of credit_score vs credit_limit colored by card_type",
        "Plot yearly_income vs total_debt as a scatter plot",
        # Box plot
        "Create a box plot comparing transaction amounts across different card_brands",
        # Heatmap
        "Create a correlation heatmap for the numerical columns (amount, credit_limit, credit_score, yearly_income, total_debt)",
        # Time series
        "Create a line chart showing the number of transactions over time (by month/year)",
        # Complex analysis
        "Analyze the relationship between age groups and average transaction amount with a bar chart",
    ],
)


if __name__ == "__main__":
    demo.launch(share=True)
