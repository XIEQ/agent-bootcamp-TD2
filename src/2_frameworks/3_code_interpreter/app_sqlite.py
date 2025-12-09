"""Code Interpreter example.

Logs traces to LangFuse for observability and evaluation.

You will need your E2B API Key.
"""

from pathlib import Path

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.eda_rajiv.finance_data_code_interpreter import FinanceDataCodeInterpreter
# from src.eda_rajiv import sql
from src import eda_rajiv
from src.utils import (
    CodeInterpreter,
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

set_up_logging()


CODE_INTERPRETER_INSTRUCTIONS = """\
The `code_interpreter` tool executes SQL queries. \
Your output is an SQL query for a SQLite database.
You can access the local filesystem using this tool. \
The data is in `/data/fintran.db`.  Any query from the user should use this file.

Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.
You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.

"""

AGENT_LLM_NAME = "gemini-2.5-flash"
async_openai_client = AsyncOpenAI()


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()
    init_module_path = Path(eda_rajiv.__file__).parent / "sql.py"
    code_interpreter = await FinanceDataCodeInterpreter.create(
                        init_module_path = str(init_module_path),

                        template_name="0v90rfl2s90xby53zujh",
                        timeout_seconds=300)
    main_agent = agents.Agent(
        name="Data Analysis Agent",
        instructions=CODE_INTERPRETER_INSTRUCTIONS,
        tools=[
            agents.function_tool(
                code_interpreter.run_query,
                name_override="code_interpreter",
            )
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME, openai_client=async_openai_client
        ),
    )

    with langfuse_client.start_as_current_span(name="Agents-SDK-Trace") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="2.1 OAI Agent SDK ReAct + LangFuse Code Interpreter",
    type="messages",
    examples=[
        "how many users?",
        "compare spending of all males vs females. Which gender spends the most?",
        "What is the sum of the column `y` in this example_a.csv?",
        "Create a linear best-fit line for the data in example_a.csv.",
    ],
)


if __name__ == "__main__":
    demo.launch(share=True)
