"""Code Interpreter example.

Logs traces to LangFuse for observability and evaluation.

You will need your E2B API Key.
"""
import sys
import asyncio
import contextlib
import signal

from pathlib import Path

# import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI

from src.eda_rajiv.finance_data_code_interpreter import  make_python_code_interpreter, make_fintran_db_code_interpreter
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

import warnings
warnings.filterwarnings("ignore")

load_dotenv(verbose=True)

set_up_logging()

def langchain_chunk_to_gradio_messages(data):
    """Convert LangGraph chunk data to Gradio ChatMessage format."""
    messages = []
    for msg in data.get('messages', []):
        if hasattr(msg, 'content'):
            content = msg.content
            # Handle multimodal content (list of blocks)
            if isinstance(content, list):
                content = ' '.join([
                    block.get('text', '') if isinstance(block, dict) else str(block)
                    for block in content
                ])
            # Map LangChain message type to Gradio role
            role_map = {
                'human': 'user',
                'ai': 'assistant',
                'system': 'system',
                'tool': 'assistant'
            }
            role = role_map.get(msg.type, 'assistant')
            messages.append(ChatMessage(role=role, content=content))
    return messages

COMMON_DATABASE_INFORMATION = '''
The data is in `/data/fintran.db`.  Any query from the user should use this file.

Example Tablular Data:
table: users_data
columns: id,current_age,retirement_age,birth_year,birth_month,gender,address,latitude,longitude,per_capita_income,yearly_income,total_debt,credit_score,num_credit_cards
data:
825,53,66,1966,11,Female, Some Address 1,34.15,-117.76,$29278,$59696,$127613,787,5
1164,43,70,1976,9,Male,,37.76,-122.44,$53797,$109687,$183855,675,1


The schema is below. Note it's SQL Create table commands but enclosed in Python variables for my convenience.
Extract the SQL Create table commands for your use.

CARDS_DATA_TABLE = """
CREATE TABLE cards_data (
    id INTEGER PRIMARY KEY,
    client_id INTEGER,
    card_brand TEXT,
    card_type TEXT,
    card_number TEXT UNIQUE,
    expires_month INT,
    expires_year INT,
    cvv INTEGER,
    has_chip TEXT,
    num_cards_issued INTEGER,
    credit_limit FLOAT,  -- Stored as TEXT due to '$' sign
    acct_open_month INTEGER,
    acct_open_year INTEGER,
    year_pin_last_changed INTEGER,
    card_on_dark_web TEXT,
    FOREIGN KEY (client_id) REFERENCES users_data(client_id)
);
"""



MCC_CODES_TABLE = """
CREATE TABLE mcc_codes (
id TEXT PRIMARY KEY,
description TEXT
);
"""
TRAIN_FRAUD_LABELS_TABLE = """
CREATE TABLE train_fraud_labels (
id TEXT PRIMARY KEY,
label TEXT
);
"""
USERS_DATA_TABLE = """
CREATE TABLE users_data (
    id INTEGER PRIMARY KEY,
    current_age INTEGER,
    retirement_age INTEGER,
    birth_year INTEGER,
    birth_month INTEGER,
    gender TEXT,
    address TEXT,
    latitude REAL,
    longitude REAL,
    per_capita_income FLOAT,
    yearly_income FLOAT,
    total_debt FLOAT,
    credit_score INTEGER,
    num_credit_cards INTEGER
);
"""
TRANSACTIONS_DATA_TABLE = """
CREATE TABLE transactions_data (
    id INTEGER PRIMARY KEY,
    date DATETIME,
    client_id INTEGER,
    card_id INTEGER,
    amount FLOAT,
    use_chip TEXT,
    merchant_id INTEGER,
    merchant_city TEXT,
    merchant_state TEXT,
    zip TEXT,
    mcc TEXT,
    errors TEXT,
    FOREIGN KEY (client_id) REFERENCES users_data(client_id),
    FOREIGN KEY (card_id) REFERENCES cards_data(id),
    FOREIGN KEY (mcc) REFERENCES mcc_codes(id)
);
"""
'''

SQL_CODE_INTERPRETER_INSTRUCTIONS = f'''\
The `code_interpreter` tool executes SQL queries. \
Your output is just an SQL query for a SQLite database.
You can access the local filesystem using this tool. \

Few Shot Examples:
Question: "What were the average transactions for spending on sports?"
SQL: sql SELECT AVG(t.transaction_amount) FROM transactions t JOIN mcc_codes m ON t.mcc = m.mcc_code WHERE m.description LIKE '% sport %' OR m.description LIKE 'Sport %' OR '% Sport';
Reason: The agent must find 'sport' but avoid matches like 'transport'. It must simulate whole-word matching using spaces inside the LIKE operator to ensure precision.

{COMMON_DATABASE_INFORMATION}

'''

PYTHON_CODE_INTERPRETER_INSTRUCTIONS = f"""\
The `code_interpreter` tool executes Python commands. \

You can access the local filesystem using this tool. \
Instead of asking the user for file inputs, you should try to find the file \
using this tool.

Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.

You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.

{COMMON_DATABASE_INFORMATION}
"""


MAIN_CODE_INTERPRETER_INSTRUCTIONS = f'''\
You are the central Data Analyst Agent. Your role is to plan, route, and synthesize results.
Your decision-making process must strictly follow these rules:

1. CORE QUERY CLASSIFICATION:
    - The `agent_sql` tool executes SQL queries. \
    - The `agent_python` executes Python queries.     

2. ACTION SEQUENCE:
    - If you feel that the question is complex for the sql tool, then you can use the Python tool to understand more about the data and refine your SQL query or just use it's output if you feel the sql query tool may not be able to return accurate information.


3. SCHEMA CONTEXT:
{COMMON_DATABASE_INFORMATION}

4. FINAL RESPONSE:
    When you are satisfied with the answer, always use the tool output to construct a concise, natural language final answer.

"""

'''

MAIN_AGENT_LLM_NAME = "gemini-2.5-pro"
AGENT_LLM_NAME = "gemini-2.5-flash"
AGENT_LLM_NAMES = {
    "worker": AGENT_LLM_NAME,  # less expensive,
    "planner": MAIN_AGENT_LLM_NAME,  # more expensive, better at reasoning and planning
}

async def _main(question: str, gr_messages: list[ChatMessage]):

    from langchain.agents import create_agent

    from langchain.chat_models import init_chat_model

    setup_langfuse_tracer()

    init_module_path = Path(eda_rajiv.__file__).parent / "sql.py"
    python_code_interpreter = await make_python_code_interpreter()
    code_interpreter = await make_fintran_db_code_interpreter(init_module_path=init_module_path) 

    from langchain_core.tools import Tool

    model = init_chat_model(f"google_genai:{AGENT_LLM_NAMES['worker']}")
    python_agent = create_agent(
        model,
        tools=[
                Tool.from_function(
                    func=python_code_interpreter.run_code,
                    name="code_interpreter",
                    description="Execute Python code and return the output. Useful for running scripts, calculations, or data analysis."
                )],
        system_prompt=PYTHON_CODE_INTERPRETER_INSTRUCTIONS,
        name="agent_python"
    )


    sql_agent = create_agent(
        model,
        tools=[
                Tool.from_function(
                    func=code_interpreter.run_query,
                    name="code_interpreter",
                    description="Execute SQL code and return the output. Useful for running scripts, calculations, or data analysis."
                )],
        system_prompt=SQL_CODE_INTERPRETER_INSTRUCTIONS,
        name="agent_sql"
    )


    # main_agent = agents.Agent(
    #     name="agent_main",
    #     instructions=MAIN_CODE_INTERPRETER_INSTRUCTIONS,
    #     tools=[
    #         python_agent.as_tool(
    #             tool_name="agent_python",
    #             tool_description="Use this Python Code Intepreter Agent to execute Python"),
    #         sql_agent.as_tool(
    #             tool_name="agent_sql",
    #             tool_description="Use this SQLite Code Intepreter Agent to execute SQL"),

    #     ],
    #     model=agents.OpenAIChatCompletionsModel(
    #         model=AGENT_LLM_NAMES["planner"], openai_client=async_openai_client
    #     ),
    # )
    # ===========
    # @tool(
    #     "subagent1_name",
    #     description="subagent1_description"
    # )
    # def call_subagent1(query: str):
    #     result = subagent1.invoke({
    #         "messages": [{"role": "user", "content": query}]
    #     })
    #     return result["messages"][-1].content

    # agent = create_agent(model="...", tools=[call_subagent1])
    from langchain.tools import tool
    @tool(
        "agent_python",
        description="Use this Python Code Intepreter Agent to execute Python"
    )
    def call_agent_python(query: str):
        result = python_agent.invoke( {
            "messages": [{"role": "user", "content": query}]

        })
        return result["messages"][-1].content

    @tool(
        "agent_sql",
        description="Use this SQLite Code Intepreter Agent to execute SQL"
    )
    def call_agent_sql(query: str):
        result = sql_agent.invoke( {
            "messages": [{"role": "user", "content": query}]

        })
        return result["messages"][-1].content

    #  main_agent = agents.Agent(
    #         name="agent_main",
    #         instructions=MAIN_CODE_INTERPRETER_INSTRUCTIONS,
    #         tools=[call_agent_python, call_agent_sql ],
    #         model=agents.OpenAIChatCompletionsModel(
    #             model=AGENT_LLM_NAMES["planner"], openai_client=async_openai_client
    #         ),
    #     )



    planner_model = init_chat_model(f"google_genai:{AGENT_LLM_NAMES['planner']}")
    main_agent = create_agent(
        planner_model,
        tools=[ call_agent_python, call_agent_sql ],
        system_prompt=MAIN_CODE_INTERPRETER_INSTRUCTIONS,
        name="agent_main"
    )


    with langfuse_client.start_as_current_span(name="TD2-EDA-Rajiv-LangChain") as span:
        span.update(input=question)

        # result_stream = agents.Runner.run_streamed(main_agent, input=question)
        # async for _item in result_stream.stream_events():
        #     gr_messages += oai_agent_stream_to_gradio_messages(_item)
        #     if len(gr_messages) > 0:
        #         yield gr_messages
        for chunk in main_agent.stream({"messages": [{"role": "user", "content": question}]},
            stream_mode="updates",
        ):
            for step, data in chunk.items():
                print(f"step: {step}")
                print(f"content: {data['messages'][-1].content_blocks}")
                gr_messages.extend(langchain_chunk_to_gradio_messages(data))

        span.update(output=gr_messages[-1].content if gr_messages and gr_messages[-1].role == "assistant" else "")

    # pretty_print(gr_messages)

    yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="SQLite Financial Transactions EDA",
    type="messages",
    examples=[
        "how many users?",
        "compare spending of all males vs females. Which gender spends the most?",
        "How much did men spend on clothes?",
        "How much did women spend on sports",
        "What is 99th percentile of purchases done during Xmas?"
    ],
)


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    demo.launch(share=False)
