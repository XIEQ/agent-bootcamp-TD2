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

import warnings
warnings.filterwarnings("ignore")

load_dotenv(verbose=True)

set_up_logging()


# CODE_INTERPRETER_INSTRUCTIONS = """\
# The `code_interpreter` tool executes SQL queries. \
# Your output is an SQL query for a SQLite database.
# You can access the local filesystem using this tool. \
# The data is in `/data/fintran.db`.  Any query from the user should use this file.

# Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.
# You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
# but you won't be able to install packages.

# """
CODE_INTERPRETER_INSTRUCTIONS = '''\
The `code_interpreter` tool executes SQL queries. \
Your output is an SQL query for a SQLite database.
You can access the local filesystem using this tool. \
The data is in `/data/fintran.db`.  Any query from the user should use this file.

Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.
You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.


Few Shot Examples:
Question: "What were the average transactions for spending on sports?"
SQL: sql SELECT AVG(t.transaction_amount) FROM transactions t JOIN mcc_codes m ON t.mcc = m.mcc_code WHERE m.description LIKE '% sport %' OR m.description LIKE 'Sport %' OR '% Sport';
Reason: The agent must find 'sport' but avoid matches like 'transport'. It must simulate whole-word matching using spaces inside the LIKE operator to ensure precision.


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
AGENT_LLM_NAME = "gemini-2.5-flash"
async_openai_client = AsyncOpenAI()
global main_count
import datetime

    
async def _main(question: str, gr_messages: list[ChatMessage]):
    print(f"============ UTC:{datetime.datetime.utcnow()} =============")

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

    with langfuse_client.start_as_current_span(name="TD2-EDA-Rajiv") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)

    print("============== gr_messages =================")
    pretty_print(gr_messages)

    yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="2.1 OAI Agent SDK ReAct + LangFuse Code Interpreter",
    type="messages",
    examples=[
        "how many users?",
        "compare spending of all males vs females. Which gender spends the most?",
        "How much did men spend on clothes?",
        "How much did women spend on sports"
    ],
)


if __name__ == "__main__":
    demo.launch(share=False)
