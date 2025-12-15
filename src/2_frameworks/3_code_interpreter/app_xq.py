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
The `code_interpreter` tool executes Python commands. \
Please note that data is not persisted. Each time you invoke this tool, \
you will need to run import and define all variables from scratch.

You can access the local filesystem using this tool. \
Instead of asking the user for file inputs, you should try to find the file \
using this tool.

The dataset is cleanded_dataset.csv. \
The metadata is cleaned_metadata.json. \

Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.

You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.
"""

CODE_INTERPRETER_INSTRUCTIONS1 = """\
The `code_interpreter` tool executes Python commands. \
Please note that data is not persisted. Each time you invoke this tool, \
you will need to run import and define all variables from scratch.

You can access the local filesystem using this tool. \
Instead of asking the user for file inputs, you should try to find the file \
using this tool.

The dataset is cleanded_dataset.csv. \
The metadata is cleaned_metadata.json. \

Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.

You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.
"""

CODE_INTERPRETER_INSTRUCTIONS2 = """
The `code_interpreter` tool executes Python commands. 
Please note that data is not persisted. Each time you invoke this tool, 
you will need to run import and define all variables from scratch.

You can access the local filesystem using this tool. 
Instead of asking the user for file inputs, you should try to find the file 
using this tool.

Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.

You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.

You are a Data Analyst, the dataset you will analyze is cleanded_dataset.csv. 

The following is the first 5 rows with header of the dataset cleanded_dataset.csv. Do not make up column name based on user question. Get the dataset schema either by loading the dataset or by using the 5 sample rows with header provided below.

transaction_id,transaction_date,client_id,card_id,amount,use_chip,merchant_id,merchant_city,merchant_state,zip,mcc,errors,card_brand,card_type,card_number,expires,cvv,has_chip,num_cards_issued,credit_limit,acct_open_date,year_pin_last_changed,card_on_dark_web,current_age,retirement_age,birth_year,birth_month,gender,address,latitude,longitude,per_capita_income,yearly_income,total_debt,credit_score,num_credit_cards,mcc_description,fraud_label,amount_len
11978328,2012-11-23 16:03:00,619,3348,263.43,Swipe Transaction,54850,Austin,MN,55912.0,4814,,Mastercard,Debit,5531418228277739,06/2021,20,YES,2,33319,06/2007,2010,No,54,65,1965,12,Male,498 Littlewood Avenue,44.01,-92.47,26478,53986,58381,748,4,Telecommunication Services,No,7
11363233,2012-07-07 13:57:00,456,4576,38.26,Swipe Transaction,68135,Cape Coral,FL,33909.0,5411,,Mastercard,Debit,5915761072331247,08/2020,754,YES,2,12594,09/2005,2014,No,54,63,1965,10,Male,600 Grant Lane,26.63,-81.99,17140,34947,49024,751,3,"Grocery Stores, Supermarkets",No,6
8117710,2010-06-10 16:59:00,209,4676,52.57,Swipe Transaction,81833,El Paso,TX,79928.0,5912,,Mastercard,Debit,5607093669748051,12/2021,968,YES,2,2507,09/2008,2010,No,61,67,1959,1,Female,2111 Burns Street,31.65,-106.15,14322,29206,25966,716,6,Drug Stores and Pharmacies,,6
12606562,2013-04-13 12:08:00,1605,1133,40.0,Swipe Transaction,27092,Amelia,OH,45102.0,4829,,Visa,Debit,4607163659779577,01/2024,954,YES,1,18358,02/2007,2010,No,39,67,1980,5,Male,9995 Pine Avenue,39.5,-84.37,19293,39336,43747,690,3,Money Transfer,No,6
12628171,2013-04-18 10:00:00,144,5247,4.58,Swipe Transaction,44578,Arkadelphia,AR,71923.0,5812,,Mastercard,Debit,5891798640440029,07/2022,370,YES,2,20289,10/2011,2011,No,44,67,1975,3,Female,5380 12th Boulevard,34.5,-93.05,15857,32330,97190,835,4,Eating Places and Restaurants,,5
 
"""

CODE_INTERPRETER_INSTRUCTIONS3 = """\
The `code_interpreter` tool executes Python commands. \
Please note that data is not persisted. Each time you invoke this tool, \
you will need to run import and define all variables from scratch.

You can access the local filesystem using this tool. \
Instead of asking the user for file inputs, you should try to find the file \
using this tool.

You are a Data Analyst, the dataset you will analyze is cleaned_dataset.csv. \

Do not make up column name based on user question. Get the dataset schema by loading the dataset.\


Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.

You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.
"""


AGENT_LLM_NAME = "gemini-2.5-flash"
async_openai_client = AsyncOpenAI()
code_interpreter = CodeInterpreter(
    local_files=[
        Path("sandbox_content/"),
        # Path("tests/tool_tests/example_files/example_a.csv"),
        Path("tests/tool_tests/example_files/cleaned_dataset.csv"),
        Path("tests/tool_tests/example_files/cleaned_metadata.json")
    ]
)


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    main_agent = agents.Agent(
        name="Data Analysis Agent",
        instructions=CODE_INTERPRETER_INSTRUCTIONS3,
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

    with langfuse_client.start_as_current_span(name="EDA-Agent-Qian") as span:
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
    #examples=[
    #    "What is the sum of the column `x` in this example_a.csv?",
    #    "What is the sum of the column `y` in this example_a.csv?",
    #    "Create a linear best-fit line for the data in example_a.csv.",
    #],
    examples=[
        "What is the unique number of client_id in the cleaned_dataset.csv?",
        "What is the unique number of card_id in the cleaned_dataset.csv?",
        "Which client_id spent the most amount in the cleaned_dataset.csv?",
        "What is the average amount spent per transaction in the cleaned_dataset.csv?",
    ],
)


if __name__ == "__main__":
    demo.launch(share=True)
