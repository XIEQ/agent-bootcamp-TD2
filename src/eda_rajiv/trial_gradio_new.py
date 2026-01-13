"""
LangChain V1 + Gradio Chat Integration

This module demonstrates:
- Streaming LangChain agent responses to Gradio chat
- Transforming LangChain messages to Gradio format
- Agent with memory (InMemorySaver)
- Avoiding duplicate messages in the chat
"""

import os
import asyncio
import json

os.environ["GOOGLE_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

import gradio as gr
from gradio.components.chatbot import ChatMessage
from pathlib import Path

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.utils.tools.gemini_grounding import GeminiGroundingWithGoogleSearch


# Global state
_agent = None
_code_interpreter = None
_shown_message_ids: set = set()  # Track ALL message IDs ever shown to user

AGENT_LLM_NAME = "gemini-2.5-pro"


async def setup_agent():
    """Set up the real LangChain agent with tools."""
    from src.eda_rajiv.app_sqlite import CODE_INTERPRETER_INSTRUCTIONS
    from src import eda_rajiv
    from src.eda_rajiv.finance_data_code_interpreter import (
        make_fintran_db_code_interpreter,
    )

    model = init_chat_model(f"google_genai:{AGENT_LLM_NAME}")

    tool_cls = GeminiGroundingWithGoogleSearch()
    init_module_path = Path(eda_rajiv.__file__).parent / "sql.py"

    a_code_interpreter = await make_fintran_db_code_interpreter(
        init_module_path=init_module_path, timeout=10 * 60
    )

    @tool
    async def code_interpreter(query: str) -> str:
        """Run the SQL Query in a sandbox and return a JSON string of the stdout and stderr"""
        return await a_code_interpreter.run_query(query)

    @tool
    async def web_search(query: str) -> str:
        """Run a Web Query and get an English Response"""
        response = await tool_cls.get_web_search_grounded_response(query)
        return response.text_with_citations

    agent = create_agent(
        model,
        tools=[code_interpreter, web_search],
        system_prompt=CODE_INTERPRETER_INSTRUCTIONS,
        checkpointer=InMemorySaver(),
    )

    return agent, a_code_interpreter


async def get_agent():
    """Get or create the LangChain agent."""
    global _agent, _code_interpreter

    if _agent is None:
        _agent, _code_interpreter = await setup_agent()

    return _agent




# if __name__ == "__main__":
#     demo = gr.ChatInterface(
#         chat_with_agent,
#         title="SQLite Financial Transactions EDA",
#         description="A LangChain agent with SQL code interpreter and web search capabilities.",
#         examples=[
#             "how many users?",
#             "compare spending of all males vs females. Which gender spends the most?",
#             "How much did men spend on clothes?",
#             "How much did women spend on sports?",
#             "What is 99th percentile of purchases done during Xmas?"
#         ],
#     )

#     demo.launch(share=False)


# ============================ SS ===========
import os
import json

os.environ["GOOGLE_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

import gradio as gr
from gradio.components.chatbot import ChatMessage

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver


# @tool
# def get_weather(city: str) -> str:
#     """Get the current weather for a city."""
#     weather_data = {
#         "new york": {"temp": 72, "condition": "sunny"},
#         "london": {"temp": 58, "condition": "cloudy"},
#         "tokyo": {"temp": 68, "condition": "rainy"},
#     }
#     data = weather_data.get(city.lower(), {"temp": 65, "condition": "unknown"})
#     return json.dumps(data)


# @tool
# def calculate(expression: str) -> str:
#     """Evaluate a math expression."""
#     try:
#         result = eval(expression)
#         return str(result)
#     except Exception as e:
#         return f"Error: {e}"


# Global state
_agent = None

AGENT_LLM_NAME = "gemini-2.5-pro"


# def setup_agent():
#     """Set up the LangChain agent with simple tools."""
#     model = init_chat_model(f"google_genai:{AGENT_LLM_NAME}")

#     agent = create_agent(
#         model,
#         tools=[get_weather, calculate],
#         system_prompt="You are a helpful assistant with access to weather and calculator tools.",
#         checkpointer=InMemorySaver(),
#     )
#     return agent


# def get_agent():
#     """Get or create the LangChain agent."""
#     global _agent
#     if _agent is None:
#         _agent = setup_agent()
#     return _agent


def langchain_message_to_gradio(message, seen_ids: set) -> list[ChatMessage]:
    """Transform a LangChain message to Gradio ChatMessage(s)."""
    messages = []

    if isinstance(message, HumanMessage):
        return messages

    msg_id = getattr(message, "id", None)
    if msg_id and msg_id in seen_ids:
        return messages
    if msg_id:
        seen_ids.add(msg_id)

    if isinstance(message, AIMessage):
        content = message.content
        if isinstance(content, list):
            text_parts = [
                block.get("text", "") if isinstance(block, dict) else block
                for block in content
                if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
            ]
            content = "\n".join(text_parts)

        if content:
            messages.append(ChatMessage(role="assistant", content=content))

        tool_calls = getattr(message, "tool_calls", [])
        for tc in tool_calls:
            tool_name = tc.get("name", "tool")
            tool_args = tc.get("args", {})
            thinking_content = f"**Calling:** `{tool_name}`\n```json\n{json.dumps(tool_args, indent=2)}\n```"
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=thinking_content,
                    metadata={"title": f"Using {tool_name}"},
                )
            )

    elif isinstance(message, ToolMessage):
        tool_name = getattr(message, "name", "tool")
        content = message.content
        try:
            parsed = json.loads(content)
            content = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass

        messages.append(
            ChatMessage(
                role="assistant",
                content=f"```json\n{content}\n```",
                metadata={"title": f"Result from {tool_name}"},
            )
        )

    return messages


_shown_message_ids: set = set()


async def chat_with_agent(message: str, history: list[ChatMessage]) -> list[ChatMessage]:
    """
    Send a message to the agent and stream back only new messages.
    Uses stream_mode="messages" to avoid receiving full history.
    Yields only NEW messages - Gradio appends these to existing history.
    """
    global _shown_message_ids

    agent = await get_agent()
    thread_id = "gradio-session-1"
    config = {"configurable": {"thread_id": thread_id}}

    # Only accumulate NEW messages - Gradio manages history automatically
    new_gr_messages = []

    # Track accumulated content per message ID for streaming chunks
    accumulated_content: dict[str, str] = {}
    # Map message ID to index in new_gr_messages for updating
    msg_id_to_index: dict[str, int] = {}

    try:
        async for msg, metadata in agent.astream(
            {"messages": [HumanMessage(content=message)]},
            config,
            stream_mode="messages",
        ):
            # Skip HumanMessage
            if isinstance(msg, HumanMessage):
                continue

            msg_id = getattr(msg, "id", None)

            if isinstance(msg, (AIMessage, AIMessageChunk)):
                # Extract content from the chunk
                content = msg.content
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "") if isinstance(block, dict) else block
                        for block in content
                        if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
                    ]
                    content = "".join(text_parts)

                if content and msg_id:
                    if msg_id in accumulated_content:
                        # Append to existing accumulated content
                        accumulated_content[msg_id] += content
                        # Update the existing message in the list
                        idx = msg_id_to_index[msg_id]
                        new_gr_messages[idx] = ChatMessage(
                            role="assistant",
                            content=accumulated_content[msg_id]
                        )
                    else:
                        # First chunk for this message ID
                        accumulated_content[msg_id] = content
                        msg_id_to_index[msg_id] = len(new_gr_messages)
                        new_gr_messages.append(ChatMessage(role="assistant", content=content))

                    yield new_gr_messages

                # Handle tool calls (these come complete, not streamed)
                tool_calls = getattr(msg, "tool_calls", [])
                for tc in tool_calls:
                    tool_name = tc.get("name", "tool")
                    tool_args = tc.get("args", {})
                    thinking_content = f"**Calling:** `{tool_name}`\n```json\n{json.dumps(tool_args, indent=2)}\n```"
                    new_gr_messages.append(
                        ChatMessage(
                            role="assistant",
                            content=thinking_content,
                            metadata={"title": f"Using {tool_name}"},
                        )
                    )
                    yield new_gr_messages

            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                content = msg.content
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass

                new_gr_messages.append(
                    ChatMessage(
                        role="assistant",
                        content=f"```json\n{content}\n```",
                        metadata={"title": f"Result from {tool_name}"},
                    )
                )
                yield new_gr_messages

    except Exception as e:
        new_gr_messages.append(ChatMessage(role="assistant", content=f"Error: {e}"))
        yield new_gr_messages


def clear_chat():
    """Clear chat and reset agent for fresh conversation."""
    global _agent, _shown_message_ids
    _agent = None
    _shown_message_ids = set()
    return []


if __name__ == "__main__":
    demo = gr.ChatInterface(
        chat_with_agent,
        title="Simple LangChain Agent",
        description="",
        examples=[
            "how many users?",
            "compare spending of all males vs females. Which gender spends the most?",
            "How much did men spend on clothes?",
            "How much did women spend on sports?",
            "What is 99th percentile of purchases done during Xmas?"

        ],
    )

    demo.launch(share=False)


