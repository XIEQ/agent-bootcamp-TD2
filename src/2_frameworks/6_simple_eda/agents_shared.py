"""
Shared agent definitions and orchestration for EDA system
"""

import agents
from openai import AsyncOpenAI

from src.utils.tools.code_interpreter import CodeInterpreter


AGENT_LLM_NAMES = {
    "planner": "gemini-2.5-pro", # more expensive, better at reasoning and planning
    "worker": "gemini-2.5-flash",  # less expensive,
}